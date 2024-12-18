from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiohttp
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    UploadFile,
    status,
)
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
from tesseract_wrapper import TesseractWrapper

from configs.ocr_config import OCRConfig
from database import (
    DocumentCreate,
    DocumentError,
    DocumentRepository,
    get_repository,
)
from utils.utils import get_unique_filename

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from database.models import Document

config = OCRConfig()
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.DEBUG if config.LOG_LEVEL == "DEBUG" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            config.LOG_DIR / f"{datetime.now(tz=datetime.UTC):%Y-%m-%d}.log",
            mode="a",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI()
# optimizer = EasyOCRWrapper(config=config)
optimizer = TesseractWrapper(config=config)


class OCRProcessor:
    """Handle OCR processing operations."""

    def __init__(self, document_repository: DocumentRepository) -> None:
        """Initialize OCR processor with repository."""
        self.repository = document_repository

    async def validate_file(self, file: UploadFile) -> str:
        """Validate uploaded file against constraints."""
        logger.info("Validating file: %s", file.filename)
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )
        new_filename = file.filename

        if await self.repository.get_by_filename(
            file.filename,
            return_bool=True,
        ):
            logger.info(
                "File already exists: %s, making it unique",
                file.filename,
            )
            new_filename = await get_unique_filename(
                file.filename,
                self.repository,
            )
            logger.info("New unique filename: %s", new_filename)

        if file.content_type not in config.ACCEPTED_FILE_TYPES:
            logger.error("Unsupported file type: %s", file.content_type)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type",
            )

        if not file.size:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file",
            )

        if file.size > config.FILE_SIZE_LIMIT:
            logger.error("File size too large: %s bytes", file.size)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size too large",
            )

        if file.content_type == "application/pdf":
            try:
                page_count = len(convert_from_bytes(file.file.read()))
                if page_count > config.MAX_PDF_PAGES:
                    logger.error("PDF has too many pages: %d", page_count)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"PDF exceeds {config.MAX_PDF_PAGES} pages",
                    )
            except PDFPageCountError as err:
                logger.exception("PDF processing error")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted PDF file",
                ) from err
            finally:
                file.file.seek(0)

        return new_filename

    @staticmethod
    def convert_file_to_images(file: UploadFile) -> list[Image.Image]:
        """Convert uploaded file to list of PIL Images."""
        logger.info("Converting file to images: %s", file.filename)

        if file.content_type == "application/pdf":
            return convert_from_bytes(file.file.read(), fmt="jpeg")

        return [Image.open(file.file)]

    @staticmethod
    def convert_to_base64(images: list[Image.Image]) -> list[str]:
        """Convert PIL Images to base64 strings."""
        encoded_images = []

        for img in images:
            converted_img = img
            if converted_img.mode in ("RGBA", "P"):
                converted_img = converted_img.convert("RGB")

            with io.BytesIO() as buffer:
                converted_img.save(buffer, format="JPEG")
                img_byte_arr = buffer.getvalue()

                encoded = base64.b64encode(img_byte_arr).decode("utf-8")
                encoded_images.append(encoded)

        return encoded_images

    async def save_document(self, file_name: str) -> Document:
        """Save document metadata to database."""
        try:
            return await self.repository.create(
                DocumentCreate(
                    file_name=file_name,
                    file_path=str(config.UPLOAD_DIR / file_name),
                    classification="",
                    summary="",
                ),
            )
        except DocumentError as err:
            logger.exception("Failed to save document")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save document",
            ) from err


@app.get("/documents")
async def get_docs(
    repository: AsyncGenerator[DocumentRepository, None] = Depends(
        get_repository,
    ),
) -> list[dict]:
    """Get all documents."""
    try:
        documents = await repository.get_all()
        return [doc.to_dict() for doc in documents]
    except DocumentError as e:
        logger.exception("Failed to retrieve documents: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        ) from e


@app.post("/ocr")
async def process_document(
    file: UploadFile,
    repository: AsyncGenerator[DocumentRepository, None] = Depends(
        get_repository,
    ),
) -> dict[str, str]:
    """Process document for OCR and forward results."""
    logger.info("Processing document: %s", file.filename)

    processor = OCRProcessor(repository)
    document_id = None
    output_file = None

    try:
        # Validate and process file
        new_filename = await processor.validate_file(file)

        if new_filename != file.filename:
            file.filename = new_filename

        output_file = Path(config.UPLOAD_DIR / new_filename)

        async with aiofiles.open(output_file, "wb") as out_file:
            while content := await file.read(1024):  # async read chunk
                await out_file.write(content)  # async write chunk

        file.file.seek(0)  # Reset file pointer

        images = processor.convert_file_to_images(file)

        # Perform OCR
        ocr_response, preprocessed_images = optimizer.process_batch(images)
        encoded_images = processor.convert_to_base64(preprocessed_images)

        # Save initial document record
        document = await processor.save_document(file.filename)
        document_id = document.id

        # Forward to processor service
        try:
            timeout = aiohttp.ClientTimeout(total=480)
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    config.PROCESSOR_URL,
                    json={
                        "ocr_result": ocr_response.model_dump(
                            exclude={"page_count"},
                        )["results"],
                        "images": encoded_images,
                        "file_name": file.filename,
                    },
                    timeout=timeout,
                ) as response,
            ):
                response.raise_for_status()
                return await response.json()

        except (aiohttp.ClientError, HTTPException) as e:
            await cleanup(repository, document_id, output_file)
            logger.exception("Downstream processing failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Document processing failed in downstream service",
            ) from e

    except Exception as e:
        await cleanup(repository, document_id, output_file)
        logger.exception("Processing error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        ) from e


async def cleanup(
    repository: DocumentRepository,
    document_id: int | None,
    output_file: Path | None,
) -> None:
    """Clean up resources on error."""
    if document_id:
        try:
            await repository.delete(document_id)
        except DocumentError as e:
            logger.exception("Failed to delete document: %s", str(e))

    if output_file and output_file.exists():
        try:
            output_file.unlink()
        except OSError as e:
            logger.exception("Failed to delete file: %s", str(e))
