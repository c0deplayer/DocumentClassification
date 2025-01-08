from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiofiles
import aiohttp
from dotenv import load_dotenv
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
from watcher import DocumentWatcher

from configs.ocr_config import OCRConfig
from database import (
    DocumentCreate,
    DocumentError,
    DocumentRepository,
    get_repository,
)
from encryption.aes import AESCipher
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
            config.LOG_DIR / f"{datetime.now():%Y-%m-%d}.log",
            mode="a",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
load_dotenv()
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    err_msg = "Missing ENCRYPTION_KEY"
    raise ValueError(err_msg)

KEY = base64.b64decode(ENCRYPTION_KEY)


cipher = AESCipher(KEY)

# Add watch directory to config
config.WATCH_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the lifespan of the FastAPI application."""
    service_url = "http://localhost:8080"
    app.state.document_watcher = DocumentWatcher(config.WATCH_DIR, service_url)
    task = asyncio.create_task(app.state.document_watcher.start())
    logger.info("Started document watcher for directory: %s", config.WATCH_DIR)

    yield  # Startup finished, now wait for shutdown

    if hasattr(app.state, "document_watcher"):
        await app.state.document_watcher.stop()
        logger.info("Stopped document watcher")
    task.cancel()


app = FastAPI(lifespan=lifespan)
optimizer = TesseractWrapper(config=config)


class OCRProcessor:
    """OCR Processing Handler.

    This class manages OCR (Optical Character Recognition) processing operations including
    file validation, image conversion, and document storage.

    Attributes:
        repository (DocumentRepository): Repository for document storage and retrieval.

    Methods:
        validate_file(file: UploadFile) -> str:
            Validates uploaded file against size, type and other constraints.
            Returns a unique filename for the document.

        convert_file_to_images(file: UploadFile) -> list[Image.Image]:
            Converts uploaded file (PDF or image) to a list of PIL Image objects.

        convert_to_base64(images: list[Image.Image]) -> list[str]:
            Converts PIL Image objects to base64 encoded strings.

        save_document(file_name: str) -> Document:
            Saves document metadata to the database.

    Raises:
        HTTPException: For various validation failures including:
            - Missing filename (400)
            - Unsupported file type (415)
            - Empty file (400)
            - File too large (413)
            - PDF page limit exceeded (400)
            - Invalid PDF (400)
            - Database storage failures (500)

    """

    """Handle OCR processing operations."""

    def __init__(self, document_repository: DocumentRepository) -> None:
        """Initialize OCR processor with repository.

        Args:
            document_repository (DocumentRepository): The repository instance for document storage and retrieval.

        Returns:
            None

        """
        """Initialize OCR processor with repository."""
        self.repository = document_repository

    async def validate_file(self, file: UploadFile) -> str:
        """Validate uploaded file against constraints.

        This method performs several validation checks on the uploaded file:
        1. Validates filename existence
        2. Ensures filename uniqueness
        3. Validates file type against accepted types
        4. Checks file size constraints
        5. For PDFs: validates page count

        Args:
            file (UploadFile): The file object to validate, containing attributes like
                filename, content_type, size, and file stream.

        Returns:
            str: The validated filename (potentially modified for uniqueness)

        Raises:
            HTTPException: With appropriate status codes for validation failures:
                - 400: Missing filename, empty file, or invalid PDF
                - 413: File size exceeds limit
                - 415: Unsupported media type

        Example:
            ```
            validated_filename = await ocr_instance.validate_file(uploaded_file)
            ```

        """
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
        """Convert uploaded file to list of PIL Images.

        This function takes an uploaded file and converts it into a list of PIL Image objects.
        For PDF files, it converts each page into a separate image. For other image files,
        it returns a single image in a list.

        Args:
            file (UploadFile): The uploaded file object containing either a PDF or image file.

        Returns:
            list[Image.Image]: A list of PIL Image objects representing the file contents.

        """
        logger.info("Converting file to images: %s", file.filename)

        if file.content_type == "application/pdf":
            return convert_from_bytes(file.file.read(), fmt="jpeg")

        return [Image.open(file.file)]

    @staticmethod
    def convert_to_base64(images: list[Image.Image]) -> list[str]:
        """Convert PIL Images to base64 strings.

        This function takes a list of PIL Image objects and converts each image to a base64-encoded string.
        Images in RGBA or P mode are converted to RGB before encoding. Each image is saved in JPEG format.

        Args:
            images (list[Image.Image]): A list of PIL Image objects to be converted

        Returns:
            list[str]: A list of base64-encoded strings representing the images

        Example:
            >>> images = [Image.open('image1.png'), Image.open('image2.jpg')]
            >>> base64_strings = convert_to_base64(images)

        """
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
        """Save document metadata to database.

        Args:
            file_name (str): Name of the document file to be saved

        Returns:
            Document: Created document instance with metadata

        Raises:
            HTTPException: If document creation fails with 500 status code

        """
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
    """Asynchronously retrieves all documents from the repository.

    Args:
        repository (AsyncGenerator[DocumentRepository, None]): An async generator that yields a DocumentRepository instance.
            Defaults to the repository provided by get_repository dependency.

    Returns:
        list[dict]: A list of documents, where each document is represented as a dictionary.

    Raises:
        HTTPException: If there's an error retrieving documents from the repository.
            Returns a 500 Internal Server Error status code.

    Example:
        >>> async def example():
        ...     docs = await get_docs()
        ...     print(docs)  # [{doc1_data}, {doc2_data}, ...]

    """
    try:
        documents = await repository.get_all()
        return [doc.to_dict() for doc in documents]
    except DocumentError as e:
        logger.exception("Failed to retrieve documents")
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
    """Process and analyze an uploaded document through OCR and subsequent processing.

    This asynchronous function handles document upload, encryption, OCR processing, and forwards
    the results to a downstream processing service.

    Args:
        file (UploadFile): The uploaded file to be processed
        repository (AsyncGenerator[DocumentRepository, None]): Repository dependency for document storage

    Returns:
        dict[str, str]: JSON response from the downstream processing service

    Raises:
        HTTPException: With status code 500 if:
            - Document processing fails in downstream service
            - Any other processing error occurs

    Flow:
        1. Validates and saves uploaded file
        2. Encrypts the saved file
        3. Performs OCR on file contents
        4. Saves initial document record
        5. Forwards OCR results to processor service
        6. Returns processor service response

    Note:
        The function includes built-in cleanup in case of failures at any stage

    """
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

        # Encrypt the file
        await cipher.encrypt_file(output_file)

        # Replace original with encrypted version
        if output_file.exists():
            output_file.unlink()

        Path(output_file.with_suffix(".tmp")).rename(output_file)

        # Path(temp_encrypted_file).rename(output_file)

        # For debugging purposes decrypt the file
        # await cipher.decrypt_file(output_file, tmp_decrypted_file)

        # if output_file.exists():
        #     output_file.unlink()

        # Path(tmp_decrypted_file).rename(output_file)

        file.file.seek(0)  # Reset file pointer

        images = processor.convert_file_to_images(file)

        logger.info("New file name: %s", output_file.name)

        # Perform OCR
        ocr_response, preprocessed_images = optimizer.process_batch(images)
        encoded_images = processor.convert_to_base64(preprocessed_images)

        # Save initial document record
        document = await processor.save_document(output_file.name)
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
                        "file_name": output_file.name,
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
        except DocumentError:
            logger.exception("Failed to delete document")

    if output_file and output_file.exists():
        try:
            output_file.unlink()
        except OSError:
            logger.exception("Failed to delete file")
