import base64
import io
import logging
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, UploadFile, status
from optimized_ocr import OptimizedOCR
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image

from configs.ocr_config import OCRConfig

config = OCRConfig()
config.LOG_DIR.mkdir(parents=True, exist_ok=True)


logging.basicConfig(
    level=logging.DEBUG if config.LOG_LEVEL == "DEBUG" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            config.LOG_DIR / f"{datetime.now():%Y-%m-%d}.log", mode="a"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI()
optimizer = OptimizedOCR(config=config)


class OCRProcessor:
    """Handle OCR processing operations."""

    @staticmethod
    def validate_file(file: UploadFile) -> None:
        """Validate uploaded file against constraints."""
        logger.info("Validating file: %s", file.filename)

        if file.content_type not in config.ACCEPTED_FILE_TYPES:
            logger.error("Unsupported file type: %s", file.content_type)
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported file type",
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
            except PDFPageCountError as e:
                logger.error("PDF processing error: %s", str(e))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid or corrupted PDF file",
                ) from e
            finally:
                file.file.seek(0)

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
            with io.BytesIO() as buffer:
                img.save(buffer, format="JPEG")
                img_byte_arr = buffer.getvalue()

                encoded = base64.b64encode(img_byte_arr).decode("utf-8")
                encoded_images.append(encoded)

        return encoded_images


@app.post("/ocr")
async def process_document(file: UploadFile) -> dict[str, str]:
    """Process document for OCR and forward results."""
    logger.info("Processing document: %s", file.filename)

    processor = OCRProcessor()

    try:
        processor.validate_file(file)
        images = processor.convert_file_to_images(file)

        ocr_response, preprocessed_images = optimizer.process_batch(images)
        encoded_images = processor.convert_to_base64(preprocessed_images)

        logger.debug("Processed images: %d", len(preprocessed_images))

        response = requests.post(
            config.PROCESSOR_URL,
            json={
                "ocr_result": ocr_response.model_dump(exclude={"page_count"})[
                    "results"
                ],
                "images": encoded_images,
            },
            timeout=30,
        )

        logger.info("Document processing completed: %s", file.filename)

        return response.json()

    except Exception as e:
        logger.error("Processing error: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed",
        ) from e
