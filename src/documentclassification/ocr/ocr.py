import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, UploadFile, status
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from PIL import Image

from .optimized_ocr import OptimizedOCR

LOG_DIR = Path("logs/ocr")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / f"{CURRENT_DATE}.log", mode="a"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
app = FastAPI()
optimizer = OptimizedOCR(target_size=1024)
FILE_SIZE_LIMIT = 20 * 1024 * 1024  # 20MB
ACCEPTED_FILE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
    "image/webp",
    "application/pdf",
}


def validate_file(file: UploadFile) -> None:
    logger.info(f"Validating file: {file.filename}")

    if file.content_type not in ACCEPTED_FILE_TYPES:
        logger.error(f"Unsupported file type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type",
        )

    if file.size > FILE_SIZE_LIMIT:
        logger.error(f"File size too large: {file.size} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size too large",
        )

    if file.content_type == "application/pdf":
        try:
            page_count = len(convert_from_bytes(file.file.read()))
            if page_count > 15:
                logger.error(f"PDF has more than 15 pages: {page_count} pages")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF has more than 15 pages",
                )
        except PDFPageCountError:
            logger.error(
                "Unable to get page count. The document stream might be empty or corrupted."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to get page count. The document stream might be empty or corrupted.",
            )
        finally:
            file.file.seek(0)


def convert_file_to_image(file: UploadFile) -> list[Image.Image]:
    logger.info(f"Converting file to image: {file.filename}")

    if file.content_type == "application/pdf":
        return convert_from_bytes(file.file.read(), fmt="jpeg")

    return [Image.open(file.file)]


def convert_images_to_base64(images: list[Image.Image]) -> list[str]:
    """
    Convert PIL Images to base64 strings properly.

    Args:
        images: List of PIL Image objects

    Returns:
        List of base64 encoded image strings
    """
    encoded_images = []

    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        encoded = base64.b64encode(img_byte_arr).decode("utf-8")
        encoded_images.append(encoded)

    return encoded_images


@app.post("/ocr")
async def read_text_from_file(file: UploadFile) -> None:
    logger.info(f"Received file for OCR: {file.filename}")

    validate_file(file)
    images = convert_file_to_image(file)
    logger.info(f"Processing OCR for file: {file.filename}")
    ocr_result = optimizer.process_batch(images)

    logger.info(f"OCR processing completed for file: {file.filename}")

    encoded_images = convert_images_to_base64(images)

    requests.post(
        "http://processor:9090/processor",
        json={
            "ocr_result": ocr_result,
            "images": encoded_images,
        },
    )
