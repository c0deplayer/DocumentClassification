import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, HTTPException, status
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError
from rich.progress import track

log_dir = Path("logs/ocr")
log_dir.mkdir(parents=True, exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"{log_dir}/{current_date}.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
app = FastAPI()
reader = easyocr.Reader(["en", "pl"], gpu=True)
FILE_SIZE = 20971520  # 20MB
accepted_file_types = [
    "image/jpeg",
    "image/png",
    "image/jpg",
    "application/pdf",
]


def validate_file(file: UploadFile) -> None:
    logger.info("Validating file: %s", file.filename)

    if file.content_type not in accepted_file_types:
        logger.error("Unsupported file type: %s", file.content_type)

        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type",
        )

    if file.size > FILE_SIZE:
        logger.error("File size too large: %d bytes", file.size)

        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size too large",
        )

    if file.content_type == "application/pdf":
        try:
            page_count = len(convert_from_bytes(file.file.read()))
            if page_count > 5:
                logger.error("PDF has more than 5 pages: %d pages", page_count)

                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="PDF has more than 5 pages",
                )
        except PDFPageCountError:
            logger.error(
                "Unable to get page count. The document stream might be empty or corrupted."
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unable to get page count. The document stream might be empty or corrupted.",
            )

        file.file.seek(0)


def convert_file_to_image(file: UploadFile) -> list[Image]:
    logger.info("Converting file to image: %s", file.filename)

    match file.content_type:
        case "application/pdf":
            return convert_from_bytes(file.file.read())
        case "image/jpeg" | "image/png" | "image/jpg":
            return [Image.open(file.file).convert("RGB")]


def convert_images_to_bytes(images: list[Image]) -> list[bytes]:
    logger.info("Converting images to bytes")

    images_bytes_list = []
    for img in images:
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        images_bytes_list.append(img_bytes.getvalue())

    return images_bytes_list


def recognize_text_from_image(
    images: list[Image],
) -> list[dict[str, list[int] | str]]:
    logger.info("Recognizing text from image")

    ocr_result = []

    for img in track(images, description="Recognizing text..."):
        for bbox, word, _ in reader.readtext(np.array(img)):
            ocr_result.append({"bounding_box": create_bounding_box(bbox), "word": word})

    return ocr_result


def create_bounding_box(bbox_data: list[tuple[float, float]]) -> list[int]:
    xs, ys = zip(*bbox_data)
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


@app.post("/ocr")
async def read_text_from_file(
    file: UploadFile,
) -> dict[str, list[dict[str, list[int] | str] | bytes]]:
    logger.info("Received file for OCR: %s", file.filename)

    validate_file(file)
    images = convert_file_to_image(file)
    ocr_result = recognize_text_from_image(images)

    logger.info("OCR processing completed for file: %s", file.filename)

    img_bytes = convert_images_to_bytes(images)

    return {
        "ocr_result": ocr_result,
        "images": [base64.b64encode(img) for img in img_bytes],
    }
