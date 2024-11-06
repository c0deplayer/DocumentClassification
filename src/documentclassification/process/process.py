import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import torchvision.transforms.v2 as v2
from fastapi import FastAPI
from PIL import Image
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

# Constants
LOG_DIR = Path("logs/process")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = LOG_DIR / f"{CURRENT_DATE}.log"

FILE_SIZE_LIMIT_MB = 20
MAX_PDF_PAGES = 5
IMAGE_TARGET_SIZE = 1000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()
processor = LayoutLMv3Processor(
    LayoutLMv3ImageProcessor(apply_ocr=False),
    LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base"),
)
transform = v2.ToPILImage()


def scale_bounding_box(
    box: list[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> list[int]:
    """Scales bounding box coordinates based on provided scales."""
    return [
        int(coord * scale) for coord, scale in zip(box, [width_scale, height_scale] * 2)
    ]


@app.post("/process")
def process_text(body: dict[str, list[dict[str, list[int] | str] | bytes]]) -> None:
    """Processes OCR result and images for further analysis."""
    logger.info("Starting text processing.")

    try:
        images = [
            Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
            for img in body.get("images", [])
        ]
        if not images:
            logger.error("No images found in the request body.")
            return

        width, height = images[0].size
        width_scale = IMAGE_TARGET_SIZE / width
        height_scale = IMAGE_TARGET_SIZE / height

        ocr_results = body.get("ocr_result", [])
        words, boxes = (
            zip(
                *(
                    (
                        item["word"],
                        scale_bounding_box(
                            item["bounding_box"], width_scale, height_scale
                        ),
                    )
                    for item in ocr_results
                )
            )
            if ocr_results
            else ([], [])
        )

        encoding = processor(
            images,
            words,
            boxes=list(boxes),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        logger.info(
            f"Encoding shapes - input_ids: {list(encoding['input_ids'].shape)}, "
            f"bbox: {list(encoding['bbox'].shape)}, "
            f"pixel_values: {list(encoding['pixel_values'].shape)}, "
            f"image size: {images[0].size}"
        )

        image_data = transform(encoding["pixel_values"][0])
        # Further processing can be done with image_data

    except Exception as e:
        logger.exception("An error occurred during text processing.")
        raise e
