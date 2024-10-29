import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import torchvision.transforms.v2 as v2
from PIL import Image
from fastapi import FastAPI
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ImageProcessor,
    LayoutLMv3TokenizerFast,
)

log_dir = Path("logs/process")
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
processor = LayoutLMv3Processor(
    LayoutLMv3ImageProcessor(apply_ocr=False),
    LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base"),
)
transform = v2.ToPILImage()


def scale_bounding_box(box, width_scale=1.0, height_scale=1.0) -> list[int]:
    return [
        int(coord * scale)
        for coord, scale in zip(
            box, [width_scale, height_scale, width_scale, height_scale]
        )
    ]


@app.post("/process")
def process_text(body: dict[str, list[dict[str, list[int] | str] | bytes]]) -> None:
    logger.info("Processing text")

    images = [Image.open(io.BytesIO(base64.b64decode(img))) for img in body["images"]]
    width, height = images[0].size
    width_scale, height_scale = 1000 / width, 1000 / height

    words, boxes = zip(
        *[
            (
                row["word"],
                scale_bounding_box(row["bounding_box"], width_scale, height_scale),
            )
            for row in body["ocr_result"]
        ]
    )

    encoding = processor(
        images,
        words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    logger.info(f"""
        input_ids:  {list(encoding["input_ids"].squeeze().shape)}
        word boxes: {list(encoding["bbox"].squeeze().shape)}
        image data: {list(encoding["pixel_values"].squeeze().shape)}
        image size: {images[0].size}
    """)

    image_data = transform(encoding["pixel_values"][0])
