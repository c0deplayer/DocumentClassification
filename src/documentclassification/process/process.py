import base64
import io
import logging
from datetime import datetime
from pathlib import Path

import torchvision.transforms.v2 as v2
from PIL import Image
from fastapi import FastAPI
from transformers import (
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    LayoutLMv3Processor,
)

Path("logs/process").mkdir(parents=True, exist_ok=True)

current_date = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(f"logs/process/{current_date}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()
feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
processor = LayoutLMv3Processor(feature_extractor, tokenizer)
transform = v2.ToPILImage()


def scale_bounding_box(
    box: list[int], width_scale: float = 1.0, height_scale: float = 1.0
) -> list[int]:
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale),
    ]


@app.post("/process")
def process_text(body: dict[str, list[dict[str, list[int] | str] | bytes]]):
    logger.info("Processing text")

    images = [
        Image.open(io.BytesIO(base64.b64decode(image))) for image in body["images"]
    ]

    width, height = images[0].size

    width_scale = 1000 / width
    height_scale = 1000 / height

    words = []
    boxes = []
    for row in body["ocr_result"]:
        boxes.append(scale_bounding_box(row["bounding_box"], width_scale, height_scale))
        words.append(row["word"])

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

    image_data = encoding["pixel_values"][0]
    transform(image_data)
