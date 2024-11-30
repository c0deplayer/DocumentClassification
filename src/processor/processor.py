import logging
from datetime import datetime
from pathlib import Path

import requests
from dataset import DocumentClassificationDataset
from fastapi import FastAPI, HTTPException
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from configs.processor_config import ProcessorConfig
from payload.processor_models import ProcessorInput, ProcessorResult

config = ProcessorConfig()
# TMP solution to avoid error during testing
try:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    config = ProcessorConfig(LOG_DIR=Path("logs/processor"))
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


class DocumentProcessor:
    """Document processing service using LayoutLMv3."""

    def __init__(self) -> None:
        """Initialize document processor."""
        self.processor = LayoutLMv3Processor(
            LayoutLMv3ImageProcessor(apply_ocr=False),
            LayoutLMv3TokenizerFast.from_pretrained("/app/data/models/layoutlmv3-base"),
        )

    def process_document(self, input_data: ProcessorInput) -> ProcessorResult:
        """
        Process document with LayoutLMv3.

        Args:
            input_data: Validated input data containing OCR results and images
        """
        try:
            encodings: list[dict] = DocumentClassificationDataset.get_encodings(
                self.processor, input_data, config
            )

            result = ProcessorResult(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                bbox=encodings["bbox"],
                pixel_values=encodings["pixel_values"],
            )

            return result

        except Exception as e:
            logger.exception("Document processing failed")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Create processor instance
document_processor = DocumentProcessor()


@app.post("/text-preprocess")
async def process_text(data: ProcessorInput) -> dict[str, str]:
    """
    Process document text and layout.

    Args:
        data: Validated input data containing OCR results and images
    """
    try:
        encodings = document_processor.process_document(data)

        encodings = {k: v.tolist() for k, v in encodings.model_dump().items()}

        # logger.info(
        #     "Text from OCR: %s", " ".join(result.word for result in data.ocr_result[0])
        # )

        response = requests.post(
            config.PREDICT_URL,
            json={
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "bbox": encodings["bbox"],
                "pixel_values": encodings["pixel_values"],
                "file_name": data.file_name,
                "text": " ".join(result.word for result in data.ocr_result[0]),
            },
            timeout=300,
        )
        return response.json()

    except Exception as e:
        logger.exception("Endpoint processing failed")
        raise HTTPException(status_code=500, detail=str(e))
