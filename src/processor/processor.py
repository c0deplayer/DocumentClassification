import logging
from datetime import datetime
from pathlib import Path

import aiohttp
from dataset import DocumentClassificationDataset
from fastapi import FastAPI, HTTPException
from lingua import Language, LanguageDetectorBuilder
from transformers import (
    LayoutLMv3ImageProcessor,
    LayoutLMv3Processor,
    LayoutLMv3TokenizerFast,
)

from configs.processor_config import ProcessorConfig
from payload.processor_models import ProcessorInput, ProcessorOutput

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
            config.LOG_DIR / f"{datetime.now():%Y-%m-%d}.log",
            mode="a",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI()
detector = (
    LanguageDetectorBuilder.from_all_languages()
    .with_preloaded_language_models()
    .build()
)


class DocumentProcessor:
    """Document processing service using LayoutLMv3."""

    def __init__(self) -> None:
        """Initialize document processor."""
        self.processor = LayoutLMv3Processor(
            LayoutLMv3ImageProcessor(apply_ocr=False),
            LayoutLMv3TokenizerFast.from_pretrained(
                "/app/data/models/layoutlmv3-base",
            ),
        )

    def process_document(self, input_data: ProcessorInput) -> ProcessorOutput:
        """Process document with LayoutLMv3.

        Args:
            input_data: Validated input data containing OCR results and images

        """
        try:
            encodings: dict = DocumentClassificationDataset.get_encodings(
                self.processor,
                input_data,
                config,
            )

            result = ProcessorOutput(
                input_ids=encodings["input_ids"].tolist(),
                attention_mask=encodings["attention_mask"].tolist(),
                bbox=encodings["bbox"].tolist(),
                pixel_values=encodings["pixel_values"].tolist(),
            )

        except Exception as e:
            logger.exception("Document processing failed")
            detail = f"Processing failed: {e!s}"
            raise HTTPException(status_code=500, detail=detail) from e

        else:
            return result


# Create processor instance
document_processor = DocumentProcessor()


@app.post("/text-preprocess")
async def process_text(data: ProcessorInput) -> dict[str, str]:
    """Process document text and layout.

    Args:
        data: Validated input data containing OCR results and images

    """
    try:
        text = " ".join(
            result.word
            for ocr_result in data.ocr_result
            for result in ocr_result
        )

        detected_language = detector.detect_language_of(text)

        if detected_language is None:
            # Default to English if language detection fails
            detected_language = Language.ENGLISH

        if detected_language != Language.ENGLISH:
            timeout = aiohttp.ClientTimeout(total=480)
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    config.PREDICT_URL,
                    json={
                        "input_ids": None,
                        "attention_mask": None,
                        "bbox": None,
                        "pixel_values": None,
                        "file_name": data.file_name,
                        "text": text,
                        "language": detected_language.name,
                    },
                    timeout=timeout,
                ) as response,
            ):
                response.raise_for_status()
                return await response.json()

        encodings = document_processor.process_document(data)

        encodings = encodings.model_dump()

        logger.debug(
            "Text from OCR: %s",
            " ".join(
                result.word
                for ocr_result in data.ocr_result
                for result in ocr_result
            ),
        )

        timeout = aiohttp.ClientTimeout(total=480)
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                config.PREDICT_URL,
                json={
                    "input_ids": encodings["input_ids"],
                    "attention_mask": encodings["attention_mask"],
                    "bbox": encodings["bbox"],
                    "pixel_values": encodings["pixel_values"],
                    "file_name": data.file_name,
                    "text": text,
                    "language": detected_language.name,
                },
                timeout=timeout,
            ) as response,
        ):
            response.raise_for_status()
            return await response.json()

    except Exception as e:
        logger.exception("Endpoint processing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
