import base64
import io
import logging
from datetime import datetime
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import ProcessorMixin

from configs.processor_config import ProcessorConfig
from payload.processor_models import OCRResult, ProcessorInput

config = ProcessorConfig()
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


class DocumentClassificationDataset(Dataset):
    """Dataset for document classification using LayoutLMv3."""

    def __init__(
        self,
        processor: ProcessorMixin,
        data: ProcessorInput,
        config: ProcessorConfig,
        labels: Optional[list[int]] = None,
        *,
        encodings: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Initialize document classification dataset.

        Args:
            processor: LayoutLMv3 processor instance
            data: Validated input data
            config: Processing configuration
            labels: Optional classification labels
        """
        self.processor = processor
        self.config = config
        self.labels = labels
        self.encodings = (
            encodings if encodings is not None else self._prepare_encodings(data)
        )

    @classmethod
    def get_encodings(
        cls,
        processor: ProcessorMixin,
        data: ProcessorInput,
        config: ProcessorConfig,
    ) -> dict[str, torch.Tensor]:
        """
        Get document encodings without instantiating the full dataset.

        This method processes the input data and returns only the encodings,
        which is useful when you don't need the full dataset functionality.

        Args:
            processor: LayoutLMv3 processor instance
            data: Validated input data containing OCR results and images
            config: Processing configuration settings

        Returns:
            dict[str, torch.Tensor]: Encoded document features

        Raises:
            ValueError: If encoding preparation fails
        """
        temp_instance = cls(
            processor=processor,
            data=data,
            config=config,
            encodings={},
        )

        return temp_instance._prepare_encodings(data)

    def _prepare_encodings(self, data: ProcessorInput) -> dict[str, torch.Tensor]:
        """Prepare encodings from input data for multiple pages."""
        try:
            images = self._decode_images(data.images)
            if not images:
                raise ValueError("No valid images found in input data")

            scales = self._calculate_scales(images[0].size)
            words_per_page, boxes_per_page = self._prepare_ocr_data(
                data.ocr_result, scales
            )

            # Process each page separately
            page_encodings = []
            for img, words, boxes in zip(images, words_per_page, boxes_per_page):
                encoding = self.processor(
                    [img],  # Processor expects a list of images
                    words,
                    boxes=boxes,
                    max_length=self.config.MAX_SEQUENCE_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                page_encodings.append(encoding)

            # Combine encodings from all pages
            combined_encodings = {
                "input_ids": torch.cat(
                    [enc["input_ids"] for enc in page_encodings], dim=0
                ),
                "attention_mask": torch.cat(
                    [enc["attention_mask"] for enc in page_encodings], dim=0
                ),
                "bbox": torch.cat([enc["bbox"] for enc in page_encodings], dim=0),
                "pixel_values": torch.cat(
                    [enc["pixel_values"] for enc in page_encodings], dim=0
                ),
            }

            logger.debug(
                "Combined encoding shapes: %s",
                {k: v.shape for k, v in combined_encodings.items()},
            )

            return combined_encodings

        except Exception as e:
            logger.exception("Error preparing encodings")
            raise ValueError(f"Encoding preparation failed: {str(e)}") from e

    @staticmethod
    def _decode_images(encoded_images: list[str]) -> list[Image.Image]:
        """Decode base64 images to PIL Images."""
        return [
            Image.open(io.BytesIO(base64.b64decode(img))).convert("RGB")
            for img in encoded_images
        ]

    def _calculate_scales(self, image_size: tuple[int, int]) -> tuple[float, float]:
        """Calculate scaling factors for image resizing."""
        width, height = image_size
        logger.debug("Image size: %s", image_size)
        width_scale = self.config.IMAGE_TARGET_SIZE / width
        height_scale = self.config.IMAGE_TARGET_SIZE / height
        logger.debug("Scaling factors: %s", (width_scale, height_scale))

        return width_scale, height_scale

    @staticmethod
    def _prepare_ocr_data(
        ocr_results: list[list[OCRResult]], scales: tuple[float, float]
    ) -> tuple[list[str], list[list[int]]]:
        """Prepare OCR data with scaling."""
        if not ocr_results:
            return [], []

        words = []
        boxes = []
        width_scale, height_scale = scales

        for page_results in ocr_results:
            tmp_word, tmp_box = [], []
            for result in page_results:
                tmp_word.append(result.word)
                scaled_box = [
                    int(coord * scale)
                    for coord, scale in zip(
                        result.bounding_box, [width_scale, height_scale] * 2
                    )
                ]
                tmp_box.append(scaled_box)

            words.append(tmp_word)
            boxes.append(tmp_box)

        return words, boxes

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = {
            key: tensor[idx].flatten()
            if key not in ("pixel_values", "bbox")
            else tensor[idx].flatten(end_dim=1)
            for key, tensor in self.encodings.items()
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])

        return item
