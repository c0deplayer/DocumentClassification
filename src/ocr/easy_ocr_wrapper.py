import logging

import cv2
import easyocr
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from configs.ocr_config import OCRConfig
from payload.ocr_models import OCRResponse
from payload.shared_models import OCRResult


class ImagePreprocessor:
    """Handle image preprocessing operations."""

    def __init__(self, target_size: int) -> None:
        """Initialize the image preprocessor.

        Args:
            target_size: Maximum dimension size to resize images to

        """
        self.target_size = target_size
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        if self.target_size == 0 or max(image.size) <= self.target_size:
            return image

        width, height = image.size
        scale = min(self.target_size / width, self.target_size / height)
        new_size = (int(width * scale), int(height * scale))

        return image.resize(new_size, resample=Image.Resampling.LANCZOS)

    def enhance_image(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply image enhancement techniques."""
        enhanced = self._clahe.apply(image)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        return np.array(denoised, dtype=np.uint8)

    def preprocess(self, image: Image.Image) -> NDArray[np.uint8]:
        """Complete image preprocessing pipeline."""
        if image.mode != "L":
            image = image.convert("L")

        image = self.resize_image(image)
        image_array = np.array(image)

        return self.enhance_image(image_array)


class EasyOCRWrapper:
    """Optimized OCR processing with performance enhancements."""

    def __init__(self, config: OCRConfig) -> None:
        """Initialize OCR processor with configuration."""
        self.config = config
        self.preprocessor = ImagePreprocessor(config.TARGET_SIZE)
        self.reader = self._initialize_reader()

    def _initialize_reader(self) -> easyocr.Reader:
        """Initialize EasyOCR with optimized settings."""
        return easyocr.Reader(
            ["en", "pl"],
            gpu=True,
            model_storage_directory="/app/data/models",
            download_enabled=True,
            quantize=True,
        )

    @staticmethod
    def create_bounding_box(bbox_data: list[tuple[float, float]]) -> list[int]:
        """Create standardized bounding box from coordinates."""
        xs, ys = zip(*bbox_data)

        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    def _process_single(self, image: NDArray[np.uint8]) -> list[OCRResult]:
        """Process a single image with optimized settings."""
        try:
            results = self.reader.readtext(image)
            return [
                OCRResult(
                    bounding_box=self.create_bounding_box(bbox),
                    word=word,
                )
                for bbox, word, _ in results
            ]
        except Exception as e:
            logging.exception("Error processing image: %s", str(e))
            return []

    def process_batch(
        self,
        images: list[Image.Image],
    ) -> tuple[OCRResponse, list[Image.Image]]:
        """Process multiple images in optimized batch."""
        processed = [self.preprocessor.preprocess(image) for image in images]
        resized = [self.preprocessor.resize_image(image) for image in images]
        results = [self._process_single(image) for image in processed]

        response = OCRResponse(results=results, page_count=len(results))

        return response, resized
