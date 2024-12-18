from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytesseract
from PIL import Image

from payload.ocr_models import OCRResponse
from payload.shared_models import OCRResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from configs.ocr_config import OCRConfig


class ImagePreprocessor:
    """Handle image preprocessing operations."""

    def __init__(self, target_size: int) -> None:
        """Initialize image preprocessor.

        Args:
            target_size: Maximum size for image dimension.

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
        # Convert to grayscale if not already
        CHANNELS_RGB = 3
        if len(image.shape) == CHANNELS_RGB:
            image = np.asarray(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                dtype=np.uint8,
            )

        # Apply CLAHE for contrast enhancement
        enhanced = self._clahe.apply(image)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)

        # Binarization using Otsu's method
        _, binary = cv2.threshold(
            denoised,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return np.asarray(binary, dtype=np.uint8)

    def preprocess(self, image: Image.Image) -> NDArray[np.uint8]:
        """Complete image preprocessing pipeline."""
        # Resize image
        image = self.resize_image(image)

        # Convert to numpy array
        image_array = np.array(image)

        # Enhance image
        return self.enhance_image(image_array)


class TesseractWrapper:
    """OCR processing with Tesseract."""

    def __init__(self, config: OCRConfig) -> None:
        """Initialize OCR processor with configuration."""
        self.config = config
        self.preprocessor = ImagePreprocessor(config.TARGET_SIZE)

        # Configure Tesseract parameters
        self.custom_config = r"--oem 3 --psm 11"

    @staticmethod
    def create_bounding_box(
        bbox_data: list[tuple[float, float]] | tuple[int, int, int, int],
    ) -> list[int]:
        """Create standardized bounding box from coordinates.

        Args:
            bbox_data: Either a list of (x,y) coordinate tuples or a tuple of (x, y, width, height)

        Returns:
            list[int]: Standardized bounding box [x_min, y_min, x_max, y_max]

        """
        if isinstance(bbox_data, tuple) and len(bbox_data) == 4:
            # Handle Tesseract format (x, y, width, height)
            x, y, w, h = bbox_data
            return [int(x), int(y), int(x + w), int(y + h)]
        # Handle original format (list of coordinate tuples)
        xs, ys = zip(*bbox_data)
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    def _process_single(self, image: NDArray[np.uint8]) -> list[OCRResult]:
        """Process a single image with Tesseract."""
        try:
            results = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=self.custom_config,
            )

            ocr_results = []
            n_boxes = len(results["text"])

            for i in range(n_boxes):
                if (
                    int(results["conf"][i]) < 0
                    or not results["text"][i].strip()
                ):
                    continue

                # Create bbox_data tuple for compatibility
                bbox = self.create_bounding_box(
                    (
                        results["left"][i],
                        results["top"][i],
                        results["width"][i],
                        results["height"][i],
                    ),
                )

                ocr_results.append(
                    OCRResult(
                        bounding_box=bbox,
                        word=results["text"][i].strip(),
                    ),
                )

        except Exception as e:
            logging.exception("Error processing image: %s", str(e))
            return []

        else:
            return ocr_results

    def process_batch(
        self,
        images: list[Image.Image],
    ) -> tuple[OCRResponse, list[Image.Image]]:
        """Process multiple images in optimized batch."""
        # Preprocess all images
        processed = [self.preprocessor.preprocess(image) for image in images]
        resized = [self.preprocessor.resize_image(image) for image in images]

        # Process each image
        results = [self._process_single(image) for image in processed]

        # # Validate results before returning
        # for page_results in results:
        #     for result in page_results:
        #         bbox = result.bounding_box
        #         if not all(0 <= coord <= 1000 for coord in bbox):
        #             logging.warning(f"Invalid bbox coordinates detected: {bbox}")

        response = OCRResponse(results=results, page_count=len(results))

        return response, resized
