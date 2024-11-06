import logging

import cv2
import easyocr
import numpy as np
import rich
from PIL import Image


class OptimizedOCR:
    def __init__(self, max_workers: int = 4, target_size: int = 1280):
        """
        Initialize OptimizedOCR with performance optimizations.

        Args:
            max_workers: Number of worker threads for parallel processing
            target_size: Target image size for resizing
        """
        self.max_workers = max_workers
        self.target_size = target_size
        self.reader = self._initialize_reader()

    def _initialize_reader(self) -> easyocr.Reader:
        """Initialize EasyOCR with optimized settings."""
        reader = easyocr.Reader(
            ["en", "pl"],
            gpu=False,
            model_storage_directory="./models",
            download_enabled=True,
            quantize=True,
        )
        return reader

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Optimize image for OCR processing.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Resize image while maintaining aspect ratio
        # h, w = image.size
        # logging.info(f"Original image size: {w}x{h}")
        # scale = min(self.target_size / w, self.target_size / h)
        # scale = 0.75
        # new_w, new_h = int(w * scale), int(h * scale)
        # image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
        # logging.info(f"Resized image size: {new_w}x{new_h}")
        # image.show()

        image = np.array(image)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # Denoise
        image = cv2.fastNlMeansDenoising(image)

        return image

    def process_batch(self, images: list[Image.Image]) -> list[list[dict]]:
        """
        Process images in batches for better CPU utilization.

        Args:
            images: list of images to process

        Returns:
            list of OCR results for each image
        """
        results = []
        preprocessed = [self.preprocess_image(img) for img in images]
        for img in rich.progress.track(
            preprocessed, description="Processing images..."
        ):
            image_results = self._process_single(img)
            results.extend(image_results)

        return results

    def _process_single(self, image: np.ndarray) -> list[dict]:
        """Process a single image with optimized settings."""
        try:
            # Use lower confidence threshold for faster processing
            results = self.reader.readtext(
                image,
                # paragraph=True,  # Group text into paragraphs
                # batch_size=1,
                # workers=8,
                # beamWidth=2,  # Reduce beam width for faster inference
                # contrast_ths=0.1,  # Lower contrast threshold
                # adjust_contrast=0.5,  # Reduce contrast adjustment
                canvas_size=self.target_size,  # Resize image for better OCR
                # low_text=0.3,  # Lower text confidence threshold
            )

            return [
                {
                    "bounding_box": self.create_bounding_box(bbox),
                    "word": word,
                }
                for bbox, word, _ in results
            ]
        except Exception as e:
            logging.error(f"Error processing image: {str(e)}")
            return []

    @staticmethod
    def create_bounding_box(bbox_data: list[tuple[float, float]]) -> list[int]:
        xs, ys = zip(*bbox_data)
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
