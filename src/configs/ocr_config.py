from dataclasses import dataclass
from pathlib import Path
from typing import Final, Set

BYTE_MEGABYTE: Final[int] = 1024 * 1024


@dataclass(frozen=True)
class OCRConfig:
    """Configuration settings for OCR processing."""

    FILE_SIZE_LIMIT: int = 20 * BYTE_MEGABYTE
    MAX_PDF_PAGES: int = 15
    TARGET_SIZE: int = 1240
    MAX_WORKERS: int = 4
    LOG_DIR: Path = Path("/app/data/logs/ocr")
    ACCEPTED_FILE_TYPES: Set[str] = frozenset(
        {
            "image/jpeg",
            "image/png",
            "image/jpg",
            "image/webp",
            "application/pdf",
        }
    )
    PROCESSOR_URL: str = "http://processor:9090/text-preprocess"
    UPLOAD_DIR: Path = Path("/app/data/uploads")
    LOG_LEVEL: str = "DEBUG"
