from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessorConfig:
    """Configuration for document processor."""

    LOG_DIR: Path = Path("/app/data/logs/processor")
    IMAGE_TARGET_SIZE: int = 1000
    MAX_SEQUENCE_LENGTH: int = 512
    BATCH_SIZE: int = 1
    MODEL_NAME: str = "microsoft/layoutlmv3-base"
    FILE_SIZE_LIMIT_MB: int = 20
    MAX_PDF_PAGES: int = 5
    LOG_LEVEL: str = "DEBUG"

    PREDICT_URL: str = "http://predictor:7070/predict-class"
