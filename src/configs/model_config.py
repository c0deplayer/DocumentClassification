from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for document processor."""

    LOG_DIR: Path = Path("/app/data/logs/predictor")
    LOG_LEVEL: str = "DEBUG"
    DOCUMENT_CLASSES: list[str] = field(
        default_factory=lambda: [
            "letter",
            "form",
            "email",
            "handwritten",
            "advertisement",
            "scientific report",
            "scientific publication",
            "specification",
            "file folder",
            "news article",
            "budget",
            "invoice",
            "presentation",
            "questionnaire",
            "resume",
            "memo",
        ]
    )
