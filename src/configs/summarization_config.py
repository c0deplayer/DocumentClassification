from dataclasses import dataclass, field
from pathlib import Path

import psutil


@dataclass(frozen=True)
class SummarizationConfig:
    """Configuration for summarization service."""

    LOG_DIR: Path = Path("/app/data/logs/summarization")
    LOG_LEVEL: str = "DEBUG"

    # Ollama configuration
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    MODEL_NAME: str = "llama3.2:3b"

    # Summary constraints
    MAX_INPUT_LENGTH: int = 10000
    DEFAULT_MAX_LENGTH: int = 200
    DEFAULT_MIN_LENGTH: int = 50

    # Model parameters
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0

    # Llama specific parameters
    NUM_CTX: int = 4096  # Context window size
    NUM_GPU: int = 0
    NUM_THREAD: int = field(
        default_factory=lambda: psutil.cpu_count(logical=False) or 1,
    )
