import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration settings."""

    ENV: str
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str
    POOL_SIZE: int = 5
    MAX_OVERFLOW: int = 10
    POOL_TIMEOUT: int = 30
    ECHO: bool = False

    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: Path = Path("/app/data/logs/database")

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        return cls(
            ENV=os.getenv("ENV"),
            DB_USER=os.getenv("DB_USER"),
            DB_PASSWORD=os.getenv("DB_PASSWORD"),
            DB_HOST=os.getenv("DB_HOST"),
            DB_PORT=os.getenv("DB_PORT"),
            DB_NAME=os.getenv("DB_NAME"),
            POOL_SIZE=int(os.getenv("DB_POOL_SIZE")),
            MAX_OVERFLOW=int(os.getenv("DB_MAX_OVERFLOW")),
            POOL_TIMEOUT=int(os.getenv("DB_POOL_TIMEOUT")),
            ECHO=bool(os.getenv("DB_ECHO").lower() == "true"),
        )
