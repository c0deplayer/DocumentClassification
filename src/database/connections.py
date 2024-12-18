import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from configs.database_config import DatabaseConfig

from .exceptions import ConnectionError as DBConnectionError

config = DatabaseConfig.from_env()
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


class DatabaseConnection:
    """Manages database connection and session creation."""

    def __init__(self) -> None:
        """Initialize database connection manager."""
        self.engine = self._create_engine()
        self.async_session_maker = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    def _create_engine(self) -> AsyncEngine:
        """Create SQLAlchemy engine with proper configuration."""
        try:
            db_url = URL.create(
                "postgresql+asyncpg",
                username=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=int(config.DB_PORT),
                database=config.DB_NAME,
            )

            return create_async_engine(
                db_url,
                pool_size=config.POOL_SIZE,
                max_overflow=config.MAX_OVERFLOW,
                pool_timeout=config.POOL_TIMEOUT,
                echo=config.ECHO,
            )

        except Exception as e:
            logger.exception("Failed to create database engine")
            error_msg = "Failed to establish database connection"
            raise DBConnectionError(error_msg) from e

    async def verify_connection(self) -> None:
        """Verify database connection."""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection verified successfully")
        except Exception as e:
            logger.exception("Failed to verify database connection")
            error_msg = "Failed to verify database connection"
            raise DBConnectionError(error_msg) from e

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create and manage an async session."""
        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                logger.exception("Database session error")
                raise
            finally:
                await session.close()
