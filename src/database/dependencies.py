from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from configs.database_config import DatabaseConfig

from .connections import DatabaseConnection
from .repository import DocumentRepository

db_config = DatabaseConfig.from_env()
db_connection = DatabaseConnection()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database session."""
    async with db_connection.get_session() as session:
        yield session


async def get_repository(
    session: AsyncSession = Depends(get_db),
) -> AsyncGenerator[DocumentRepository, None]:
    """Dependency for getting document repository."""
    yield DocumentRepository(session)
