from __future__ import annotations

from datetime import datetime

from sqlalchemy import TIMESTAMP, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all database models."""


class Document(Base):
    """Document model with improved type hints and validation."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )
    file_path: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        default=datetime.utcnow,
    )
    classification: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    summary: Mapped[str | None] = mapped_column(String, nullable=True)

    def to_dict(self) -> dict:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "created_at": self.created_at.isoformat(),
            "classification": self.classification,
            "summary": self.summary,
        }
