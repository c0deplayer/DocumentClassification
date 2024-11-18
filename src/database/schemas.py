from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    """Base schema for document operations."""

    file_name: str = Field(..., description="Name of the document file")
    file_path: str = Field(..., description="Path to the document file")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Document creation timestamp"
    )
    classification: Optional[str] = Field(
        None, description="Document classification category"
    )
    summary: Optional[str] = Field(None, description="Document summary text")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document."""

    pass


class DocumentUpdate(BaseModel):
    """Schema for updating document attributes."""

    classification: Optional[str] = None
    summary: Optional[str] = None


class DocumentResponse(DocumentBase):
    """Schema for document response data."""

    id: int = Field(..., description="Document unique identifier")

    class Config:
        """Pydantic configuration."""

        from_attributes = True
