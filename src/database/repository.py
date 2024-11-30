from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from .exceptions import (
    DocumentError,
    DocumentNotFoundError,
    DocumentSaveError,
    DocumentUpdateError,
)
from .models import Document
from .schemas import DocumentCreate, DocumentUpdate


class DocumentRepository:
    """Repository for document-related database operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self.session = session

    async def create(self, document: DocumentCreate) -> Document:
        """Create a new document record."""
        try:
            db_document = Document(
                file_name=document.file_name,
                file_path=document.file_path,
                created_at=document.created_at or datetime.now(datetime.timezone.utc),
                classification=document.classification,
                summary=document.summary,
            )

            self.session.add(db_document)
            await self.session.commit()
            await self.session.refresh(db_document)

            return db_document

        except SQLAlchemyError as e:
            raise DocumentSaveError(
                f"Failed to save document {document.file_name}", original_error=e
            )

    async def get_by_id(self, document_id: int) -> Document:
        """Retrieve document by ID."""
        query = select(Document).filter(Document.id == document_id)
        result = await self.session.execute(query)
        document = result.scalar_one_or_none()

        if not document:
            raise DocumentNotFoundError(f"Document with ID {document_id} not found")
        return document

    async def get_by_filename(
        self, file_name: str, *, return_bool: bool = False
    ) -> Document | bool:
        """Retrieve document by filename."""
        query = select(Document).filter(Document.file_name == file_name)
        result = await self.session.execute(query)
        document = result.scalar_one_or_none()

        if not document:
            if return_bool:
                return False
            raise DocumentNotFoundError(f"Document {file_name} not found")

        return document

    async def get_all(self) -> Sequence[Document]:
        """Retrieve all documents."""
        query = select(Document)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def update(self, document_id: int, update_data: DocumentUpdate) -> Document:
        """Update document attributes."""
        try:
            document = await self.get_by_id(document_id)

            update_dict = update_data.model_dump(exclude_unset=True)
            for key, value in update_dict.items():
                setattr(document, key, value)

            await self.session.commit()
            await self.session.refresh(document)

            return document

        except SQLAlchemyError as e:
            raise DocumentUpdateError(
                f"Failed to update document {document_id}", original_error=e
            )

    async def update_classification(
        self, file_name: str, classification: str
    ) -> Document:
        """Update document classification."""
        try:
            document = await self.get_by_filename(file_name)
            document.classification = classification

            await self.session.commit()
            await self.session.refresh(document)

            return document

        except SQLAlchemyError as e:
            raise DocumentUpdateError(
                f"Failed to update classification for {file_name}", original_error=e
            )

    async def update_summary(self, file_name: str, summary: str) -> Document:
        """Update document summary."""
        try:
            document = await self.get_by_filename(file_name)
            document.summary = summary

            await self.session.commit()
            await self.session.refresh(document)

            return document

        except SQLAlchemyError as e:
            raise DocumentUpdateError(
                f"Failed to update summary for {file_name}", original_error=e
            )

    async def delete(self, document_id: int) -> None:
        """Delete document by ID."""
        try:
            document = await self.get_by_id(document_id)
            await self.session.delete(document)
            await self.session.commit()

        except SQLAlchemyError as e:
            raise DocumentError(
                f"Failed to delete document {document_id}", original_error=e
            )
