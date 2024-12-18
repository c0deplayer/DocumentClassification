from __future__ import annotations


class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        """Initialize DatabaseError exception.

        Args:
            message: Error message to display
            original_error: Original exception that caused this error, if any

        """
        super().__init__(message)
        self.original_error = original_error


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""


class DocumentError(DatabaseError):
    """Base exception for document-related operations."""


class DocumentNotFoundError(DocumentError):
    """Raised when a document is not found."""


class DocumentSaveError(DocumentError):
    """Raised when saving a document fails."""


class DocumentUpdateError(DocumentError):
    """Raised when updating a document fails."""
