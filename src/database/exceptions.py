from typing import Optional


class DatabaseError(Exception):
    """Base exception for database operations."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class DocumentError(DatabaseError):
    """Base exception for document-related operations."""

    pass


class DocumentNotFoundError(DocumentError):
    """Raised when a document is not found."""

    pass


class DocumentSaveError(DocumentError):
    """Raised when saving a document fails."""

    pass


class DocumentUpdateError(DocumentError):
    """Raised when updating a document fails."""

    pass
