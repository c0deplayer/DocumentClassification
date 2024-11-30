import os
import re
from typing import AsyncGenerator

from database.repository import DocumentRepository


def make_unique_filename(original_filename: str, existing_filenames: set[str]) -> str:
    """
    Generate a unique filename by appending a number if the filename already exists.

    Args:
        original_filename: The original filename to make unique
        existing_filenames: Set of existing filenames to check against

    Returns:
        A unique filename with (n) appended if necessary

    Examples:
        >>> make_unique_filename("test.pdf", {"test.pdf"})
        'test (1).pdf'
        >>> make_unique_filename("test.pdf", {"test.pdf", "test (1).pdf"})
        'test (2).pdf'
        >>> make_unique_filename("my file.pdf", {"my file.pdf"})
        'my file (1).pdf'
    """
    if original_filename not in existing_filenames:
        return original_filename

    # Split the filename into name and extension
    name, ext = os.path.splitext(original_filename)

    # Check if filename already ends with a number in parentheses
    pattern = r"(.*?)\s*(?:\((\d+)\))?$"
    match = re.match(pattern, name)
    if not match:
        return original_filename

    base_name = match.group(1)
    counter = 1

    # Keep incrementing counter until we find a unique filename
    while True:
        new_filename = f"{base_name} ({counter}){ext}"
        if new_filename not in existing_filenames:
            return new_filename
        counter += 1


async def get_unique_filename(
    filename: str,
    repository: AsyncGenerator[DocumentRepository, None],
    max_attempts: int = 100,
) -> str:
    """
    Generate a unique filename that doesn't exist in the database.

    Args:
        filename: Original filename to make unique
        repository: Document repository instance
        max_attempts: Maximum number of attempts to find unique name

    Returns:
        Unique filename

    Raises:
        ValueError: If unable to generate unique filename within max_attempts
    """
    try:
        # Get all existing filenames from database
        documents = await repository.get_all()
        existing_filenames = {doc.file_name for doc in documents}

        # Generate unique filename
        unique_name = make_unique_filename(filename, existing_filenames)

        # Verify we didn't exceed max attempts (check the number in parentheses)
        match = re.search(r"\((\d+)\)", unique_name)
        if match and int(match.group(1)) > max_attempts:
            raise ValueError(
                f"Unable to generate unique filename for {filename} "
                f"within {max_attempts} attempts"
            )

        return unique_name

    except Exception as e:
        raise ValueError(f"Error generating unique filename: {str(e)}")
