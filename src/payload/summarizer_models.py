# src/summarization/summarizer.py

from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Request model for summarization."""

    file_name: str = Field(..., description="Name of the file to summarize")
    text: str = Field(..., description="Text content to summarize")
    classification: str = Field(..., description="Predicted classes for the text")
