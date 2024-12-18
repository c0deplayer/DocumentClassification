from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    """Request model for summarization."""

    file_name: str = Field(..., description="Name of the file to summarize")
    text: str = Field(..., description="Text content to summarize")
    classification: str = Field(
        ..., description="Predicted classes for the text"
    )


class SummaryResponse(BaseModel):
    """Structured response from the summarization model."""

    summary: str = Field(..., description="Generated summary text")
    word_count: int = Field(..., description="Number of words in summary")
