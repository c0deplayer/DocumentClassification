from pydantic import BaseModel, ConfigDict, Field

from .shared_models import OCRResult


class OCRResponse(BaseModel):
    """Response model for OCR processing."""

    results: list[list[OCRResult]] = Field(
        ...,
        description="list of OCR results from the document",
    )

    page_count: int = Field(1, description="Number of pages processed", ge=1)

    model_config = ConfigDict(frozen=True)
