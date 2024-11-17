from pydantic import BaseModel, ConfigDict, Field


class OCRResult(BaseModel):
    """Structured OCR result using Pydantic BaseModel."""

    bounding_box: list[int] = Field(
        ...,
        description="Coordinates of bounding box [x_min, y_min, x_max, y_max]",
        min_items=4,
        max_items=4,
    )
    word: str = Field(..., description="Recognized text")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "bounding_box": [100, 100, 200, 150],
                "word": "Example",
            }
        },
    )


class OCRResponse(BaseModel):
    """Response model for OCR processing."""

    results: list[list[OCRResult]] = Field(
        ..., description="list of OCR results from the document"
    )

    page_count: int = Field(1, description="Number of pages processed", ge=1)

    model_config = ConfigDict(frozen=True)
