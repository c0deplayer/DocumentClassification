from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class OCRResult(BaseModel):
    """Structured OCR result using Pydantic BaseModel."""

    bounding_box: list[int] = Field(
        default=...,
        min_length=4,
        max_length=4,
        description="Coordinates in format [x_min, y_min, x_max, y_max]",
    )
    word: str = Field(..., description="Recognized text")

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "bounding_box": [100, 100, 200, 150],
                "word": "Example",
            },
        },
    )


class ProcessorResult(BaseModel):
    """Result data for document processing."""

    input_ids: list | None
    attention_mask: list | None
    bbox: list | None
    pixel_values: list | None
    file_name: str = Field(..., description="Name of the processed file")
    text: str = Field(..., description="Text extracted from OCR")
    language: str = Field(..., description="Detected language")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "bbox": [[0, 0, 100, 50]],
                "pixel_values": [[0, 0, 0, 255, 255, 255]],
                "file_name": "example.pdf",
                "text": "Example text",
                "language": "en",
            },
        },
    )
