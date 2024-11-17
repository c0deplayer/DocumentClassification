import torch
from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    """Represents a bounding box for OCR results."""

    coordinates: list[int] = Field(
        ...,
        min_items=4,
        max_items=4,
        description="Coordinates in format [x_min, y_min, x_max, y_max]",
    )


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


class ProcessorInput(BaseModel):
    """Input data for document processing."""

    ocr_result: list[list[OCRResult]]
    images: list[str] = Field(..., description="Base64 encoded images")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ocr_result": [{"word": "Example", "bounding_box": [0, 0, 100, 50]}],
                "images": ["base64_encoded_image_string"],
            }
        }
    )


class ProcessorResult(BaseModel):
    """Result data for document processing."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    pixel_values: torch.Tensor

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "bbox": [[0, 0, 100, 50]],
                "pixel_values": [[0, 0, 0, 255, 255, 255]],
            }
        },
    )
