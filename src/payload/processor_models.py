from pydantic import BaseModel, ConfigDict, Field

from .shared_models import OCRResult


class ProcessorInput(BaseModel):
    """Input data for document processing."""

    ocr_result: list[list[OCRResult]]
    images: list[str] = Field(..., description="Base64 encoded images")
    file_name: str = Field(..., description="Name of the processed file")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ocr_result": [
                    {"word": "Example", "bounding_box": [0, 0, 100, 50]},
                ],
                "images": ["base64_encoded_image_string"],
            },
        },
    )


class ProcessorOutput(BaseModel):
    """Result data for document processing."""

    input_ids: list
    attention_mask: list
    bbox: list
    pixel_values: list

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "bbox": [[0, 0, 100, 50]],
                "pixel_values": [[0, 0, 0, 255, 255, 255]],
            },
        },
    )
