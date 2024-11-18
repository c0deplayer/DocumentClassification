from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict


@dataclass
class PagePrediction:
    """Stores prediction results for a single page."""

    page_number: int
    logits: torch.Tensor
    confidence: float
    predicted_class: str
    text_length: int


class ProcessorResult(BaseModel):
    """Result data for document processing."""

    input_ids: list
    attention_mask: list
    bbox: list
    pixel_values: list
    file_name: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "bbox": [[0, 0, 100, 50]],
                "pixel_values": [[0, 0, 0, 255, 255, 255]],
                "file_name": "example.pdf",
            }
        },
    )
