from dataclasses import dataclass

import torch


@dataclass
class PagePrediction:
    """Stores prediction results for a single page."""

    page_number: int
    logits: torch.Tensor
    confidence: float
    predicted_class: str
    text_length: int
