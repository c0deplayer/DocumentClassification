import logging
from datetime import datetime
from typing import Any, AsyncGenerator

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, HTTPException
from torchmetrics import Accuracy
from transformers import LayoutLMv3ForSequenceClassification, PreTrainedModel

from configs.model_config import ModelConfig
from database import DocumentError, DocumentRepository, get_repository
from payload.model_models import PagePrediction, ProcessorResult

config = ModelConfig()
config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if config.LOG_LEVEL == "DEBUG" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            config.LOG_DIR / f"{datetime.now():%Y-%m-%d}.log", mode="a"
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI()


class ModelModule(pl.LightningModule):
    def __init__(self, n_classes: int = 15):
        super().__init__()
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "/app/data/models/layoutlmv3-base-finetuned-rvlcdip",
            num_labels=n_classes,
            use_safetensors=True,
        )
        self.model.config.id2label = {
            k: v for k, v in enumerate(config.DOCUMENT_CLASSES)
        }
        self.model.config.label2id = {
            v: k for k, v in enumerate(config.DOCUMENT_CLASSES)
        }
        self.train_accuracy = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        output = self(input_ids, attention_mask, bbox, pixel_values, labels)

        self.log("train_loss", output.loss)
        self.log(
            "train_acc",
            self.train_accuracy(output.logits, labels),
            on_step=True,
            on_epoch=True,
        )

        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        bbox = batch["bbox"]
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]

        output = self(input_ids, attention_mask, bbox, pixel_values, labels)

        self.log("val_loss", output.loss)
        self.log(
            "val_acc",
            self.val_accuracy(output.logits, labels),
            on_step=False,
            on_epoch=True,
        )

        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        return optimizer


model = ModelModule(n_classes=len(config.DOCUMENT_CLASSES))
model.eval()


def _calculate_page_weights(page_predictions: list[PagePrediction]) -> torch.Tensor:
    """Calculate weights for each page based on multiple factors.

    Args:
        page_predictions: List of predictions for each page

    Returns:
        Tensor of normalized weights for each page
    """
    text_lengths = torch.tensor([pred.text_length for pred in page_predictions])
    confidences = torch.tensor([pred.confidence for pred in page_predictions])

    if text_lengths.max() > 0:
        normalized_lengths = text_lengths / text_lengths.max()
    else:
        normalized_lengths = torch.ones_like(text_lengths)

    weights = (normalized_lengths * confidences) + 1e-6

    for i, (raw_weight, length, conf) in enumerate(
        zip(weights, text_lengths, confidences)
    ):
        logger.debug(
            f"Page {i+1} pre-normalization: "
            f"weight={raw_weight:.4f}, "
            f"text_length={length}, "
            f"confidence={conf:.4f}"
        )

    temperature = 1.0
    normalized_weights = F.softmax(weights / temperature, dim=0)

    for i, weight in enumerate(normalized_weights):
        logger.debug(f"Page {i+1} final weight: {weight:.4f}")

    return normalized_weights


def aggregate_predictions(
    model: PreTrainedModel, page_predictions: list[PagePrediction]
) -> dict[str, Any]:
    """Aggregate predictions from multiple pages using weighted voting.

    Args:
        page_predictions: List of predictions for each page

    Returns:
        Dictionary containing final classification results
    """
    if not page_predictions:
        raise ValueError("No valid predictions to aggregate")

    weights = _calculate_page_weights(page_predictions)

    stacked_logits = torch.stack([pred.logits for pred in page_predictions])
    weighted_logits = (stacked_logits * weights.unsqueeze(-1)).sum(dim=0)

    probabilities = F.softmax(weighted_logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=-1)

    return {
        "predicted_class": model.config.id2label[predicted_class.item()],
        "confidence": confidence.item(),
        "page_predictions": [
            {
                "page": pred.page_number,
                "class": pred.predicted_class,
                "confidence": pred.confidence,
                "weight": weight.item(),
            }
            for pred, weight in zip(page_predictions, weights)
        ],
        "probability_distribution": {
            model.config.id2label[i]: prob.item()
            for i, prob in enumerate(probabilities)
        },
    }


@app.post("/predict-class")
async def predict(
    data: ProcessorResult,
    repository: AsyncGenerator[DocumentRepository, None] = Depends(get_repository),
) -> dict[str, str]:
    """
    Predict document class from processed data.

    Args:
        data: Processed data from document processor
        repository: Document repository instance
    """

    try:
        logger.info("Received data for prediction")
        input_ids = torch.tensor(data.input_ids)
        attention_mask = torch.tensor(data.attention_mask)
        bbox = torch.tensor(data.bbox)
        pixel_values = torch.tensor(data.pixel_values)

        with torch.inference_mode():
            output = model(input_ids, attention_mask, bbox, pixel_values)

        probabilities = F.softmax(output.logits, dim=-1)
        confidence, predicted_class = torch.max(probabilities, dim=-1)

        predictions = []
        for i in range(output.logits.size(0)):
            pred_class = predicted_class[i].item()
            conf = confidence[i].item()
            text_length = torch.sum(attention_mask[i]).item()

            predictions.append(
                PagePrediction(
                    page_number=i,
                    logits=output.logits[i],
                    confidence=conf,
                    predicted_class=model.model.config.id2label[pred_class],
                    text_length=text_length,
                )
            )

        detailed_results = aggregate_predictions(model.model, predictions)

        logger.debug("Prediction results: %s", detailed_results)
        logger.info("Predicted class: %s", detailed_results["predicted_class"])

        try:
            await repository.update_classification(
                data.file_name,
                detailed_results["predicted_class"],
            )
        except DocumentError as e:
            logger.error("Failed to update classification: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail="Failed to update classification",
            ) from e

        return {"predicted_classes": detailed_results["predicted_class"]}

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
