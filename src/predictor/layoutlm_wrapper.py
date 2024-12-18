import lightning.pytorch as pl
import torch
from torchmetrics import Accuracy
from transformers import LayoutLMv3ForSequenceClassification

from configs.model_config import ModelConfig


class LayoutLMV3Wrapper(pl.LightningModule):
    """PyTorch Lightning module for document classification using LayoutLMv3."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the model.

        Args:
            n_classes: Number of document classes to predict

        """
        super().__init__()
        self.n_classes = len(config.DOCUMENT_CLASSES)
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "/app/data/models/layoutlmv3-base-finetuned-rvlcdip",
            num_labels=self.n_classes,
            use_safetensors=True,
        )
        self.config = config
        self.model.config.id2label = dict(enumerate(config.DOCUMENT_CLASSES))
        self.model.config.label2id = {
            v: k for k, v in enumerate(config.DOCUMENT_CLASSES)
        }
        self.train_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
        )
        self.val_accuracy = Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> object:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            bbox: Bounding box coordinates
            pixel_values: Image pixel values

        Returns:
            Model output containing logits and loss

        """
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
        )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Training step for one batch.

        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of current batch (unused)

        Returns:
            Training loss

        """
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

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Execute validation step for one batch.

        Args:
            batch: Dictionary containing batch data
            batch_idx: Index of current batch (unused)

        Returns:
            Validation loss

        """
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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizers for training.

        Returns:
            Adam optimizer instance

        """
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)
