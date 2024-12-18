import logging
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

import aiohttp
import torch
import torch.nn.functional as f
from fastapi import Depends, FastAPI, HTTPException
from layoutlm_wrapper import LayoutLMV3Wrapper
from lingua import Language
from llm_model import LLMModelWrapper
from transformers import PreTrainedModel

from configs.model_config import ModelConfig
from database import DocumentError, DocumentRepository, get_repository
from payload.predictor_models import PagePrediction
from payload.shared_models import ProcessorResult

config = ModelConfig()
config.LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG if config.LOG_LEVEL == "DEBUG" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            config.LOG_DIR / f"{datetime.now():%Y-%m-%d}.log",
            mode="a",
        ),
        logging.StreamHandler(),
    ],
)


logger = logging.getLogger(__name__)

app = FastAPI()


layoutlm_model = LayoutLMV3Wrapper(config)
layoutlm_model.eval()

llm_model = LLMModelWrapper(config)


def _calculate_page_weights(
    page_predictions: list[PagePrediction],
) -> torch.Tensor:
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
        zip(weights, text_lengths, confidences),
    ):
        logger.debug(
            "Page %(page)d pre-normalization: weight=%(weight).4f, text_length=%(length)d, confidence=%(conf).4f",
            {
                "page": i + 1,
                "weight": raw_weight,
                "length": length,
                "conf": conf,
            },
        )

    temperature = 1.0
    normalized_weights = f.softmax(weights / temperature, dim=0)

    for i, weight in enumerate(normalized_weights):
        logger.debug(
            "Page %(page)d final weight: %(weight).4f",
            {"page": i + 1, "weight": weight},
        )

    return normalized_weights


def aggregate_predictions(
    model: PreTrainedModel,
    page_predictions: list[PagePrediction],
) -> dict[str, Any]:
    """Aggregate predictions from multiple pages using weighted voting.

    Args:
        model: Pre-trained model containing label mapping configuration
        page_predictions: List of predictions for each page

    Returns:
        Dictionary containing final classification results

    """
    if not page_predictions:
        msg = "No valid predictions to aggregate"
        raise ValueError(msg)

    weights = _calculate_page_weights(page_predictions)

    stacked_logits = torch.stack([pred.logits for pred in page_predictions])
    weighted_logits = (stacked_logits * weights.unsqueeze(-1)).sum(dim=0)

    probabilities = f.softmax(weighted_logits, dim=-1)
    confidence, predicted_class = torch.max(probabilities, dim=-1)

    return {
        "predicted_class": model.config.id2label[int(predicted_class.item())],
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
    repository: AsyncGenerator[DocumentRepository, None] = Depends(
        get_repository,
    ),
) -> dict[str, str]:
    """Predict document class from processed data.

    Args:
        data: Processed data from document processor
        repository: Document repository instance

    Returns:
        Dictionary containing prediction results and summary

    """
    try:
        logger.info("Received data for prediction")

        # Initialize variables for prediction results
        detailed_results = None

        # Choose prediction method based on language
        if getattr(Language, data.language) != Language.ENGLISH:
            # Use LLM for non-English documents
            logger.info("Using LLM model for non-English document")
            prediction_result = await llm_model.predict_class(
                data.text,
                data.language,
            )
            logger.debug("LLM prediction result: %s", prediction_result)
            # Parse the LLM output (format: "class_label|confidence_score")
            predicted_class, confidence = prediction_result.split("|")
            confidence = float(confidence) / 100

            detailed_results = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "page_predictions": [
                    {
                        "page": 0,
                        "class": predicted_class,
                        "confidence": confidence,
                        "weight": 1.0,
                    },
                ],
                "probability_distribution": {
                    predicted_class: confidence,
                },
            }
        else:
            # Use LayoutLM for English documents
            logger.info("Using LayoutLM model for English document")
            input_ids = torch.tensor(data.input_ids)
            attention_mask = torch.tensor(data.attention_mask)
            bbox = torch.tensor(data.bbox)
            pixel_values = torch.tensor(data.pixel_values)

            with torch.inference_mode():
                output = layoutlm_model(
                    input_ids,
                    attention_mask,
                    bbox,
                    pixel_values,
                )

            probabilities = f.softmax(output.logits, dim=-1)
            confidence, predicted_class = torch.max(probabilities, dim=-1)

            predictions = []
            for i in range(output.logits.size(0)):
                pred_class = int(predicted_class[i].item())
                conf = confidence[i].item()
                text_length = int(torch.sum(attention_mask[i]).item())

                predictions.append(
                    PagePrediction(
                        page_number=i,
                        logits=output.logits[i],
                        confidence=conf,
                        predicted_class=layoutlm_model.model.config.id2label[
                            pred_class
                        ],
                        text_length=text_length,
                    ),
                )

            detailed_results = aggregate_predictions(
                layoutlm_model.model,
                predictions,
            )

        logger.debug("Prediction results: %s", detailed_results)
        logger.info("Predicted class: %s", detailed_results["predicted_class"])

        # Update classification in repository
        try:
            await repository.update_classification(
                data.file_name,
                detailed_results["predicted_class"],
            )
        except DocumentError as e:
            logger.exception("Failed to update classification")
            raise HTTPException(
                status_code=500,
                detail="Failed to update classification",
            ) from e

        # Send to summarizer
        timeout = aiohttp.ClientTimeout(total=480)
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                config.SUMMARIZER_URL,
                json={
                    "file_name": data.file_name,
                    "text": data.text,
                    "classification": detailed_results["predicted_class"],
                },
                timeout=timeout,
            ) as response,
        ):
            response.raise_for_status()
            return await response.json()

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e
