from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, NotRequired, TypedDict

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs.model_config import ModelConfig

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


class ClassificationResponse(BaseModel):
    """Structured response from the classification model."""

    class_label: str = Field(..., description="Predicted document class")
    confidence: float = Field(..., description="Prediction confidence score")


class OllamaOptions(TypedDict):
    """Type definition for Ollama API options."""

    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    num_ctx: int
    num_gpu: int
    num_thread: int


class OllamaRequest(TypedDict, total=False):
    """Type definition for Ollama API request."""

    model: str  # required
    prompt: str  # required
    format: NotRequired[str]
    options: NotRequired[OllamaOptions]
    system: NotRequired[str]
    stream: NotRequired[bool]
    raw: NotRequired[bool]
    keep_alive: NotRequired[str | int]


class LLMModelWrapper:
    """Document classification using direct Ollama integration."""

    CLASSIFICATION_SYSTEM_PROMPT = """You are a document classification expert tasked with precise analysis.
    Core Requirements:
    1. Analyze text provided in {language}
    2. Classify into categories: {categories}
    3. Output JSON format: {{"class_label": "category", "confidence": number}}

    Guidelines:
    - Evaluate based on content, style, terminology and format
    - Look for distinctive markers and key phrases
    - Consider document structure and formatting patterns
    - Maintain consistent standards across documents
    - Use confidence scores that realistically reflect certainty
    - Confidence range: 0-100
    - Return clean JSON only, no other text"""

    CLASSIFICATION_PROMPT = """Task Analysis:
    You are examining a document in {language} to determine its category from: {categories}

    Document Classification Guidelines:
    1. Analyze these key aspects:
        - Subject matter and main themes
        - Technical terminology usage
        - Writing style and tone
        - Document structure and sections
        - Target audience indicators
        - Domain-specific formatting

    2. Evidence Collection:
        - Identify definitive category markers
        - Note absence of expected features
        - Compare against typical examples
        - Check for mixed/ambiguous signals
        - Evaluate consistency throughout

    3. Confidence Scoring:
        - 90-100: Unambiguous match with strong indicators
        - 70-89: Clear match with some uncertainty
        - 50-69: Probable match with mixed signals
        - Below 50: Significant uncertainty

    Input Document:
    {text}

    Required Response Format:
    - Pure JSON object
    - Must include class_label and confidence
    - Example: {{"class_label": "example_category", "confidence": 85}}
    - No explanatory text outside JSON

    Respond with classification:"""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize classifier with configuration."""
        self.config = config
        self.base_url = f"{config.OLLAMA_BASE_URL}/api/generate"

    def _prepare_request(
        self,
        text: str,
        language: str,
    ) -> OllamaRequest:
        """Prepare the request payload for Ollama API."""
        system_prompt = self.CLASSIFICATION_SYSTEM_PROMPT.format(
            language=language,
            categories=self.config.DOCUMENT_CLASSES,
        )

        user_prompt = self.CLASSIFICATION_PROMPT.format(
            text=text,
            language=language,
            categories=self.config.DOCUMENT_CLASSES,
        )

        return {
            "model": self.config.MODEL_NAME,
            "prompt": user_prompt,
            "system": system_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.config.TEMPERATURE,
                "top_p": self.config.TOP_P,
                "frequency_penalty": self.config.FREQUENCY_PENALTY,
                "presence_penalty": self.config.PRESENCE_PENALTY,
                "num_ctx": self.config.NUM_CTX,
                "num_gpu": self.config.NUM_GPU,
                "num_thread": self.config.NUM_THREAD,
            },
            "keep_alive": "15m",
        }

    def _process_raw_response(self, content: str) -> ClassificationResponse:
        """Process raw response when JSON parsing fails.

        Args:
            content: Raw response content from the model

        Returns:
            Structured classification response

        Raises:
            ValueError: If unable to extract valid classification

        """
        # Remove any potential markdown code blocks
        content = re.sub(r"```json\s*|\s*```", "", content)

        # Try to find class and confidence using regex
        class_match = re.search(
            r"class_label[\"']?\s*:\s*[\"']([^\"']+)[\"']",
            content,
        )
        conf_match = re.search(
            r"confidence[\"']?\s*:\s*(0?\.\d+|1\.0?|1|0)",
            content,
        )

        if not (class_match and conf_match):
            raise ValueError(
                "Unable to extract valid classification from response",
            )

        class_label = class_match.group(1)
        confidence = float(conf_match.group(1))

        if class_label not in self.config.DOCUMENT_CLASSES:
            raise ValueError(f"Invalid class label: {class_label}")

        if not 0 <= confidence <= 100:
            logger.warning(
                "Invalid confidence score: %s, calculating...",
                confidence,
            )
            confidence = min(100, confidence)

        return ClassificationResponse(
            class_label=class_label,
            confidence=confidence,
        )

    def _validate_classification_response(
        self,
        response_data: dict[str, Any],
    ) -> ClassificationResponse:
        """Validate and process the classification response.

        Args:
            response_data: Parsed JSON response

        Returns:
            Validated classification response

        Raises:
            ValueError: If response doesn't meet requirements

        """
        if not isinstance(response_data, dict):
            raise ValueError("Response must be a JSON object")

        if (
            "class_label" not in response_data
            or "confidence" not in response_data
        ):
            raise ValueError(
                'Response must contain "class_label" and "confidence" fields',
            )

        class_label = response_data["class_label"]
        confidence = response_data["confidence"]

        if not isinstance(class_label, str) or not isinstance(
            confidence,
            (int, float),
        ):
            raise ValueError("Invalid field types in response")

        if class_label not in self.config.DOCUMENT_CLASSES:
            raise ValueError(f"Invalid class label: {class_label}")

        if not 0 <= confidence <= 100:
            logger.warning(
                "Invalid confidence score: %s, calculating...",
                confidence,
            )
            confidence = min(100, confidence)

        return ClassificationResponse(
            class_label=class_label,
            confidence=float(confidence),
        )

    async def predict_class(
        self,
        text: str,
        language: str,
    ) -> str:
        """Predict document class using the LLM model.

        Args:
            text: Document text to classify
            language: Language of the document

        Returns:
            Predicted document class in format "class_label|confidence"

        Raises:
            HTTPException: If classification fails

        """

        def _raise_invalid_response():
            msg = "Invalid response format from Ollama"
            raise ValueError(msg)

        def _raise_empty_response():
            msg = "Empty response from Ollama"
            raise ValueError(msg)

        try:
            request_data = self._prepare_request(text, language)

            timeout = aiohttp.ClientTimeout(total=300)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(
                    self.base_url,
                    json=request_data,
                ) as response,
            ):
                response.raise_for_status()
                result = await response.json()

                if not isinstance(result, dict) or "response" not in result:
                    _raise_invalid_response()

                content = result["response"].strip()
                if not content:
                    _raise_empty_response()

                try:
                    # Try to parse as JSON first
                    response_data = json.loads(content)
                    classification = self._validate_classification_response(
                        response_data,
                    )
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Failed to parse JSON response, falling back to raw "
                        "processing: %s",
                        str(e),
                    )
                    # Fall back to raw response processing
                    classification = self._process_raw_response(content)

                # Return in the expected format "class_label|confidence"
                return (
                    f"{classification.class_label}|{classification.confidence}"
                )

        except aiohttp.ClientError as e:
            logger.exception("Ollama API communication error")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to communicate with Ollama: {e!s}",
            ) from e

        except ValueError as e:
            logger.exception("Invalid response format from Ollama")
            raise HTTPException(
                status_code=500,
                detail=str(e),
            ) from e

        except Exception as e:
            logger.exception("Classification failed")
            raise HTTPException(
                status_code=500,
                detail=f"Classification failed: {e!s}",
            ) from e


# Create model instance
llm_model = LLMModelWrapper(config)
