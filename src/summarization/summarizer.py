from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, NotRequired, TypedDict, cast

import aiohttp
from fastapi import Depends, FastAPI, HTTPException

from configs.summarization_config import SummarizationConfig
from database import DocumentError, DocumentRepository, get_repository
from payload.summarizer_models import SummarizationRequest, SummaryResponse

config = SummarizationConfig()
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


class OllamaOptions(TypedDict):
    """A TypedDict class representing configuration options for Ollama API.

    Attributes:
        temperature (float): Controls randomness in the response. Higher values (e.g., 0.8) make the output more random,
                            while lower values (e.g., 0.2) make it more focused and deterministic.
        top_p (float): Controls diversity via nucleus sampling. Restricts cumulative probability of tokens considered
                       for sampling. Range 0.0-1.0.
        frequency_penalty (float): Reduces likelihood of repeated tokens. Positive values discourage repetition while
                                 negative values encourage it.
        presence_penalty (float): Influences likelihood of discussing new topics. Positive values encourage covering
                                new topics while negative values focus on existing ones.
        num_ctx (int): Sets the size of the context window (in tokens) for the model's input/output.
        num_gpu (int): Specifies the number of GPUs to utilize for model inference.
        num_thread (int): Defines the number of CPU threads to use for processing.

    """

    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    num_ctx: int
    num_gpu: int
    num_thread: int


class OllamaRequest(TypedDict, total=False):
    """A TypedDict class representing a request to the Ollama API.

    Attributes:
        model (str): The name of the model to use for the request. Required.
        prompt (str): The prompt text to send to the model. Required.
        format (str, optional): The desired output format.
        options (OllamaOptions, optional): Additional options for the model.
        system (str, optional): System prompt to modify model behavior.
        stream (bool, optional): Whether to stream the response.
        raw (bool, optional): Whether to return raw response.
        keep_alive (Union[str, int], optional): Duration to keep the model loaded.

    """

    model: str  # required
    prompt: str  # required
    format: NotRequired[str]
    options: NotRequired[OllamaOptions]
    system: NotRequired[str]
    stream: NotRequired[bool]
    raw: NotRequired[bool]
    keep_alive: NotRequired[str | int]


class Summarizer:
    """A class for generating text summaries using the Ollama API.

    This class handles the preparation, validation, and processing of text summarization
    requests through the Ollama API. It includes functionality for JSON response parsing,
    error handling, and content validation.

    Attributes:
        SUMMARY_SYSTEM_PROMPT (str): Template for system-level instructions to the model.
        SUMMARY_PROMPT (str): Template for the main summarization prompt.
        config (SummarizationConfig): Configuration settings for the summarizer.
        base_url (str): Base URL for the Ollama API endpoint.

    Example:
        ```python
        config = SummarizationConfig()
        summarizer = Summarizer(config)
        summary = await summarizer.generate_summary(
            text="Long text to summarize",
            min_length=100,
            max_length=300,
            classification="technical"
        ```

    Note:
        The class expects the Ollama API to be available and properly configured.
        It handles responses in JSON format and includes fallback mechanisms for
        non-JSON responses.

        HTTPException: When API communication fails or response validation fails.
        ValueError: When response format or content validation fails.

    """

    SUMMARY_SYSTEM_PROMPT = """You are an expert summarization analyst specializing in precise content distillation.

    Core Requirements:
    1. Process text in target format: {classification}
    2. Generate summaries between {min_length}-{max_length} words
    3. Output JSON format: {{"summary": "extracted key points", "word_count": number}}

    Technical Guidelines:
    - Create self-contained, complete thought units
    - Use professional language suited to {classification} context
    - Employ clear transition phrases between ideas
    - Maintain consistent technical depth throughout
    - Structure content with logical progression
    - Focus on essential information density
    - Return only valid JSON output"""

    SUMMARY_PROMPT = """Document Analysis Parameters:
    - Document Type: {classification}
    - Length Constraints: {min_length} to {max_length} words
    - Output Format: {{"summary": "content", "word_count": number}}

    Summarization Guidelines:
    1. Content Analysis:
        - Identify core themes and key arguments
        - Extract essential supporting evidence
        - Preserve critical technical details
        - Maintain original document tone
        - Focus on {classification}-specific elements

    2. Summary Construction:
        - Begin with main thesis/findings
        - Include critical methodology details
        - Preserve key statistical data
        - Connect ideas with clear transitions
        - End with significant conclusions

    3. Technical Requirements:
        - Write complete semantic units
        - Use precise technical terminology
        - Maintain formal language register
        - Apply consistent formatting
        - Generate database-compatible output

    Source Document:
    {text}

    Response Specifications:
    - Pure JSON structure
    - Contains summary and word_count
    - Meets length requirements
    - No explanatory text
    - No terminal periods except between thoughts
    - Do not use points or bullet lists

    Generate summary:"""

    def __init__(self, config: SummarizationConfig) -> None:
        """Initialize the summarizer with configuration settings.

        Args:
            config (SummarizationConfig): Configuration object containing settings for summarization,
                including OLLAMA_BASE_URL for API endpoint.

        Returns:
            None

        Example:
            >>> config = SummarizationConfig(OLLAMA_BASE_URL="http://localhost:11434")
            >>> summarizer = Summarizer(config)

        """
        self.config = config
        self.base_url = f"{config.OLLAMA_BASE_URL}/api/generate"

    def _prepare_request(
        self,
        text: str,
        min_length: int,
        max_length: int,
        classification: str,
    ) -> OllamaRequest:
        """Prepare a request for the Ollama API to generate a summary of the given text.

        This method constructs a request dictionary with the necessary parameters for text summarization,
        including system and user prompts, model configuration, and various generation parameters.

        Args:
            text (str): The input text to be summarized.
            min_length (int): The minimum length of the generated summary.
            max_length (int): The maximum length of the generated summary.
            classification (str): The classification category of the text.

        Returns:
            OllamaRequest: A dictionary containing all necessary parameters for the Ollama API request,
            including:
                - model: The name of the language model to use
                - prompt: The formatted user prompt
                - system: The formatted system prompt
                - format: The expected response format (json)
                - stream: Whether to stream the response
                - options: Model-specific parameters (temperature, top_p, etc.)
                - keep_alive: Duration to keep the model loaded

        """
        system_prompt = self.SUMMARY_SYSTEM_PROMPT.format(
            min_length=min_length,
            max_length=max_length,
            classification=classification,
        )

        user_prompt = self.SUMMARY_PROMPT.format(
            text=text,
            min_length=min_length,
            max_length=max_length,
            classification=classification,
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

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        r"""Extract a JSON object from a text string.

        This method processes the input text by removing any markdown JSON code block markers
        and attempts to parse the first JSON object found in the text.

        Args:
            text (str): The input text containing a JSON object, potentially within markdown code blocks.

        Returns:
            dict[str, Any]: The parsed JSON object as a dictionary.

        Raises:
            ValueError: If no JSON object is found in the text or if the JSON parsing fails.

        Example:
            >>> text = "```json\\n{\\\"key\\\": \\\"value\\\"}\\n```"
            >>> result = _extract_json_from_text(text)
            >>> print(result)
            {'key': 'value'}

        """
        # Remove any markdown code blocks
        text = re.sub(r"```json\s*|\s*```", "", text)

        # Try to find JSON pattern
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON object found in response")

        try:
            return cast(dict[str, Any], json.loads(json_match.group()))
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e!s}") from e

    def _process_raw_response(self, content: str) -> SummaryResponse:
        """Process raw response when JSON parsing fails."""
        # Remove any potential markdown code blocks
        content = re.sub(r"```json\s*|\s*```", "", content)

        # Try to find content that looks like a summary
        text = re.sub(r"\s+", " ", content).strip()

        # Count words in the extracted text
        word_count = len(text.split())

        if not text:
            raise ValueError("Unable to extract valid summary from response")

        return SummaryResponse(
            summary=text,
            word_count=word_count,
        )

    def _validate_summary_response(
        self,
        response_data: dict[str, Any],
        min_length: int,
        max_length: int,
    ) -> SummaryResponse:
        """Validate and process the summary response."""
        if not isinstance(response_data, dict):
            raise ValueError("Response must be a JSON object")

        if "summary" not in response_data or "word_count" not in response_data:
            raise ValueError(
                'Response must contain "summary" and "word_count" fields',
            )

        summary = response_data["summary"]
        word_count = response_data["word_count"]

        if not isinstance(summary, str) or not isinstance(
            word_count,
            (int, float),
        ):
            raise ValueError("Invalid field types in response")

        # Convert word_count to int if it's a float
        word_count = int(word_count)

        actual_word_count = len(summary.split())
        if actual_word_count < min_length or actual_word_count > max_length:
            logger.warning(
                "Summary length (%d words) outside allowed range",
                actual_word_count,
            )

        return SummaryResponse(
            summary=summary,
            word_count=actual_word_count,
        )

    async def generate_summary(
        self,
        text: str,
        min_length: int = 100,
        max_length: int = 300,
        classification: str = "unclassified",
    ) -> str:
        """Generate a summary of the input text."""

        def _validate_ollama_response(result: object) -> None:
            """Validate Ollama API response format."""
            error_msg = "Invalid response format from Ollama"
            if not isinstance(result, dict) or "response" not in result:
                raise ValueError(error_msg)

        def _validate_content(content: str) -> None:
            """Validate response content is not empty."""
            error_msg = "Empty response from Ollama"
            if not content:
                raise ValueError(error_msg)

        try:
            request_data = self._prepare_request(
                text,
                min_length,
                max_length,
                classification,
            )

            timeout = aiohttp.ClientTimeout(total=300)
            async with (
                aiohttp.ClientSession(timeout=timeout) as session,
                session.post(self.base_url, json=request_data) as response,
            ):
                response.raise_for_status()
                result = await response.json()

                _validate_ollama_response(result)

                content = result["response"].strip()
                _validate_content(content)

                logger.debug("Raw response from Ollama: %s", content)

                try:
                    # Try to extract and parse JSON from response
                    response_data = self._extract_json_from_text(content)
                    summary_response = self._validate_summary_response(
                        response_data,
                        min_length,
                        max_length,
                    )
                    # Return only the summary text
                    return summary_response.summary.replace(" \n", ".")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        "Failed to parse JSON response, falling back to raw "
                        "processing: %s",
                        str(e),
                    )
                    # Fall back to raw response processing
                    summary_response = self._process_raw_response(content)
                    return summary_response.summary.replace(" \n", ".")

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
            logger.exception("Summarization failed")
            raise HTTPException(
                status_code=500,
                detail=f"Summarization failed: {e!s}",
            ) from e


summarizer = Summarizer(config)


@app.post("/summarize")
async def summarize_document(
    request: SummarizationRequest,
    repository: DocumentRepository = Depends(get_repository),
) -> dict[str, str]:
    """Generate a summary for the given document."""
    try:
        summary = await summarizer.generate_summary(
            text=request.text,
            min_length=config.DEFAULT_MIN_LENGTH,
            max_length=config.DEFAULT_MAX_LENGTH,
            classification=request.classification,
        )

        try:
            await repository.update_summary(request.file_name, summary)
        except DocumentError as e:
            logger.exception("Failed to update summary")
            raise HTTPException(
                status_code=500,
                detail="Failed to save summary",
            ) from e

    except Exception as e:
        logger.exception("Summarization endpoint failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e

    else:
        return {
            "file_name": request.file_name,
            "summary": summary,
            "classification": request.classification,
        }
