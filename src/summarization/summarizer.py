import logging
from datetime import datetime
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_ollama.chat_models import ChatOllama

from configs.summarization_config import SummarizationConfig
from database import DocumentError, DocumentRepository, get_repository
from payload.summarizer_models import SummarizationRequest

config = SummarizationConfig()
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


class Summarizer:
    """Text summarization using Ollama and LangChain."""

    SUMMARY_SYSTEM_PROMPT = """You are a precise summarization assistant that creates database-friendly summaries. Follow these rules exactly:
    1. Output only the summary text, no introductions or extra words
    2. Write complete sentences without periods at the end (use spaces between sentences)
    3. Stay strictly between {min_length} and {max_length} words
    4. Write in a formal tone appropriate for {classification} documents
    5. Use clear connecting words between sentences for flow"""

    SUMMARY_PROMPT = """<Instructions>
    Write a summary of the {classification} text below
    The summary must be between {min_length} and {max_length} words
    Do not use periods at the end of sentences, only between sentences
    Focus on key information relevant to {classification} type
    Only output the summary text, nothing else
    </Instructions>

    <Text>
    {text}
    </Text>

    <OutputRules>
    - Only output the summary
    - No introduction or meta text
    - No periods at end of sentences
    - Must be {min_length}-{max_length} words
    </OutputRules>

    Summary:"""

    def __init__(self):
        """Initialize summarizer with Ollama model."""
        self.llm = ChatOllama(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            base_url=config.OLLAMA_BASE_URL,
            frequency_penalty=config.FREQUENCY_PENALTY,
            presence_penalty=config.PRESENCE_PENALTY,
            num_ctx=config.NUM_CTX,
            num_gpu=config.NUM_GPU,
            num_thread=config.NUM_THREAD,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.SUMMARY_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(self.SUMMARY_PROMPT),
            ]
        )

        self.chain = self.prompt | self.llm

    async def generate_summary(
        self,
        text: str,
        min_length: int = 100,
        max_length: int = 300,
        classification: str = "unclassified",
    ) -> str:
        """
        Generate a summary of the input text.

        Args:
            text: Input text to summarize
            min_length: Minimum summary length in words
            max_length: Maximum summary length in words

        Returns:
            Generated summary text
        """
        try:
            response = await self.chain.ainvoke(
                {
                    "text": text,
                    "min_length": min_length,
                    "max_length": max_length,
                    "classification": classification,
                }
            )
            return response.content.strip().replace(" \n", ". ")
        except Exception as e:
            logger.error("Summarization failed: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail="Summarization failed",
            )


summarizer = Summarizer()


@app.post("/summarize")
async def summarize_document(
    request: SummarizationRequest,
    repository: AsyncGenerator[DocumentRepository, None] = Depends(get_repository),
) -> dict[str, str]:
    """
    Generate a summary for the given document.

    Args:
        request: Summarization request containing text and parameters
        repository: Document repository instance

    Returns:
        Dictionary containing the generated summary
    """
    try:
        summary = await summarizer.generate_summary(
            text=request.text,
            min_length=config.DEFAULT_MIN_LENGTH,
            max_length=config.DEFAULT_MAX_LENGTH,
            classification=request.classification,
        )

        # Update document in database with summary
        try:
            await repository.update_summary(request.file_name, summary)
        except DocumentError as e:
            logger.error("Failed to update summary: %s", str(e))
            raise HTTPException(
                status_code=500,
                detail="Failed to save summary",
            )

        return {"summary": summary, "classification": request.classification}

    except Exception as e:
        logger.exception("Summarization endpoint failed")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
