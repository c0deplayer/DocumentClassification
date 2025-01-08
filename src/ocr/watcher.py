from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Final

import aiofiles
import aiohttp
import fastapi
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from configs.ocr_config import OCRConfig

# Constants
RETRY_ATTEMPTS: Final[int] = 3
RETRY_DELAY: Final[int] = 5  # seconds

config = OCRConfig()
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


class DocumentWatcher(FileSystemEventHandler):
    """Watches for new documents and sends them to OCR service via HTTP."""

    def __init__(
        self,
        watch_directory: Path,
        service_url: str = "http://localhost:8080",
    ) -> None:
        """Initialize the document watcher.

        Args:
            watch_directory: Directory to monitor for new files
            service_url: URL of the OCR service (default: http://localhost:8080)

        """
        self.watch_directory = watch_directory
        self.service_url = service_url.rstrip("/")
        self.processing_queue: asyncio.Queue[Path] = asyncio.Queue()
        self.event_queue: Queue[tuple[str, Path]] = Queue()
        self._is_running = False
        self._observer: Observer | None = None
        self._session: aiohttp.ClientSession | None = None
        self._processing_files: set[str] = set()  # Track files being processed

    async def start(self) -> None:
        """Start the watcher and processing loop."""
        self._is_running = True
        self._observer = Observer()
        self._observer.schedule(
            self,
            str(self.watch_directory),
            recursive=False,
        )
        self._observer.start()

        # Create aiohttp session
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        self._session = aiohttp.ClientSession(timeout=timeout)

        try:
            while self._is_running:
                try:
                    # Process events from the sync queue
                    while not self.event_queue.empty():
                        event_type, file_path = self.event_queue.get_nowait()
                        if (
                            event_type in ("created", "closed", "modified")
                            and file_path.name not in self._processing_files
                        ):
                            self._processing_files.add(file_path.name)
                            await self.processing_queue.put(file_path)
                        self.event_queue.task_done()

                    # Process files from the async queue
                    try:
                        file_path = await asyncio.wait_for(
                            self.processing_queue.get(),
                            timeout=1.0,
                        )
                        await self._process_file(file_path)
                        self.processing_queue.task_done()
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        logger.exception(
                            "Error in processing loop: %s",
                        )
                        await asyncio.sleep(1)
                except Exception:
                    logger.exception(
                        "Error in main loop: %s",
                    )
                    await asyncio.sleep(1)

        finally:
            if self._observer:
                self._observer.stop()
                self._observer.join()
            if self._session:
                await self._session.close()

    async def stop(self) -> None:
        """Stop the watcher and processing loop."""
        self._is_running = False
        if self._session:
            await self._session.close()
        if self._observer:
            self._observer.stop()
            self._observer.join()

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        self._handle_file_event(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return
        self._handle_file_event(event.src_path, "modified")

    def on_closed(self, event: FileSystemEvent) -> None:
        """Handle file close events."""
        if event.is_directory:
            return
        self._handle_file_event(event.src_path, "closed")

    def _handle_file_event(self, file_path: str, event_type: str) -> None:
        """Handle various file events.

        Args:
            file_path: Path to the file
            event_type: Type of event (created, modified, closed)

        """
        path = Path(file_path)
        logger.info(
            "File %s event: %s",
            path.name,
            event_type,
        )
        self.event_queue.put((event_type, path))

    async def _attempt_ocr_request(self, file_path: Path) -> bool:
        """Attempt to open and send the file to OCR. Return True if successful, False otherwise."""
        try:
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()

            data = aiohttp.FormData()
            data.add_field(
                "file",
                content,
                filename=file_path.name,
                content_type=self._get_content_type(file_path),
            )

            logger.info(
                "Sending file %s to OCR service",
                file_path.name,
            )
            async with self._session.post(
                f"{self.service_url}/ocr",
                data=data,
            ) as response:
                if response.status == fastapi.status.HTTP_200_OK:
                    logger.info(
                        "Successfully processed file: %s",
                        file_path.name,
                    )
                    try:
                        file_path.unlink()
                        logger.info("Deleted file: %s", file_path.name)
                    except OSError:
                        logger.exception(
                            "Error deleting file %s",
                            file_path.name,
                        )
                    return True

                logger.error(
                    "Failed to process file %s. Status: %d",
                    file_path.name,
                    response.status,
                )
        except Exception:
            logger.exception(
                "Error processing file %s",
                file_path.name,
            )
        return False

    async def _process_file(self, file_path: Path) -> None:
        """Process a file by sending it to the OCR service API."""
        if not self._session:
            logger.error("HTTP session not initialized")
            self._processing_files.discard(file_path.name)
            return

        if not file_path.exists():
            logger.error("File %s does not exist", file_path)
            self._processing_files.discard(file_path.name)
            return

        if file_path.suffix.lower() not in {
            ".pdf",
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            logger.warning("Unsupported file type: %s", file_path.suffix)
            self._processing_files.discard(file_path.name)
            return

        try:
            # Wait a short time to ensure file is completely written
            await asyncio.sleep(5)

            success = False
            for attempt in range(RETRY_ATTEMPTS):
                if await self._attempt_ocr_request(file_path):
                    success = True
                    break
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_DELAY)

            if not success:
                logger.error(
                    "Failed to process file after %d attempts: %s",
                    RETRY_ATTEMPTS,
                    file_path.name,
                )
        finally:
            self._processing_files.discard(file_path.name)

    @staticmethod
    def _get_content_type(file_path: Path) -> str:
        """Get the MIME type for a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            str: MIME type string

        """
        extension_map = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return extension_map.get(
            file_path.suffix.lower(),
            "application/octet-stream",
        )
