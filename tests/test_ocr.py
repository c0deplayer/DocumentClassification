import concurrent.futures
import sys
from pathlib import Path

import requests
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.documentclassification.ocr.ocr import app


def send_request(file_path: Path):
    with file_path.open("rb") as file:
        response = client.post("/ocr", files={"file": file})
        assert response.status_code == 200
        assert "ocr_result" in response.json()
        print(f"Response JSON for {file_path.name}: {response.json()['ocr_result']}")

        requests.post("http://127.0.0.1:8000/process", json=response.json())


def test_multiple_calls():
    test_files = [
        Path("/Users/codeplayer/Downloads/IST-DS2_116901_Jakub_Kujawa_LABI.pdf"),
        Path(
            "/Users/codeplayer/Documents/LaTeX/Administracja Sieciami Komputerowymi - LAB II/img/zad1/SS-8.jpg"
        ),
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request, test_file) for test_file in test_files]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    client = TestClient(app)

    _test_file = Path(
        "/Users/codeplayer/Documents/LaTeX/Administracja Sieciami Komputerowymi - LAB II/img/zad1/SS-8.jpg"
    )

    send_request(_test_file)
