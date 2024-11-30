# Klasyfikacja Dokumentów

[![en](https://img.shields.io/badge/lang-en-green.svg)](./README.en.md)
[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)
[![Build and Push Docker Images](https://github.com/c0deplayer/DocumentClassification/actions/workflows/docker-build.yml/badge.svg?branch=main)](https://github.com/c0deplayer/DocumentClassification/actions/workflows/docker-build.yml)

## Cel projektu

Projekt koncentruje się na implementacji zaawansowanego systemu klasyfikacji dokumentów, wykorzystującego potencjał chmury obliczeniowej oraz technik uczenia maszynowego. System został zaprojektowany z myślą o automatyzacji procesów analizy i kategoryzacji różnorodnych dokumentów.

---

## Wykorzystane technologie

### Podstawowe komponenty
- **Framework ML**: [PyTorch 2.5.0](https://pytorch.org/)
- **Model bazowy**: [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- **OCR Engine**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) oraz [Tesseract](https://github.com/tesseract-ocr/tesseract)
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Konteneryzacja**: [Docker](https://www.docker.com/) z obrazami wieloarchitekturowymi (amd64/arm64)

### Narzędzia pomocnicze
- **Zarządzanie zależnościami**: [PDM](https://pdm-project.org/en/latest/)
- **Formatowanie kodu:** [Ruff](https://docs.astral.sh/ruff/)
- **CI/CD:** [GitHub Actions](https://docs.github.com/en/actions)
- **Przetwarzanie obrazów**: [Pillow](https://python-pillow.org/), [pdf2image](https://pdf2image.readthedocs.io/en/latest/index.html)
  
## Wymagane biblioteki

> [!IMPORTANT]
> Minimalna wspierana wersja Python to 3.12.

Wszystkie wymagane biblioteki znajdują się w pliku `pyproject.toml` oraz `requirements.txt`. Aby zainstalować wszystkie
zależności, należy uruchomić poniższą komendę:

```bash
pip install -r requirements.txt
```

lub przy użyciu `pdm`:

```bash
pdm install
```
---

## Wymagania funkcjonalne

### Podstawowe

#### Obsługa dokumentów

* Wsparcie dla różnorodnych formatów dokumentów (PDF, JPEG, PNG, WEBP)
* Przetwarzanie dokumentów w języku angielskim
* Klasyfikacja dokumentów z założonym opóźnieniem do 15 sekund

#### Bezpieczeństwo

* Bezpieczne przechowywanie dokumentów z szyfrowaniem
* Logowanie operacji systemowych

#### Integracja i API

* REST API zgodne ze standardem OpenAPI 3.0

* Endpointy do:
    - Przesyłania dokumentów
    - Pobierania wyników klasyfikacji
    - Zarządzania dokumentami
    - Wyszukiwania i filtrowania

* Pełna dokumentacja API w formacie OpenAPI

#### Przechowywanie danych

* Baza danych do przechowywania:
    - Dokumentów w formie zaszyfrowanej
    - Wyników klasyfikacji
    - Adnotacji

* CI/CD do automatyzacji procesu wdrożenia
  - Wsparcie dla kontenerów wieloarchitekturowych
  - Automatyczne budowanie obrazów Docker

### Dodatkowe (opcjonalne)

#### Wielojęzyczność

* Obsługa przynajmniej języka polskiego i angielskiego
* Automatyczne wykrywanie języka dokumentu

#### Analiza i przetwarzanie

* Ekstrakcja kluczowych informacji z dokumentu
* Analiza układu dokumentu

#### Interfejs użytkownika

* Intuicyjny interfejs webowy do:

    - Przeglądania dokumentów według kategorii
    - Zarządzania klasyfikacją
    - Wizualizacji statystyk
    - Eksportu danych i raportów

---

## Ograniczenia systemu

### Ograniczenia plików

#### Dokumenty tekstowe

* Formaty: PDF
* Maksymalny rozmiar: 20MB
* Limit stron: 8

#### Obrazy i skany

* Formaty zdjęć: JPEG, PNG, WEBP
    - Maksymalny rozmiar: 20MB
    - Maksymalna rozdzielczość: 2480px x 3508px
* Akceptowane tryby kolorów: RGB, CMYK, skala szarości

### Ograniczenia przetwarzania

#### Generowanie opisów

* Długość opisu:
    - Minimum: 100 słów
    - Maksimum: 300 słów
* Czas przetwarzania:
    - Generowanie opisu: maksymalnie 10 sekund
    - Klasyfikacja dokumentu: maksymalnie 15 sekund
    - OCR: maksymalnie 15 sekund na stronę

---

## Harmonogram realizacji projektu

| Data        | Etap                            | Kluczowe zadania                                                                                                                          |
|-------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 8 XI 2024   | Wstępna konfiguracja środowiska | • Konfiguracja środowiska i kontenerów<br>• Utworzenie repozytorium<br>• Utworzenie potoku przygotowania danych<br>• Implementacja podstawowego OCR<br> |
| 22 XI 2024  | Model bazowy      | • Implementacja LayoutLMv3<br>• Konfiguracja bazy danych<br>• Podstawowa klasyfikacja                                                  |
| 6 XII 2024  | Optymalizacja            | • Analiza różnych metod uczenia maszynowego<br>• Implementacja generowania podsumowań<br>• Wybór optymalnego rozwiązania                        |
| 20 XII 2024 | Rozszerzenia    | • Implementacja dodatkowych funkcji                                                                 |
| 17 I 2025   | Interfejs użytkownika           | • Implementacja interfejsu webowego<br>• Dokumentacja końcowa                                                                             |
