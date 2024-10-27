# Klasyfikacja Dokumentów w Chmurze Obliczeniowej

[![en](https://img.shields.io/badge/lang-en-green.svg)](./README.en.md)
[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)

## Cel projektu

Projekt koncentruje się na implementacji zaawansowanego systemu klasyfikacji dokumentów, wykorzystującego potencjał
chmury obliczeniowej oraz technik uczenia maszynowego. System został zaprojektowany z myślą o automatyzacji procesów
analizy i kategoryzacji różnorodnych dokumentów.

Celem projektu jest stworzenie skalowalnego, wydajnego systemu klasyfikacji i analizy dokumentów w środowisku chmurowym,
wykorzystującego zaawansowane techniki uczenia maszynowego.
---

## Wymagania funkcjonalne

### Podstawowe

#### Obsługa dokumentów

* Wsparcie dla różnorodnych formatów dokumentów
* Przetwarzanie dokumentów w języku angielskim
* Klasyfikacja dokumentów z niewielkim opóźnieniem

#### Bezpieczeństwo

* Szyfrowanie komunikacji klient-serwer
* Bezpieczne przechowywanie dokumentów z szyfrowaniem end-to-end

#### Integracja i API

* REST API do komunikacji z systemem

* Endpoints do:
    - Przesyłania dokumentów
    - Pobierania wyników klasyfikacji
    - Zarządzania dokumentami
    - Wyszukiwania i filtrowania

* Dokumentacja API

#### Przechowywanie danych

* Baza danych do przechowywania:
    - Dokumentów w formie zaszyfrowanej
    - Wyników klasyfikacji
    - Adnotacji

### Dodatkowe

#### Wielojęzyczność

* Obsługa dokumentów w różnych językach (minimum 3 języki)
* Automatyczne wykrywanie języka dokumentu

#### Analiza i przetwarzanie

* Generowanie automatycznych streszczeń dokumentów
* Ekstrakcja kluczowych słów i fraz

#### Interfejs użytkownika

* Intuicyjny interfejs webowy do:

    - Przeglądania dokumentów według kategorii
    - Zarządzania klasyfikacją
    - Wizualizacji statystyk
    - Eksportu danych i raportów

---

## Ograniczenia systemu [WIP]

### Ograniczenia plików

#### Dokumenty tekstowe

* Formaty: PDF, DOCX, TXT
* Maksymalny rozmiar: 20MB
* Limit stron: 5

#### Obrazy i skany

* Formaty zdjęć: JPEG, PNG, WEBP
    - Maksymalny rozmiar: 20MB
    - Maksymalna rozdzielczość: 2480px x 3508px
* Formaty skanów: TIFF, BMP
    - Maksymalna rozdzielczość: 600 DPI
    - Minimalna rozdzielczość: 150 DPI
* Akceptowane tryby kolorów: RGB, CMYK, skala szarości

### Ograniczenia językowe

#### Pełna obsługa

* Języki: angielski
* Funkcjonalności:
    - OCR
    - Klasyfikacja dokumentów
    - Generowanie opisów

#### Podstawowa obsługa

(N/A)

### Ograniczenia przetwarzania

#### Generowanie opisów

* Długość opisu:
    - Minimum: 100 słów
    - Maksimum: 300 słów
* Czas przetwarzania:
    - Generowanie opisu: maksymalnie 10 sekund
    - Klasyfikacja dokumentu: maksymalnie 10 sekund
    - OCR: maksymalnie 15 sekund na stronę

---

## Harmonogram realizacji projektu

| Data        | Etap                            | Kluczowe zadania                                                                                                                          |
|-------------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| 8 XI 2024   | Wstępna konfiguracja środowiska | • Konfiguracja środowiska chmurowego oraz lokalnego<br>• Utworzenie repozytorium<br>• Utworzenie potoku przygotowania danych<br>• OCR<br> |
| 22 XI 2024  | Model bazowy i baza danych      | • Implementacja podstawowej metody uczenia maszynowego<br>• Konfiguracja bazy danych<br>                                                  |
| 6 XII 2024  | Optymalizacja modelu            | • Analiza różnych metod uczenia maszynowego<br>• Testy wydajności i dokładności<br>• Wybór optymalnego rozwiązania                        |
| 20 XII 2024 | Rozszerzenie funkcjonalności    | • Implementacja obsługi wielu języków<br>• System generowania opisów<br>                                                                  |
| 17 I 2025   | Interfejs użytkownika           | • Implementacja interfejsu webowego<br>• Dokumentacja końcowa                                                                             |