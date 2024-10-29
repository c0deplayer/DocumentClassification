# Document Classification

[![pdm-managed](https://img.shields.io/endpoint?url=https%3A%2F%2Fcdn.jsdelivr.net%2Fgh%2Fpdm-project%2F.github%2Fbadge.json)](https://pdm-project.org)

## Project Objective

The project focuses on the implementation of an advanced document classification system utilizing the potential of cloud
computing and machine learning techniques. The system is designed to automate the processes of analyzing and
categorizing various types of documents.

The goal is to create a scalable and efficient document classification and analysis system within a cloud environment,
leveraging advanced machine learning techniques.

---

## Required Libraries

> [!IMPORTANT]
> The minimum supported Python version is 3.12.

All required libraries are listed in the `pyproject.toml` and `requirements.txt` files. To install all dependencies, run
the
following command:

```bash
pip install -r requirements.txt
```

---

## System Requirements

### Core Functional Requirements

#### Document Handling

- Support for various document formats
- Processing of documents in English
- Document classification with minimal latency

#### Security

- Encryption of client-server communication
- Secure document storage with end-to-end encryption

#### Integration and API

- REST API for system communication
- Endpoints for:
    - Uploading documents
    - Retrieving classification results
    - Managing documents
    - Searching and filtering

- API Documentation

#### Data Storage

- Database for storing:
    - Documents in encrypted form
    - Classification results
    - Annotations

### Additional Functional Requirements

#### Multilingual Support

- Handling of documents in multiple languages (minimum 3)
- Automatic language detection

#### Analysis and Processing

- Generation of automatic document summaries
- Extraction of keywords and phrases

#### User Interface

- Intuitive web interface for:
    - Browsing documents by category
    - Managing classification
    - Visualizing statistics
    - Exporting data and reports

---

## System Constraints [WIP]

### File Limitations

#### Text Documents

- Formats: PDF
- Maximum size: 20MB
- Page limit: 5

#### Images and Scans

- Image formats: JPEG, PNG
- Scan formats: TIFF, BMP
    - Maximum resolution: 600 DPI
    - Minimum resolution: 150 DPI
- Accepted color modes: RGB, CMYK, grayscale

### Language Limitations

#### Full Support

- Languages: English
- Functionalities:
    - OCR
    - Document classification
    - Description generation

#### Basic Support

(N/A)

### Processing Constraints

#### Description Generation

- Description length:
    - Minimum: 100 words
    - Maximum: 300 words
- Processing time:
    - Description generation: max 10 seconds
    - Document classification: max 10 seconds
    - OCR: max 10 seconds per page

---

## Project Timeline

| Date        | Stage                     | Key Tasks                                                                                                            |
|-------------|---------------------------|----------------------------------------------------------------------------------------------------------------------|
| 8 Nov 2024  | Initial Environment Setup | • Cloud and local environment setup<br>• Repository creation<br>• Data preparation pipeline setup<br>• OCR           |
| 22 Nov 2024 | Baseline Model & Database | • Implementation of basic machine learning method<br>• Database configuration                                        |
| 6 Dec 2024  | Model Optimization        | • Analysis of various machine learning methods<br>• Performance and accuracy testing<br>• Optimal solution selection |
| 20 Dec 2024 | Feature Expansion         | • Implementation of multilingual support<br>• Description generation system                                          |
| 17 Jan 2025 | User Interface            | • Web interface implementation<br>• Final documentation                                                              |
