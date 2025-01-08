import base64
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median
from zoneinfo import ZoneInfo

import requests
from app import app, db_session
from dotenv import load_dotenv
from flask import (
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from forms import DocumentForm
from sqlalchemy import select
from werkzeug.utils import secure_filename

from database.models import Document
from encryption.aes import AESCipher

load_dotenv()
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("Missing ENCRYPTION_KEY")

KEY = base64.b64decode(ENCRYPTION_KEY)
cipher = AESCipher(KEY)


@app.route("/download/<int:document_id>")
async def download_document(document_id: int) -> object:
    """Download an encrypted document by decrypting and sending it to the client.

    This asynchronous function retrieves an encrypted document from the database,
    creates a temporary decrypted copy, and sends it to the client as an attachment.
    The temporary file is automatically cleaned up after the response is sent.

    Args:
        document_id (int): The ID of the document to download

    Returns:
        object: Either a file response containing the decrypted document,
                or a redirect response in case of errors

    Raises:
        No exceptions are raised directly, all are caught and handled internally
        resulting in redirect responses with appropriate error messages

    Notes:
        - Uses temp files for decryption
        - Ensures cleanup of temp files even if errors occur
        - Logs errors to the application logger
        - Uses flash messages to communicate errors to the user

    """
    try:
        # Get document
        query = select(Document).filter(Document.id == document_id)
        result = db_session.execute(query)
        document = result.scalar_one_or_none()

        if document is None:
            flash("Document not found!", "error")
            return redirect(url_for("index"))

        # Create temp file for decrypted content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        # Decrypt file
        encrypted_file = Path(document.file_path)
        if not encrypted_file.exists():
            flash("File not found on disk!", "error")
            return redirect(url_for("index"))

        try:
            await cipher.decrypt_file(encrypted_file, temp_path)

            # Send file and ensure temp file is cleaned up
            response = send_file(
                temp_path,
                as_attachment=True,
                download_name=document.file_name,
            )

            # Delete temp file after sending
            @response.call_on_close
            def cleanup() -> None:
                if temp_path.exists():
                    temp_path.unlink()

        except Exception:
            current_app.logger.exception("Error decrypting file")
            if temp_path.exists():
                temp_path.unlink()
            flash("Error decrypting file", "error")
            return redirect(url_for("index"))

        else:
            return response

    except Exception:
        current_app.logger.exception("Error downloading document")
        flash("Error downloading document", "error")
        return redirect(url_for("index"))


@app.get("/")
def index() -> str:
    """Handle the main page request of the document classification application.

    This function processes the main page request, handling document listing with
    sorting and filtering capabilities. It supports:
    - Sorting by file name, classification, creation date, and summary
    - Filtering by document categories
    - Default sorting by creation date in descending order

    Query Parameters:
        sort (str, optional): Column to sort by ('file_name', 'classification',
                              'created_at', 'summary')
        direction (str, optional): Sort direction ('asc' or 'desc')
        category (list, optional): List of categories to filter documents by

    Returns:
        str: Rendered HTML template with:
            - List of documents
            - Current sort column and direction
            - Default sort settings
            - Available categories
            - Selected categories
            - Error message if something goes wrong

    Raises:
        Exception: Catches any exceptions during execution and renders an error template

    """
    try:
        # Get sort parameters
        sort_column = request.args.get("sort")
        sort_direction = request.args.get("direction")

        # Get selected categories from query params
        selected_categories = request.args.getlist("category")

        # Get all unique categories for the filter UI
        categories_query = select(Document.classification).distinct()
        categories_result = db_session.execute(categories_query)
        all_categories = [
            cat[0] for cat in categories_result if cat[0]
        ]  # Exclude None values

        # Build base query
        query = select(Document)

        # Apply category filter if any categories selected
        if selected_categories:
            query = query.where(
                Document.classification.in_(selected_categories),
            )

        # Apply sorting
        valid_columns = {
            "file_name": Document.file_name,
            "classification": Document.classification,
            "created_at": Document.created_at,
            "summary": Document.summary,
        }

        # Default sorting state
        default_sort = "created_at"
        default_direction = "desc"

        # If no sort specified or in reset state, use default sorting
        if not sort_column or not sort_direction:
            query = query.order_by(Document.created_at.desc())
            sort_column = default_sort
            sort_direction = default_direction
        else:
            # Use requested sort if valid
            sort_col = valid_columns.get(sort_column, Document.created_at)
            if sort_direction == "asc":
                query = query.order_by(sort_col.asc())
            else:
                query = query.order_by(sort_col.desc())

        result = db_session.execute(query)
        documents = result.scalars().all()

        return render_template(
            "index.html.j2",
            documents=documents,
            current_sort=sort_column,
            current_direction=sort_direction,
            default_sort=default_sort,
            default_direction=default_direction,
            all_categories=sorted(all_categories),
            selected_categories=selected_categories,
        )
    except Exception:
        app.logger.exception("Error fetching documents")
        flash("Error loading documents", "error")
        return render_template("index.html.j2", documents=[])


@app.route("/document/<int:document_id>", methods=["GET", "POST"])
def edit_document(document_id: int) -> object:
    """Edit a document in the database.

    This route handler allows updating the classification and summary of an existing document.
    It loads the document by ID, displays an edit form populated with current values,
    and saves changes on form submission.

    Args:
        document_id (int): The ID of the document to edit

    Returns:
        object: Either:
            - A redirect to the index page (after successful update or on error)
            - The rendered edit document template with form

    Raises:
        Exception: If there is an error accessing the database or processing the request.
            The exception is caught and logged, and user is redirected to index with error message.

    Flash Messages:
        - "Document not found!" (error) if document_id doesn't exist
        - "Document updated successfully!" (success) on successful update
        - "Error updating document" (error) if an exception occurs

    """
    try:
        # Get document
        query = select(Document).filter(Document.id == document_id)
        result = db_session.execute(query)
        document = result.scalar_one_or_none()

        if document is None:
            flash("Document not found!", "error")
            return redirect(url_for("index"))

        form = DocumentForm(obj=document)

        if form.validate_on_submit():
            document.classification = form.classification.data
            document.summary = form.summary.data
            db_session.commit()
            flash("Document updated successfully!", "success")
            return redirect(url_for("index"))

        return render_template(
            "edit_document.html.j2",
            form=form,
            document=document,
        )
    except Exception:
        app.logger.exception("Error editing document")
        flash("Error updating document", "error")
        return redirect(url_for("index"))


@app.get("/statistics")
def statistics() -> str:
    """Generate and display document statistics.

    This function calculates various statistics about the documents in the database, including:
    - Total number of documents
    - Number of unclassified documents
    - Number of documents without summaries
    - Classification distribution
    - Summary statistics (count, average length, median length, min/max lengths)
    - Classification distribution by month
    - Daily document counts for the last 30 days

    Returns:
        str: Rendered HTML template with statistics or redirects to index on error

    Raises:
        Exception: If there is an error generating the statistics, logs error and redirects to index

    """
    try:
        doc_query = select(Document)
        result = db_session.execute(doc_query)
        documents = result.scalars().all()

        # Basic document counts
        total_docs = len(documents)
        unclassified = sum(1 for doc in documents if not doc.classification)
        no_summary = sum(1 for doc in documents if not doc.summary)

        # Classification distribution
        classification_counts = Counter(
            doc.classification for doc in documents if doc.classification
        )

        # Summary statistics
        word_counts = [
            len(doc.summary.split()) for doc in documents if doc.summary
        ]

        summary_stats = {
            "count": len(word_counts),
            "avg_length": round(mean(word_counts)) if word_counts else 0,
            "median_length": round(median(word_counts)) if word_counts else 0,
            "min_length": min(word_counts, default=0),
            "max_length": max(word_counts, default=0),
        }

        # Classification distribution over time
        class_by_month = defaultdict(Counter)
        for doc in documents:
            if doc.classification and doc.created_at:
                month_key = doc.created_at.strftime("%Y-%m")
                class_by_month[month_key][doc.classification] += 1

        # Sort by month and convert to regular dict for template
        class_by_month = dict(sorted(class_by_month.items()))

        # Daily counts (last 30 days)
        now = datetime.now(ZoneInfo("UTC"))
        thirty_days_ago = now - timedelta(days=30)

        daily_counts = Counter(
            doc.created_at.astimezone(ZoneInfo("UTC")).date()
            for doc in documents
            if doc.created_at is not None
        )

        # Fill missing dates
        all_dates = [
            (thirty_days_ago + timedelta(days=x)).date() for x in range(31)
        ]
        for date in all_dates:
            if date not in daily_counts:
                daily_counts[date] = 0

        daily_data = sorted(daily_counts.items())

        return render_template(
            "statistics.html.j2",
            total_docs=total_docs,
            classification_data=dict(classification_counts),
            daily_data=daily_data,
            unclassified=unclassified,
            no_summary=no_summary,
            summary_stats=summary_stats,
            class_by_month=class_by_month,
        )
    except Exception:
        app.logger.error("Error generating statistics")
        flash("Error loading statistics", "error")
        return redirect(url_for("index"))


@app.post("/upload")
def upload_file():
    """Upload and process a file through OCR service.

    This endpoint handles file upload requests and forwards them to an OCR service for processing.
    The file must be present in the request and have a non-empty filename.

    Returns:
        tuple: JSON response and HTTP status code
            Success (200):
                    "message": "File uploaded successfully"
                }
            Error (400):
                    "message": "No file selected"
                }
            Error (500):
                    "message": "Error uploading file"
                }
            Error (other):
                    "message": "Error processing file"
                }

    Raises:
        Exception: If there's an error during file upload or processing

    """
    try:
        if "file" not in request.files:
            return jsonify(
                {"status": "error", "message": "No file selected"},
            ), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify(
                {"status": "error", "message": "No file selected"},
            ), 400

        if file:
            filename = secure_filename(file.filename)
            mime_type = file.content_type

            # Send file to OCR service
            files = {"file": (filename, file.stream, mime_type)}
            response = requests.post("http://ocr:8080/ocr", files=files)

            if response.status_code == 200:
                return jsonify(
                    {
                        "status": "success",
                        "message": "File uploaded successfully",
                    },
                )
            return jsonify(
                {
                    "status": "error",
                    "message": "Error processing file",
                },
            ), response.status_code

    except Exception:
        app.logger.exception("Upload error")
        return jsonify(
            {
                "status": "error",
                "message": "Error uploading file",
            },
        ), 500
