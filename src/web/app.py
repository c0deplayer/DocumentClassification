from __future__ import annotations

from typing import TYPE_CHECKING

from flask import Flask
from flask_bootstrap import Bootstrap5
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug import run_simple

from configs.database_config import DatabaseConfig

if TYPE_CHECKING:
    from datetime import datetime

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secret-key-here"  # Change this!

config = DatabaseConfig.from_env()

# Create sync engine
engine = create_engine(
    f"postgresql://{config.DB_USER}:{config.DB_PASSWORD}"
    f"@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}",
    echo=config.ECHO,
)

# Create session factory
session_factory = sessionmaker(bind=engine)
db_session = scoped_session(session_factory)

bootstrap = Bootstrap5(app)


# Add custom filter for date formatting
def format_date(date: datetime) -> str:
    """Format date to string."""
    return date.strftime("%Y-%m-%d")


# Register the filter
app.jinja_env.filters["format_date"] = format_date

from routes import *  # noqa: E402, F403


@app.teardown_appcontext
def shutdown_session(_exception: Exception | None = None) -> None:
    """Remove the database session at the end of the request."""
    db_session.remove()


if __name__ == "__main__":
    run_simple("localhost", 5050, app, use_debugger=True)
