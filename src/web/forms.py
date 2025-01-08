from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired


class DocumentForm(FlaskForm):
    """Form for editing documents."""

    file_name = StringField("File Name", validators=[DataRequired()])
    classification = StringField("Classification")
    summary = TextAreaField("Summary")
