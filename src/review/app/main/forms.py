from flask import current_app
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, BooleanField, IntegerField, SelectField
from wtforms.validators import ValidationError, DataRequired, Length, NumberRange

from app.models import User, Example


NULL_SPECIES = '---'


class UploadExamplesForm(FlaskForm):
    filepath = StringField("Data Directory", validators=[DataRequired()])
    submit = SubmitField("Upload")


class UploadPredictionsForm(FlaskForm):
    model_name = StringField("Model Name", validators=[DataRequired(), Length(max=60)])
    file = FileField("Predictions File", validators=[DataRequired()])
    submit = SubmitField('Upload')


def get_species_choices():
    with current_app.app_context():
        slist = Example.query.with_entities(Example.label).distinct()
        return sorted([s[0] for s in slist])


def build_review_form():

    class ReviewForm(FlaskForm):
        accept = BooleanField(label="Accept")
        new_label = SelectField(
            label="Different species from predicted?",
            default=NULL_SPECIES,
            choices=[NULL_SPECIES, 'unknown'] + get_species_choices())
        confidence = IntegerField(
            label="Confidence",
            default=100,
            description=(
                "Enter your confidence in your response as an integer "
                "from 0 to 100, with 0 being not confident at all and "
                "100 being abosolutely certain."),
            validators=[NumberRange(min=0, max=100, message="Value must be an integer.")])
        submit_review = SubmitField("Confirm")

    return ReviewForm()


def build_example_form(default_species):

    class ExampleForm(FlaskForm):
        species = SelectField(
            label="Show examples for: ",
            default=default_species,
            choices=get_species_choices())
        submit_examples = SubmitField("Show examples")

    return ExampleForm()


class EditProfileForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    submit = SubmitField('Submit')

    def __init__(self, original_username, *args, **kwargs):
        super(EditProfileForm, self).__init__(*args, **kwargs)
        self.original_username = original_username

    def validate_username(self, username):
        if username.data != self.original_username:
            user = User.query.filter_by(username=self.username.data).first()
            if user is not None:
                raise ValidationError('Please use a different username.')


class EmptyForm(FlaskForm):
    submit = SubmitField('Submit')


