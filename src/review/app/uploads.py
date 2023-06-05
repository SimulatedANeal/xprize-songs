import csv
import os
from dateutil import parser
from glob import glob

from flask_login import current_user

from app import db
from app.models import Prediction, Sample, Example


def add_examples(directory):
    for species in os.listdir(directory):
        full_path = os.path.join(directory, species)
        if os.path.isdir(full_path) and species != 'ambient':
            for audio_file in glob(os.path.join(full_path, '*.wav')):
                Example.maybe_create(
                    filepath=audio_file,
                    label=species,
                    user=current_user)


def add_predictions(filepath, model_name, delimiter=','):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            audio_clip = Sample.get_or_create(
                filepath=row['source'],
                start_time_s=row['snippet_t_start_s'],
                end_time_s=row['snippet_t_end_s'],
                user=current_user)
            prediction = Prediction(
                ai_label=row['top_species'],
                ai_call_probability=row['call_probability'],
                ai_species_probability=row['top_species_probability'],
                ai_detection_method=model_name,
                prediction_timestamp=parser.parse(row['prediction_timestamp']),
                audio_id=audio_clip.id)
            db.session.add(prediction)
        db.session.commit()
