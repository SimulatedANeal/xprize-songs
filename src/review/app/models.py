import os
from datetime import datetime
from hashlib import md5
from time import time
from collections import Counter

import jwt
import tensorflow as tf
from flask import current_app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func

from app import db, login
from app.util import AudioSample, get_wav_file


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    reviewed = db.relationship('Prediction', backref='reviewer', lazy=True)

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return 'https://www.gravatar.com/avatar/{}?d=identicon&s={}'.format(
            digest, size)

    def get_reset_password_token(self, expires_in=600):
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            current_app.config['SECRET_KEY'], algorithm='HS256')

    @staticmethod
    def verify_reset_password_token(token):
        try:
            id = jwt.decode(
                token, current_app.config['SECRET_KEY'],
                algorithms=['HS256'])['reset_password']
        except:
            return
        return User.query.get(id)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ai_label = db.Column(db.String(100), nullable=False)
    ai_call_probability = db.Column(db.Float)
    ai_species_probability = db.Column(db.Float)
    ai_detection_method = db.Column(db.String(60))
    prediction_timestamp = db.Column (db.DateTime)
    audio_id = db.Column(db.Integer, db.ForeignKey('sample.id'), nullable=False)
    human_label = db.Column(db.String(100), default=None)
    reviewed_by = db.Column(db.Integer, db.ForeignKey('user.id'), default=None)
    reviewed_on = db.Column(db.DateTime, default=None)
    review_confidence_score = db.Column(db.Integer, default=None)
    accepted = db.Column(db.Boolean, default=False)

    @classmethod
    def get_random_unreviewed(cls):
        species = cls.query \
            .filter_by(reviewed_by=None) \
            .with_entities(cls.ai_label).distinct() \
            .order_by(func.random()).first()[0]
        unreviewed = cls.query \
            .filter_by(reviewed_by=None, ai_label=species) \
            .order_by(cls.ai_call_probability.desc()).first()
        return unreviewed


class Sample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.Text, nullable=False)
    start_time_s = db.Column(db.Float, nullable=False)
    stop_time_s = db.Column(db.Float, nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    predictions = db.relationship('Prediction', backref='sample', lazy=True)

    @classmethod
    def get_or_create(cls, filepath, start_time_s, end_time_s, user):
        audio_sample = cls.query.filter_by(
            filepath=filepath,
            start_time_s=start_time_s,
            stop_time_s=end_time_s
        ).all()
        if len(audio_sample):
            audio_sample = audio_sample[0]
        else:
            audio_sample = cls(
                filepath=filepath,
                start_time_s=start_time_s,
                stop_time_s=end_time_s,
                added_by=user.id)
            db.session.add(audio_sample)
            db.session.commit()
        return audio_sample

    def load_audio(self, sample_rate):
        sample = AudioSample(
            filepath=self.filepath,
            time_start_s=self.start_time_s,
            time_end_s=self.stop_time_s)
        data = sample.get_data(sample_rate)
        encoded = tf.audio.encode_wav(data[..., tf.newaxis], sample_rate=sample_rate)
        path = os.path.join(current_app.static_folder, 'audio', f'sample_{self.id}.wav')
        tf.io.write_file(path, encoded)
        return data.numpy()


class Example(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filepath = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(100), nullable=False)
    date_added = db.Column(db.DateTime, default=datetime.utcnow)
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    rejected = db.Column(db.Boolean, default=False)
    rejected_by = db.Column(db.Integer, db.ForeignKey('user.id'))

    @classmethod
    def maybe_create(cls, filepath, label, user):
        species_example = cls.query.filter_by(filepath=filepath, label=label).all()
        if len(species_example):
            species_example = species_example[0]
        else:
            species_example = cls(
                filepath=filepath,
                label=label,
                added_by=user.id)
            db.session.add(species_example)
            db.session.commit()
        return species_example

    def load_audio(self, sample_rate):
        data, sample_rate, length_s = get_wav_file(self.filepath, resample_rate=sample_rate)
        encoded = tf.audio.encode_wav(data, sample_rate=sample_rate)
        path = os.path.join(current_app.static_folder, 'audio', f'example_{self.id}.wav')
        tf.io.write_file(path, encoded)
        return tf.squeeze(data).numpy()

