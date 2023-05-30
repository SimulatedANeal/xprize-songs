import io
import os
import random
from datetime import datetime

from flask import render_template, flash, redirect, url_for, request, current_app, Response
from flask_login import current_user, login_required
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from werkzeug.utils import secure_filename

from app import db
from app.main import bp, forms
from app.models import User, Prediction, Sample, Example
from app.uploads import add_predictions, add_examples


SAMPLE_RATE = 256000


def waveform_and_spectrogram(signal_data, sample_rate=SAMPLE_RATE, nfft=1024, size=(6, 4)):
    fig = Figure(figsize=size)
    ax0 = fig.add_subplot(2, 1, 1)
    ax0.plot(signal_data)
    ax0.set_title('Waveform')
    ax0.set_ylabel('Amplitude')
    ax0.set_xlabel('Samples')
    ax1 = fig.add_subplot(2, 1, 2)
    ax1.specgram(signal_data, Fs=sample_rate, NFFT=nfft)
    ax1.set_title('Spectrogram')
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Time (s)')
    fig.tight_layout()
    return fig


@bp.before_app_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.utcnow()
        db.session.commit()


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    return render_template('index.html', title='Home')


@bp.route('/plot/sample/spectrogram/<sample_id>.png')
@login_required
def plot_sample_spectrogram(sample_id):
    sample = Sample.query.get(sample_id)
    data = sample.load_audio(sample_rate=SAMPLE_RATE)
    fig = waveform_and_spectrogram(signal_data=data, size=(8,8))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@bp.route('/plot/example/spectrogram/<example_id>.png')
@login_required
def plot_example_spectrogram(example_id):
    ex = Example.query.get(example_id)
    data = ex.load_audio(sample_rate=SAMPLE_RATE)
    fig = waveform_and_spectrogram(signal_data=data, size=(6,6))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def get_examples(species, n=3):
    examples = Example.query.filter_by(label=species).all()
    examples = random.sample(examples, k=min(n, len(examples)))
    for example in examples:
        example.load_audio(sample_rate=SAMPLE_RATE)
    return examples


@bp.route('/review/', defaults={'prediction_id': None}, methods=['GET', 'POST'])
@bp.route('/review/<prediction_id>', methods=['GET', 'POST'])
@login_required
def review(prediction_id=None):
    form = forms.build_review_form()
    if prediction_id is None:
        pred = Prediction.get_random_unreviewed()
        return redirect(url_for('main.review', prediction_id=pred.id))
    else:
        pred = Prediction.query.get(prediction_id)

    example_species = request.args.get('example_spec') or pred.label
    xform = forms.build_example_form(default_species=example_species)

    sample = pred.sample
    sample.load_audio(SAMPLE_RATE)

    if request.method == 'GET':
        form.accept = pred.accepted
    elif 'submit_examples' in request.form and xform.validate_on_submit():
        return redirect(url_for(
            'main.review',
            prediction_id=pred.id,
            example_spec=xform.species.data))
    elif 'submit_review' in request.form and form.validate_on_submit():
        pred.accepted = form.accept.data
        pred.reviewed_by = current_user.id
        pred.reviewed_on = datetime.utcnow()
        if form.new_label.data != forms.NULL_SPECIES:
            pred.label = form.new_label.data
            pred.ai_detection_method = None
            pred.probability = None
        pred.review_confidence_score = form.confidence.data
        # TODO: Add to examples?
        db.session.commit()
        return redirect(url_for('main.review'))

    return render_template(
        'review.html',
        title='Review',
        pred=pred,
        form=form,
        example_form=xform,
        examples=get_examples(example_species, n=3))


# def confirmed_ids():
#     ...


@bp.route('/upload/examples', methods=['GET', 'POST'])
@login_required
def upload_examples():
    form = forms.UploadExamplesForm()
    if form.validate_on_submit():
        local_filepath = form.filepath.data
        if not os.path.exists(local_filepath):
            flash("Invalid data directory")
            return redirect(url_for('main.upload_examples'))
        flash("Uploading... will redirect to home page when complete.")
        add_examples(directory=local_filepath)
        return redirect(url_for('main.index'))
    return render_template('upload.html', form=form)


@bp.route('/upload/predictions', methods=['GET', 'POST'])
@login_required
def upload_predictions():
    form = forms.UploadPredictionsForm()
    if form.validate_on_submit():
        filename = secure_filename(form.file.data.filename)
        filepath = os.path.join(current_app.config['UPLOADS_FOLDER'], filename)
        if not os.path.exists(current_app.config['UPLOADS_FOLDER']):
            os.makedirs(current_app.config['UPLOAD_FOLDER'])
        form.file.data.save(filepath)
        flash("Uploading... will redirect to home page when complete.")
        add_predictions(filepath, model_name=form.model_name.data)
        return redirect(url_for('main.index'))
    return render_template('upload.html', form=form)


@bp.route('/user/<username>')
@login_required
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    form = forms.EmptyForm()
    return render_template('user.html', user=user, form=form)


@bp.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = forms.EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('main.edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
    return render_template('edit_profile.html', title='Edit Profile', form=form)
