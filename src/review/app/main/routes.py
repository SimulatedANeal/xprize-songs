import csv
import io
import os
import random
from datetime import datetime

from flask import render_template, flash, redirect, url_for, request, \
    current_app, Response, send_file
from flask_login import current_user, login_required
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


from werkzeug.utils import secure_filename

from app import db
from app.main import bp, forms
from app.models import User, Prediction, Sample, Example
from app.uploads import add_predictions, add_examples


SAMPLE_RATE = 128000


def export_predictions(delimiter=','):
    proxy = io.StringIO()
    writer = csv.writer(proxy, delimiter=delimiter)
    columns = [
        'source_file', 'snippet_start_s', 'snippet_end_s',
        'accepted', 'ai_detection_method', 'ai_call_probability',
        'ai_label', 'ai_species_probability', 'inference_timestamp',
        'human_label', 'reviewed_by', 'reviewed_at', 'review_confidence_score']
    writer.writerow(columns)
    qq = Prediction.query.filter(Prediction.reviewed_by.isnot(None)).all()
    for p in qq:
        writer.writerow([
            p.sample.filepath, p.sample.start_time_s, p.sample.stop_time_s,
            p.accepted, p.ai_detection_method, p.ai_call_probability,
            p.ai_label,  p.ai_species_probability, p.prediction_timestamp,
            p.human_label, p.reviewer.username, p.reviewed_on, p.review_confidence_score])
    return proxy


def waveform_and_spectrogram(signal_data, sample_rate=SAMPLE_RATE, nfft=1024, size=(6, 6)):
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
    confirmed = Prediction.query.filter_by(accepted=True)
    confirmed_species = confirmed.with_entities(Prediction.human_label).distinct().count()
    num_confirmed = confirmed.count()
    num_unreviewed = Prediction.query.filter_by(reviewed_by=None).count()

    reviewed = Prediction.query \
        .filter(Prediction.reviewed_by.isnot(None)) \
        .order_by(Prediction.reviewed_on.desc())
    reviewed = reviewed.paginate(
        page=request.args.get('page', 1, type=int),
        per_page=current_app.config['ROWS_PER_PAGE'],
        error_out=False)
    return render_template(
        'index.html',
        title='Home',
        unreviewed=num_unreviewed,
        nconfirmed=num_confirmed,
        nspecies=confirmed_species,
        recently_reviewed=reviewed)


@bp.route('/plot/sample/spectrogram/<sample_id>.png')
@login_required
def plot_sample_spectrogram(sample_id):
    w = request.args.get('width', 6, type=int)
    h = request.args.get('height', 6, type=int)
    sample = Sample.query.get(sample_id)
    data = sample.load_audio(sample_rate=SAMPLE_RATE)
    fig = waveform_and_spectrogram(signal_data=data, size=(w, h))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@bp.route('/plot/example/spectrogram/<example_id>.png')
@login_required
def plot_example_spectrogram(example_id):
    w = request.args.get('width', 4, type=int)
    h = request.args.get('height', 4, type=int)
    ex = Example.query.get(example_id)
    data = ex.load_audio(sample_rate=SAMPLE_RATE)
    fig = waveform_and_spectrogram(signal_data=data, size=(w, h))
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
        # TODO: Pick species with least reviews first
        pred = Prediction.get_random_unreviewed()
        if pred is None:
            flash("No un-reviewed predictions.")
            return redirect(url_for('main.index'))
        return redirect(url_for('main.review', prediction_id=pred.id))
    else:
        pred = Prediction.query.get(prediction_id)

    example_species = request.args.get('example_spec') or (
        pred.human_label if pred.human_label else pred.ai_label)
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
        # TODO: Invalid examples?
    elif 'submit_review' in request.form and form.validate_on_submit():
        pred.accepted = form.accept.data
        pred.reviewed_by = current_user.id
        pred.reviewed_on = datetime.utcnow()
        hlabel = form.new_label.data
        pred.human_label = hlabel if hlabel != forms.NULL_SPECIES else pred.ai_label
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
        add_predictions(filepath)
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


@bp.route('/export', methods=['GET', 'POST'])
@login_required
def export_confirmed_ids():
    mem = io.BytesIO()
    proxy = export_predictions()
    mem.write(proxy.getvalue().encode('utf-8'))
    mem.seek(0)
    proxy.close()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return send_file(
        mem,
        download_name=f'waponi_confirmed_cricket_ids_export_{timestamp}.csv',
        as_attachment=True)

# TODO: Review reviewed predictions
