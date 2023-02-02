import argparse
import os
import pathlib

import tensorflow_io as tfio
import numpy as np
import tensorflow as tf

from log_util import make_log_confusion_matrix_fn
from model import build_basic_model


MS = 1000
EPOCHS = 10
BATCH_SIZE = 48
SAMPLING_RATE = 96000
SEQ_LENGTH = 256 / MS * SAMPLING_RATE  # 256ms
NFFT = 512
IMG_SIZE = 64


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE)


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    window = int(SAMPLING_RATE * 2 / MS)  # ms window for spectrogram
    stride = int(window / 2)
    spectrogram = tf.signal.stft(
        waveform, fft_length=NFFT, frame_length=window, frame_step=stride)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def main():
    args = get_args()
    log_dir = args.directory_tb_logs
    save_dir = args.directory_model_save
    data_dir = args.directory_dataset
    train_dir = pathlib.Path(os.path.join(data_dir, 'train'))
    test_dir = pathlib.Path(os.path.join(data_dir, 'test'))

    print("Loading training data")
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=train_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        sampling_rate=SAMPLING_RATE,
        seed=seed,
        # output_sequence_length=SEQ_LENGTH,
        subset='both')
    print("Loading test data")
    test_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=test_dir,
        batch_size=BATCH_SIZE,
        sampling_rate=SAMPLING_RATE,
        seed=seed)
    label_names = np.array(train_ds.class_names)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = test_ds.map(squeeze, tf.data.AUTOTUNE)

    print("label names:", label_names)
    num_labels = len(label_names)

    train_ds = make_spec_ds(train_ds)
    val_ds = make_spec_ds(val_ds)
    test_ds = make_spec_ds(test_ds)

    norm_layer = tf.keras.layers.Normalization()
    print("Fitting Normalization layer to training set")
    norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
            norm_layer,
            tf.keras.layers.RandomTranslation(
                height_factor=0.2,  # spectograms are transposed, so this is time dimension
                width_factor=0.,
                fill_mode='wrap')
        ],
        name='preprocessing')
    model = build_basic_model(
        train_ds=train_ds,
        num_labels=num_labels,
        preprocessing=preprocessing)

    train_ds = train_ds.cache().shuffle(BATCH_SIZE * 4).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=make_log_confusion_matrix_fn(
                    model=model,
                    file_writer_cm=tf.summary.create_file_writer(
                        os.path.join(log_dir, 'image', 'cm')),
                    file_writer_wrong=tf.summary.create_file_writer(
                        os.path.join(log_dir, 'image', 'missed')),
                    test_ds=test_ds,
                    label_names=label_names,
                    include_ambient_noise=True,
                    preproc_layer=preprocessing))])

    model.evaluate(test_ds, return_dict=True)

    if save_dir:
        model.save(save_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-dataset", "-d", type=str, required=True)
    parser.add_argument("--directory-model-save", "-m", type=str, default=None)
    parser.add_argument("--directory-tb-logs", '-l', type=str, default='tb_logs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
