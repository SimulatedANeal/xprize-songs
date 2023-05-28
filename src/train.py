import argparse
import os
import pathlib
import pickle

import numpy as np
import tensorflow as tf
# from matplotlib import pyplot as plt

# from log_util import make_log_confusion_matrix_fn
from model import build_model, save_model
# from src.data import plotting
from src.data.transform import get_spectrogram_fn
from src.data.datasets import audio_dataset_from_directory


# Training parameters
EPOCHS = 15
BATCH_SIZE = 64
# Spectrogram parameters
NFFT = 1024
FFT_WINDOW_MS = 2
FFT_WINDOW_STRIDE_MS = 1
# Model definition
IMG_SIZE = 96
CONVOLUTIONS = ((16, 3), (32, 3), (64, 3))
DENSE = (50,)
SPECIES_LAYERS = (40,)

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def convert_to_multitask(ds, ix_ambient):
    # TODO: Might buggy if 'ambient' is not last label
    return ds.map(
        map_func=lambda spec, label: (spec, {
            "species": label[:, :ix_ambient],
            "call": tf.where(tf.equal(label[:, ix_ambient], 1), 0, 1)}),
        num_parallel_calls=tf.data.AUTOTUNE)


def build_datasets(directory, nfft, sample_rate=None):
    print(f"Using sample rate: {sample_rate}")
    directory_train = pathlib.Path(os.path.join(directory, 'train'))
    directory_test = pathlib.Path(os.path.join(directory, 'test'))
    print("Loading training data")
    train_ds, val_ds = audio_dataset_from_directory(
        directory=directory_train,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        label_mode='categorical',
        # sampling_rate=sample_rate,
        seed=seed,
        subset='both')
    print("Loading test data")
    test_ds = audio_dataset_from_directory(
        directory=directory_test,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        # sampling_rate=sample_rate,
        seed=seed)
    label_names = list(train_ds.class_names)
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
    test_ds = test_ds.map(squeeze, tf.data.AUTOTUNE)
    print("label names:", label_names)

    spectro_fn = get_spectrogram_fn(
        sample_rate, nfft,
        window_ms=FFT_WINDOW_MS,
        stride_ms=FFT_WINDOW_STRIDE_MS)

    def make_spec_ds(ds):
        return ds.map(
            map_func=lambda audio, label: (spectro_fn(audio), label),
            num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = make_spec_ds(train_ds)
    val_ds = make_spec_ds(val_ds)
    test_ds = make_spec_ds(test_ds)

    # Cache and shuffle (train)
    train_ds = train_ds.cache().shuffle(BATCH_SIZE * 8).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    label_ambient_ix = label_names.index('ambient')
    train_ds = convert_to_multitask(train_ds, label_ambient_ix)
    val_ds = convert_to_multitask(val_ds, label_ambient_ix)
    test_ds = convert_to_multitask(test_ds, label_ambient_ix)

    return train_ds, val_ds, test_ds, label_names[:-1]


def train_and_evaluate(
        data_dir, log_dir, save_dir,
        input_size, convolutions, hidden_layers, species_layers):

    window, sample_rate = data_dir.split('/')[-1].split('_')
    window_length_ms = int(window[:-2])
    sample_rate = int(sample_rate[:-3]) * 1000
    ds_train, ds_val, ds_test, labels = build_datasets(data_dir, NFFT, sample_rate)

    # Code for quickly viewing an example batch of spectrograms
    for example_spectrograms, example_spect_labels in ds_train.take(1):
        print(f"Batched input shape: {example_spectrograms.shape}")

    model_config = dict(
        sample_rate=sample_rate,
        window_len_ms=window_length_ms,
        nfft=NFFT,
        labels=labels)
    convlayersname = [f'{f}f{s}' for f, s in convolutions]
    denselayers = list(hidden_layers) + [f"{s}s" for s in species_layers]
    model_name = f"model_{int(sample_rate / 1000)}khz" \
                 f"_{NFFT}" \
                 f"_{input_size}x{input_size}" \
                 f"_{'-'.join(convlayersname)}" \
                 f"_{'-'.join(map(str, denselayers))}" \
                 f"_{len(labels)}"

    model, preprocessing_layer = build_model(
        train_ds=ds_train,
        labels=labels,
        input_size=input_size,
        sample_rate=sample_rate,
        nfft=NFFT,
        fft_window_ms=FFT_WINDOW_MS,
        fft_window_stride_ms=FFT_WINDOW_STRIDE_MS,
        conv_layers=convolutions,
        hidden_dense_size=hidden_layers,
        species_hidden_layers=species_layers)
    log_dir = os.path.join(log_dir, model_name)

    print("Training model...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True)
        ])

    model.summary()

    print("Evaluating model on test set...")
    model.evaluate(
        ds_test,
        return_dict=True,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ])

    print("Example prediction step...")
    for spectrograms, spect_labels in ds_test.take(5):
        print("Start batch")
        embeddings = model.embed(spectrograms)
        print("Embeddings: ", embeddings)
        call_pred = model.get_call_probability(embeddings)
        print("Call probability: ", call_pred)
        call_indices = tf.where(call_pred >= 0.5)[:, 0]
        call_indices = tf.expand_dims(call_indices, axis=1)
        print("Call indices: ", call_indices)
        call_embeddings = tf.gather_nd(embeddings, call_indices)
        species_pred = model.predict_species(call_embeddings)
        print("Species probability: ", species_pred)
        print("End batch\n")

    if save_dir:
        model_dir = os.path.join(save_dir, model_name)
        save_model(model, model_config, model_dir)
        with open(os.path.join(model_dir, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)


def main():
    args = get_args()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_and_evaluate(
        data_dir=args.directory_dataset,
        log_dir=args.directory_tb_logs,
        save_dir=args.directory_model_save,
        input_size=IMG_SIZE,
        hidden_layers=DENSE,
        convolutions=CONVOLUTIONS,
        species_layers=SPECIES_LAYERS)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-dataset", "-d", type=str, required=True)
    parser.add_argument("--directory-model-save", "-m", type=str, default=None)
    parser.add_argument("--directory-tb-logs", '-l', type=str, default='tb_logs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
