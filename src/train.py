import argparse
import os
import pathlib

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from log_util import make_log_confusion_matrix_fn
from model import build_basic_model, build_multitask_model
from src.data.transform import get_spectrogram_fn
from src.data.datasets import audio_dataset_from_directory
from src.data import plotting


MS = 1000
EPOCHS = 10
BATCH_SIZE = 64

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


def convert_to_multitask(ds, ix_ambient):
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

    spectro_fn = get_spectrogram_fn(sample_rate, nfft, window_ms=2, stride_ms=1)

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
    return train_ds, val_ds, test_ds, label_names


def train_and_evaluate(
        data_dir, log_dir, save_dir,
        sample_rate, nfft, input_size,
        hidden_layers, convolutions,
        multitask):
    ds_train, ds_val, ds_test, labels = build_datasets(data_dir, nfft, sample_rate)
    num_labels = len(labels)

    convlayersname = [f'{f}f{s}' for f, s in convolutions]
    model_name = f"model_{int(sample_rate / 1000)}k" \
                 f"_{nfft}" \
                 f"_{input_size}x{input_size}" \
                 f"_{'-'.join(convlayersname)}" \
                 f"_{'-'.join(map(str, hidden_layers))}" \
                 f"_{num_labels}"

    if multitask:
        label_ambient_ix = labels.index('ambient')
        ds_train = convert_to_multitask(ds_train, label_ambient_ix)
        ds_val = convert_to_multitask(ds_val, label_ambient_ix)
        ds_test = convert_to_multitask(ds_test, label_ambient_ix)
        model, preprocessing_layer = build_multitask_model(
            train_ds=ds_train,
            num_labels=num_labels,
            input_size=input_size,
            conv_layers=convolutions,
            hidden_dense_size=hidden_layers)
        model_name = f"{model_name}_multitask"
    else:
        model, preprocessing_layer = build_basic_model(
            train_ds=ds_train,
            num_labels=num_labels,
            input_size=input_size,
            conv_layers=convolutions,
            hidden_dense_size=hidden_layers)
    log_dir = os.path.join(log_dir, model_name)

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=make_log_confusion_matrix_fn(
                    model=model,
                    file_writer_cm=tf.summary.create_file_writer(os.path.join(log_dir, 'image', 'cm')),
                    file_writer_wrong=tf.summary.create_file_writer(os.path.join(log_dir, 'image', 'missed')),
                    test_ds=ds_test,
                    label_names=labels,
                    preproc_layer=preprocessing_layer,
                    multitask=multitask))])

    model.evaluate(
        ds_test,
        return_dict=True,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        ])

    if save_dir:
        model_dir = os.path.join(save_dir, model_name)
        model.save(model_dir)


def main():
    args = get_args()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # # Code for quickly viewing an exacmple batch of spectrograms
    # ds_train, ds_val, ds_test, labels = build_datasets(
    #     directory=args.directory_dataset,
    #     nfft=512,
    #     sample_rate=256000,
    # )
    # for example_spectrograms, example_spect_labels in ds_train.take(1):
    #     break
    # rows = 3
    # cols = 3
    # n = rows * cols
    # fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    # for i in range(n):
    #     r = i // cols
    #     c = i % cols
    #     ax = axes[r][c]
    #     plotting.spectrogram(example_spectrograms[i].numpy(), ax)
    #     ax.set_title(labels[example_spect_labels[i].numpy()])
    # plt.show()

    # Grid Search
    for sr in (256000,):
        for nfft in (1024,):
            for img_size in (64, 128):
                for mt in (True, False):
                    for conv in [
                            ((32, 3), (64, 3)),
                            ((32, 3), (32, 3), (32, 3)),
                            ((32, 5), (32, 3), (64, 3))]:
                        for hidden in [
                                (128,),
                                (32, 64),
                                (64, 128),
                                (32, 64, 64)]:
                            train_and_evaluate(
                                data_dir=args.directory_dataset,
                                log_dir=args.directory_tb_logs,
                                save_dir=args.directory_model_save,
                                sample_rate=sr,
                                nfft=nfft,
                                input_size=img_size,
                                hidden_layers=hidden,
                                convolutions=conv,
                                multitask=mt)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-dataset", "-d", type=str, required=True)
    parser.add_argument("--directory-model-save", "-m", type=str, default=None)
    parser.add_argument("--directory-tb-logs", '-l', type=str, default='tb_logs')
    return parser.parse_args()


if __name__ == '__main__':
    main()
