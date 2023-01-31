import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow_io as tfio  # Need to import before Tensorflow
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


MS = 1000
BATCH_SIZE = 32
SAMPLING_RATE = 96000
SEQ_LENGTH = 256 / MS * SAMPLING_RATE  # 256ms
NFFT = 512


# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels


def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)


def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  window = int(SAMPLING_RATE * 2 / MS)  # ms window for spectogram
  stride = int(window / 2)
  spectrogram = tf.signal.stft(
      waveform, fft_length=NFFT, frame_length=window, frame_step=stride)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram


def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)


def main():
    args = get_args()
    data_dir = args.directory_dataset
    train_dir = pathlib.Path(os.path.join(data_dir, 'train'))
    test_dir = pathlib.Path(os.path.join(data_dir, 'test'))
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=train_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        sampling_rate=SAMPLING_RATE,

        seed=seed,
        # output_sequence_length=SEQ_LENGTH,
        subset='both')
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
    print(train_ds.element_spec)

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
        break

    input_shape = example_spectrograms.shape[1:]
    print('Input shape:', input_shape)
    num_labels = len(label_names)

    # rows = 3
    # cols = 3
    # n = rows * cols
    # fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    #
    # for i in range(n):
    #     r = i // cols
    #     c = i % cols
    #     ax = axes[r][c]
    #     plot_spectrogram(example_spectrograms[i].numpy(), ax)
    #     ax.set_title(label_names[example_spect_labels[i].numpy()])
    #
    # plt.show()

    train_spectrogram_ds = train_spectrogram_ds.cache().shuffle(BATCH_SIZE * 4).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.cache().prefetch(tf.data.AUTOTUNE)

    # for example_audio, example_labels in train_ds.take(1):
    #     print(example_audio.shape)
    #     print(example_labels.shape)

    # for i in range(64):
    #     label = label_names[example_labels[i]]
    #     waveform = example_audio[i]
    #     spectrogram = get_spectrogram(waveform)
    #
    #     fig, axes = plt.subplots(2, figsize=(12, 8))
    #     timescale = np.arange(waveform.shape[0])
    #     axes[0].plot(timescale, waveform.numpy())
    #     axes[0].set_title('Waveform')
    #     axes[0].set_xlim([0, SEQ_LENGTH])
    #
    #     plot_spectrogram(spectrogram.numpy(), axes[1])
    #     axes[1].set_title('Spectrogram')
    #     plt.suptitle(label.title())
    #     plt.show()

    # rows = 3
    # cols = 3
    # n = rows * cols
    # fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    # for i in range(n):
    #     if i >= n:
    #         break
    #     r = i // cols
    #     c = i % cols
    #     ax = axes[r][c]
    #     ax.plot(example_audio[i].numpy())
    #     ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    #     label = label_names[example_labels[i]]
    #     ax.set_title(label)
    #     ax.set_ylim([-1.1, 1.1])
    #
    # plt.show()

    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    print("Fitting Normalization layer to training set")
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    EPOCHS = 10
    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    model.save('model')

    metrics = history.history
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, 100 * np.array(metrics['accuracy']), 100 * np.array(metrics['val_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')

    model.evaluate(test_spectrogram_ds, return_dict=True)
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s, lab: lab)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory-dataset", "-d", type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    main()
