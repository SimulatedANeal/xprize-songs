import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from src.data.label import Label


def spectrogram(spectrogram, ax):
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


def waveform_and_spectrogram(waveform, spectrogram_fn, label=None, show_shape=False):
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])
    axes[0].plot(timescale, waveform)
    axes[0].set_title('Waveform')

    gram = spectrogram_fn(waveform)
    if show_shape:
        print(gram.shape)
    spectrogram(gram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    if label:
        plt.suptitle(label.title())
    plt.show()


def example_stats(all_species_dict, title, bar_fn=len, y_label='Count'):
    labels = list(all_species_dict.keys())
    width = 0.35  # the width of the bars
    total_height = [0] * len(labels)
    fig, ax = plt.subplots()
    for l in Label.non_noise():
        xx = [bar_fn([row for row in call_dict[l]])
              for species, call_dict in all_species_dict.items()]
        ax.bar(labels, xx, width, label=l, bottom=total_height)
        total_height = [x + h for (x, h) in zip(xx, total_height)]
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    plt.show()


def confusion_matrix(y_true, y_pred, label_names, show=False):
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=label_names,
                yticklabels=label_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.tight_layout()
    if show:
        plt.show()
    return figure


def image_grid(images, labels):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    square = int(np.ceil(np.sqrt(len(images))))
    figure = plt.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        # Start next subplot.
        plt.subplot(square, square, i + 1, title=labels[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return figure
