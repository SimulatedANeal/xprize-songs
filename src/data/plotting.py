import numpy as np
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from src.data.label import Label


def waveform_and_spectrogram(signal_data, sample_rate, nfft=1024, size=(6, 4)):
    fig, axes = plt.subplots(2, figsize=size)
    axes[0].plot(signal_data)
    axes[0].set_title('Waveform')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlabel('Samples')
    axes[1].specgram(signal_data, Fs=sample_rate, NFFT=nfft)
    axes[1].set_title('Spectrogram')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlabel('Time (s)')
    fig.tight_layout()
    return fig


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
