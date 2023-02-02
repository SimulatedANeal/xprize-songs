import io

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from data.plotting import confusion_matrix, image_grid


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def make_log_confusion_matrix_fn(
        model, file_writer_cm, file_writer_wrong, test_ds, label_names,
        preproc_layer, include_ambient_noise=True):

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        y_pred = model.predict(test_ds)
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.concat(list(test_ds.map(lambda s, lab: lab)), axis=0)

        # Log incorrectly labelled examples as an image summary
        wrong = tf.where(
            tf.not_equal(
                tf.cast(y_true, tf.int32),
                tf.cast(y_pred, tf.int32))
        ).numpy()

        with file_writer_wrong.as_default():
            for label, species in enumerate(label_names):
                species_examples = tf.where(
                    tf.equal(tf.cast(y_true, tf.int32), label)
                ).numpy()
                species_ixs = np.intersect1d(wrong, species_examples, assume_unique=True)
                to_use = set(species_ixs.tolist())
                predicted = [p for p in y_pred.numpy()[species_ixs]]
                wrong_img = [
                    np.transpose(spec, [1,0,2])[::-1, :, :]
                    for i, spec in enumerate(
                        test_ds.map(
                            lambda x, l: preproc_layer(x)
                        ).unbatch().as_numpy_iterator())
                    if i in to_use]
                captions = [label_names[p] for p in predicted]
                fig = image_grid(images=wrong_img, labels=captions)
                all_wrong = plot_to_image(fig)
                tf.summary.image(f"incorrect/{species}", all_wrong, step=epoch)
        # Log the confusion matrix as an image summary.
        figure = confusion_matrix(y_true, y_pred, label_names, show=False)
        cm_image = plot_to_image(figure)
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    return log_confusion_matrix
