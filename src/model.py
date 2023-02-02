from tensorflow.keras import layers
from tensorflow.keras import models


def build_basic_model(train_ds, num_labels, preprocessing):
    # Instantiate the `tf.keras.layers.Normalization` layer.
    input_shape = train_ds.element_spec[0].shape[1:]
    print(f"Input Shape: {input_shape}")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing,
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(64, 3, activation='relu'),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()
    return model
