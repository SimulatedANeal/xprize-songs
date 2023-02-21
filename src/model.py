import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


def build_preprocessing(train_ds, resize_to, translate_time=0.2):
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    print("Fitting Normalization layer to training set")
    norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))
    preprocessing = tf.keras.Sequential(
        [
            tf.keras.layers.Resizing(resize_to, resize_to),
            norm_layer,
            # spectrograms are transposed, so height is time dimension
            tf.keras.layers.RandomTranslation(
                height_factor=translate_time,
                width_factor=0.,
                fill_mode='wrap')
        ],
        name='preprocessing')
    return preprocessing


def build_basic_model(train_ds, num_labels, input_size, conv_layers=((32, 3), (64, 3)), hidden_dense_size=(128,)):
    input_shape = train_ds.element_spec[0].shape[1:]
    preprocessing = build_preprocessing(train_ds, resize_to=input_size)
    print(f"Input Shape: {input_shape}")
    model = models.Sequential(name="cricket_model")
    model.add(layers.Input(shape=input_shape))
    model.add(preprocessing)
    for num_filters, size in conv_layers:
        model.add(layers.Conv1D(num_filters, size, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    for hsize in hidden_dense_size:
        model.add(layers.Dense(hsize, activation='relu'))
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_labels))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='species_loss'),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name='species_accuracy')],)

    return model, preprocessing


def build_multitask_model(train_ds, num_labels, input_size, conv_layers=((32, 3), (64, 3)), hidden_dense_size=(128,)):
    input_shape = train_ds.element_spec[0].shape[1:]
    preprocessing = build_preprocessing(train_ds, resize_to=input_size)
    print(f"Input Shape: {input_shape}")
    spectrogram = layers.Input(shape=input_shape)
    x = preprocessing(spectrogram)
    for num_filters, size in conv_layers:
        x = layers.Conv1D(num_filters, size, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    for hsize in hidden_dense_size:
        x = layers.Dense(hsize, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
    embedding = x
    species_pred = layers.Dense(num_labels, name='species')(embedding)
    call_pred = layers.Dense(1, activation='sigmoid', name='call')(embedding)
    model = tf.keras.Model(inputs=spectrogram, outputs=[species_pred, call_pred])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss={
            "call": tf.keras.losses.BinaryCrossentropy(),
            "species": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        metrics={
            "species": ["accuracy"],
            "call": [
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalseNegatives(name='fn'),
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
            ]
        },
        loss_weights={"species": 0.5, "call": 0.5},
    )

    return model, preprocessing
