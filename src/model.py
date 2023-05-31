import json
import os

import tensorflow as tf
from tensorflow.keras import layers


MS = 1000


def macro_f1(y, y_hat, thresh=0.5):
    """Compute the macro F1-score on a batch of observations (average F1 across labels)

    Args:
        y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        thresh: probability value above which we predict positive

    Returns:
        macro_f1 (scalar Tensor): value of macro F1 for the batch
    """
    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)
    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)
    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)
    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)
    f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    macro_f1 = tf.reduce_mean(f1, name='macro_f1')
    return macro_f1


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


def build_embedder(input_shape, preprocessing_layer, convolutional_layers, dense_layers):
    embedding_model = tf.keras.Sequential(name='spectrogram_embedder')
    embedding_model.add(layers.Input(shape=input_shape, name="spectrogram_raw"))
    embedding_model.add(preprocessing_layer)
    for num_filters, size in convolutional_layers:
        embedding_model.add(layers.Conv1D(num_filters, size, activation='relu'))
    embedding_model.add(layers.Dropout(0.25))
    embedding_model.add(layers.Flatten())
    for hidden_size in dense_layers[:-1]:
        embedding_model.add(layers.Dense(hidden_size, activation='relu'))
        embedding_model.add(layers.Dropout(0.45))
    embedding_model.add(layers.BatchNormalization())
    embedding_model.add(layers.Dense(dense_layers[-1], activation='tanh', name='embedding'))
    embedding_model.add(layers.Dropout(0.3))
    return embedding_model



def build_model(
        train_ds, labels, input_size,
        sample_rate, nfft, fft_window_ms, fft_window_stride_ms,
        conv_layers=((32, 3), (64, 3)),
        hidden_dense_size=(96, 64),
        species_hidden_layers=(64,),
        learning_rate=1e-5):

    input_shape = train_ds.element_spec[0].shape[1:]
    print(f"Input Shape: {input_shape}")
    preprocessing = build_preprocessing(train_ds, resize_to=input_size)
    window = int(sample_rate * fft_window_ms / MS)
    stride = int(sample_rate * fft_window_stride_ms / MS)
    print(f"Using nfft {nfft}, window {window}, and stride {stride}")


    class CricketModel(tf.keras.Model):

        def __init__(
                self, nfft, nfft_window, nfft_window_stride,
                embedding_model, label_names,
                species_hidden_layers=(),
                name='cricket_model',
                **kwargs):
            super().__init__(name=name, **kwargs)
            self._nfft = nfft
            self._fft_window = nfft_window
            self._fft_stride = nfft_window_stride
            self._labels = label_names
            self._species_dense_stack = species_hidden_layers
            self.embedding_model = embedding_model
            self.call_layer = layers.Dense(1, activation='sigmoid', name='call')
            self.species_stack = tf.keras.Sequential(name='species')
            for size in self._species_dense_stack:
                self.species_stack.add(layers.Dense(size, activation='relu'))
                self.species_stack.add(layers.Dropout(0.45))
            self.species_stack.add(layers.Dense(
                units=len(self._labels),
                activation='softmax',
                name='species_id'))

        def call(self, inputs, training=None, mask=None):
            embeddings = self.embedding_model(inputs, training=training)
            call_pred = self.call_layer(embeddings)
            species_pred = self.species_stack(embeddings)
            return {'call': call_pred, 'species': species_pred}

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None], dtype=tf.float32)
        ])
        def audio_to_spectrogram(self, raw_audio_data):
            # Convert the waveform to a spectrogram via STFT.
            spectrogram = tf.signal.stft(
                raw_audio_data,
                fft_length=self._nfft,
                frame_length=self._fft_window,
                frame_step=self._fft_stride)
            # Obtain the magnitude of the STFT.
            spectrogram = tf.abs(spectrogram)
            # Add a `channels` dimension
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram

        @tf.function(input_signature=[
            tf.TensorSpec(
                shape=[None, None, nfft // 2 + 1, 1],
                dtype=tf.float32)
        ])
        def embed(self, spectrogram):
            return self.embedding_model(spectrogram, training=False)

        @tf.function(input_signature=[
            tf.TensorSpec(
                shape=[None, hidden_dense_size[-1]],
                dtype=tf.float32)
        ])
        def get_call_probability(self, embedding):
            return self.call_layer(embedding, training=False)

        @tf.function(input_signature=[
            tf.TensorSpec(
                shape=[None, hidden_dense_size[-1]],
                dtype=tf.float32)
        ])
        def predict_species(self, embedding):
            return self.species_stack(embedding, training=False)

        def get_config(self):
            config = super().get_config()
            config.update({
                "nfft": self._nfft,
                "nfft_window": self._fft_window,
                "nfft_window_stride": self._fft_stride,
                "embedding_model": self.embedding_model,
                "species_hidden_layers": self._species_dense_stack,
                "label_names": self._labels,
            })
            return config


    embed_spectrogram = build_embedder(
        input_shape=input_shape,
        preprocessing_layer=preprocessing,
        convolutional_layers=conv_layers,
        dense_layers=hidden_dense_size)

    model = CricketModel(
        nfft=nfft,
        nfft_window=window,
        nfft_window_stride=stride,
        embedding_model=embed_spectrogram,
        label_names=labels,
        species_hidden_layers=species_hidden_layers)

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),
        loss={
            "species": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            "call": tf.keras.losses.BinaryCrossentropy(from_logits=False),
        },
        metrics={
            "species": [
                "accuracy",
                "categorical_crossentropy",
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                macro_f1
            ],
            "call": [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
            ]
        },
        loss_weights={"species": 0.75, "call": 0.25},
    )

    return model, preprocessing


def save_model(model, config, directory):
    config_fp= os.path.join(directory, 'config.json')
    model_fp = os.path.join(directory, 'model')
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(config_fp, 'w') as f:
        json.dump(config, f)
    model.save(model_fp)


def load_model(directory):
    config_fp= os.path.join(directory, 'config.json')
    model_fp = os.path.join(directory, 'model')
    with open(config_fp, 'r') as f:
        config = json.load(f)
    model = tf.keras.models.load_model(
        model_fp, custom_objects={
            'macro_f1': macro_f1,
        })
    return model, config
