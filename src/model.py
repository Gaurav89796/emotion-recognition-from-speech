"""
model.py
--------
Defines deep learning model architectures for Speech Emotion Recognition:
  - CNN
  - LSTM
  - CNN + LSTM hybrid

All models take input of shape (n_features, time_steps, 1) for CNN-based
models, or (time_steps, n_features) for the pure LSTM model.
"""

from tensorflow.keras import layers, models


def build_cnn(input_shape, num_classes):
    """
    2D CNN operating on the feature map (e.g. MFCC + chroma + mel stacked)
    treated as an image: (n_features, time_steps, 1).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_lstm(input_shape, num_classes):
    """
    Pure LSTM model operating on sequences: (time_steps, n_features).
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),

        layers.LSTM(64),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_cnn_lstm(input_shape, num_classes):
    """
    Hybrid CNN + LSTM model.

    Input shape: (n_features, time_steps, 1)
    The CNN extracts local spectral patterns; the result is reshaped into
    a sequence and fed to an LSTM to capture temporal dynamics.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Reshape (freq, time, channels) -> (time, freq * channels)
        layers.Permute((2, 1, 3)),
        layers.Reshape((-1, model_inner_dim(input_shape))),

        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def model_inner_dim(input_shape):
    """
    Helper to compute the flattened feature dimension after two
    (2,2) max-pooling operations, used by build_cnn_lstm's Reshape layer.
    """
    n_features, _, channels = input_shape
    pooled_features = n_features // 2 // 2
    return pooled_features * 64  # 64 = number of filters in last Conv2D


def get_model(architecture, input_shape, num_classes):
    """
    Factory function to retrieve a model by name.

    Args:
        architecture: one of "cnn", "lstm", "cnn_lstm"
        input_shape: shape tuple for the Input layer
        num_classes: number of emotion classes

    Returns:
        Uncompiled tf.keras.Model
    """
    architecture = architecture.lower()
    if architecture == "cnn":
        return build_cnn(input_shape, num_classes)
    elif architecture == "lstm":
        return build_lstm(input_shape, num_classes)
    elif architecture == "cnn_lstm":
        return build_cnn_lstm(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
