"""Neural architectures used in the new Sentinel-2 Anza experiment."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def encoder(image: tf.Tensor, dropout: float = 0.25) -> tf.Tensor:
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(image)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def scalar_head(x: tf.Tensor, dropout: float = 0.25) -> tf.Tensor:
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    return layers.Dense(1, name="z")(x)


def build_cnn(input_shape=(9, 9, 4), dropout: float = 0.25):
    image = layers.Input(shape=input_shape, name="image")
    x = layers.GlobalAveragePooling2D()(encoder(image, dropout))
    return models.Model(image, scalar_head(x, dropout), name="CNN")


def build_ratio_cnn(input_shape=(9, 9, 4), dropout: float = 0.25):
    image = layers.Input(shape=input_shape, name="image")
    ratio = layers.Input(shape=(1,), name="log_blue_green_ratio")
    x = layers.GlobalAveragePooling2D()(encoder(image, dropout))
    x = layers.Concatenate()([x, ratio])
    return models.Model([image, ratio], scalar_head(x, dropout), name="Ratio_CNN")


def build_cnn_bilstm(input_shape=(9, 9, 4), dropout: float = 0.25, units: int = 32):
    image = layers.Input(shape=input_shape, name="image")
    # The recurrent sequence consists of four spatial feature tokens, not time steps.
    x = encoder(image, dropout)
    x = layers.Reshape((4, 128), name="four_spatial_tokens")(x)
    x = layers.Bidirectional(layers.LSTM(units, return_sequences=False))(x)
    return models.Model(image, scalar_head(x, dropout), name="CNN_BiLSTM")


def build_dense_cnn(input_shape=(9, 9, 4), dropout: float = 0.25):
    image = layers.Input(shape=input_shape, name="image")
    x = encoder(image, dropout)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ZeroPadding2D(((0, 1), (0, 1)))(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    z_grid = layers.Conv2D(1, 1, activation="linear", name="z_grid")(x)
    return models.Model(image, z_grid, name="Dense_CNN")


def masked_patch_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Per-patch MSE over measured water labels only (target channels: z, mask)."""
    z = y_true[..., 0]
    mask = y_true[..., 1]
    pred = y_pred[..., 0]
    squared = tf.square(pred - z) * mask
    return tf.reduce_sum(squared, axis=(1, 2)) / tf.maximum(tf.reduce_sum(mask, axis=(1, 2)), 1.0)


def build_multitask_cnn(input_shape=(9, 9, 4), dropout: float = 0.25):
    image = layers.Input(shape=input_shape, name="image")
    features = layers.GlobalAveragePooling2D()(encoder(image, dropout))
    z = scalar_head(features, dropout)
    c = layers.Dense(64, activation="relu")(features)
    c = layers.Dropout(dropout)(c)
    c = layers.Dense(32, activation="relu")(c)
    depth_bin = layers.Dense(3, activation="softmax", name="depth_bin")(c)
    return models.Model(image, {"z": z, "depth_bin": depth_bin}, name="Depth_bin_CNN")


BUILDERS = {
    "CNN": build_cnn,
    "RatioCNN": build_ratio_cnn,
    "CNN_BiLSTM": build_cnn_bilstm,
    "DenseCNN": build_dense_cnn,
    "DepthBinCNN": build_multitask_cnn,
}
