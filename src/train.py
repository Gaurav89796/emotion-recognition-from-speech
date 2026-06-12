"""
train.py
--------
Loads extracted features and labels, splits into train/val/test,
trains a chosen architecture (CNN, LSTM, or CNN+LSTM), and saves
the trained model along with the label encoder.

Usage:
    python src/train.py --architecture cnn --epochs 50 --batch_size 32
"""

import os
import argparse
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from model import get_model


def load_data(features_path, labels_path):
    X = np.load(features_path)
    y = np.load(labels_path, allow_pickle=True)
    return X, y


def prepare_data(X, y, architecture):
    """
    Encode labels, normalize features, and reshape for the chosen architecture.
    """
    # Encode string labels -> integers -> one-hot
    encoder = LabelEncoder()
    y_int = encoder.fit_transform(y)
    y_onehot = to_categorical(y_int)

    # Normalize features (z-score per sample)
    mean = X.mean()
    std = X.std()
    X_norm = (X - mean) / (std + 1e-8)

    if architecture in ("cnn", "cnn_lstm"):
        # Shape: (samples, n_features, time_steps, 1)
        X_norm = X_norm[..., np.newaxis]
    elif architecture == "lstm":
        # Shape: (samples, time_steps, n_features) -> transpose last two dims
        X_norm = np.transpose(X_norm, (0, 2, 1))

    return X_norm, y_onehot, encoder, (mean, std)


def main():
    parser = argparse.ArgumentParser(description="Train a Speech Emotion Recognition model")
    parser.add_argument("--features", default="data/X_features.npy", help="Path to feature .npy file")
    parser.add_argument("--labels", default="data/y_labels.npy", help="Path to labels .npy file")
    parser.add_argument("--architecture", default="cnn", choices=["cnn", "lstm", "cnn_lstm"],
                         help="Model architecture to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    X, y = load_data(args.features, args.labels)
    print(f"Loaded {X.shape[0]} samples with feature shape {X.shape[1:]}")

    X, y, encoder, norm_stats = prepare_data(X, y, args.architecture)
    num_classes = y.shape[1]
    input_shape = X.shape[1:]

    print(f"Input shape: {input_shape}, Num classes: {num_classes}")
    print(f"Classes: {list(encoder.classes_)}")

    # Train / validation / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Build model
    model = get_model(args.architecture, input_shape, num_classes)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks
    checkpoint_path = os.path.join(args.output_dir, f"best_{args.architecture}_model.h5")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Save final model, label encoder, and normalization stats
    model.save(os.path.join(args.output_dir, f"final_{args.architecture}_model.h5"))

    with open(os.path.join(args.output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    with open(os.path.join(args.output_dir, "norm_stats.pkl"), "wb") as f:
        pickle.dump(norm_stats, f)

    print(f"\nModel and artifacts saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
