"""
predict.py
----------
Run inference on a new audio file using a trained Speech Emotion
Recognition model.

Usage:
    python src/predict.py --file path/to/audio.wav --architecture cnn
"""

import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import extract_features


def load_artifacts(model_dir, architecture):
    model_path = f"{model_dir}/final_{architecture}_model.h5"
    model = load_model(model_path)

    with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open(f"{model_dir}/norm_stats.pkl", "rb") as f:
        norm_stats = pickle.load(f)

    return model, encoder, norm_stats


def preprocess_for_inference(file_path, architecture, norm_stats, n_mfcc=40, max_pad_len=174):
    features = extract_features(file_path, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
    if features is None:
        raise ValueError(f"Could not extract features from {file_path}")

    mean, std = norm_stats
    features_norm = (features - mean) / (std + 1e-8)

    if architecture in ("cnn", "cnn_lstm"):
        # (n_features, time_steps) -> (1, n_features, time_steps, 1)
        features_norm = features_norm[np.newaxis, ..., np.newaxis]
    elif architecture == "lstm":
        # (n_features, time_steps) -> (1, time_steps, n_features)
        features_norm = np.transpose(features_norm, (1, 0))[np.newaxis, ...]

    return features_norm


def predict_emotion(file_path, architecture="cnn", model_dir="models"):
    model, encoder, norm_stats = load_artifacts(model_dir, architecture)
    X = preprocess_for_inference(file_path, architecture, norm_stats)

    probabilities = model.predict(X, verbose=0)[0]
    predicted_idx = np.argmax(probabilities)
    predicted_label = encoder.inverse_transform([predicted_idx])[0]

    results = {
        label: float(prob)
        for label, prob in zip(encoder.classes_, probabilities)
    }

    return predicted_label, results


def main():
    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file")
    parser.add_argument("--file", required=True, help="Path to the .wav audio file")
    parser.add_argument("--architecture", default="cnn", choices=["cnn", "lstm", "cnn_lstm"])
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    predicted_label, probabilities = predict_emotion(args.file, args.architecture, args.model_dir)

    print(f"\nPredicted emotion: {predicted_label.upper()}\n")
    print("Class probabilities:")
    for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label:12s}: {prob:.4f}")


if __name__ == "__main__":
    main()
