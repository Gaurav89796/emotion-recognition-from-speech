"""
preprocessing.py
-----------------
Handles audio loading, augmentation, and feature extraction (MFCCs + extras)
for the Speech Emotion Recognition project.

Supports building a labeled dataset from RAVDESS, TESS, and EMO-DB datasets.
Each dataset has its own filename convention for encoding emotion labels,
so dataset-specific parsers are provided below.
"""

import os
import glob
import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Emotion label maps for each dataset
# ---------------------------------------------------------------------------

# RAVDESS filename format: 03-01-06-01-02-01-12.wav
# The 3rd number (index 2) is the emotion code.
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# TESS filenames contain the emotion as a word, e.g. "OAF_back_angry.wav"
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "ps": "surprised",       # "ps" = pleasant surprise
    "sad": "sad",
}

# EMO-DB filenames encode emotion in the 6th character, e.g. "03a01Fa.wav"
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

# A unified label set we will train on. Adjust as needed.
TARGET_EMOTIONS = ["angry", "happy", "sad", "neutral", "fearful", "disgust", "surprised"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(file_path, sr=22050, n_mfcc=40, max_pad_len=174):
    """
    Load an audio file and extract a feature matrix consisting of:
      - MFCCs
      - Chroma STFT
      - Mel-spectrogram (in dB)
      - Spectral contrast
      - Tonnetz

    All features are stacked along the frequency axis and padded/truncated
    to a fixed number of time frames so every sample has the same shape.

    Returns:
        np.ndarray of shape (n_features, max_pad_len)
    """
    try:
        y, sample_rate = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)

    # If clip is too short, pad it
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    stft = np.abs(librosa.stft(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sample_rate)

    features = np.vstack([mfcc, chroma, mel_db, contrast, tonnetz])

    # Pad or truncate along time axis to a fixed length
    if features.shape[1] < max_pad_len:
        pad_width = max_pad_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode="constant")
    else:
        features = features[:, :max_pad_len]

    return features


# ---------------------------------------------------------------------------
# Dataset-specific label parsers
# ---------------------------------------------------------------------------

def parse_ravdess_label(file_path):
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    if len(parts) < 3:
        return None
    code = parts[2]
    return RAVDESS_EMOTION_MAP.get(code)


def parse_tess_label(file_path):
    filename = os.path.basename(file_path).lower()
    for key, emotion in TESS_EMOTION_MAP.items():
        if key in filename:
            return emotion
    return None


def parse_emodb_label(file_path):
    filename = os.path.basename(file_path)
    if len(filename) < 6:
        return None
    code = filename[5]
    return EMODB_EMOTION_MAP.get(code)


DATASET_PARSERS = {
    "ravdess": parse_ravdess_label,
    "tess": parse_tess_label,
    "emodb": parse_emodb_label,
}


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(dataset_dirs, n_mfcc=40, max_pad_len=174, target_emotions=None):
    """
    Walk through one or more dataset directories, extract features and labels.

    Args:
        dataset_dirs: dict mapping dataset name ("ravdess", "tess", "emodb")
                       to its root directory containing .wav files
                       (recursively searched).
        n_mfcc: number of MFCC coefficients to extract.
        max_pad_len: fixed number of time frames per sample.
        target_emotions: optional list restricting which emotion labels to keep.

    Returns:
        X: np.ndarray of shape (n_samples, n_features, max_pad_len)
        y: np.ndarray of shape (n_samples,) with string emotion labels
    """
    if target_emotions is None:
        target_emotions = TARGET_EMOTIONS

    X, y = [], []

    for dataset_name, root_dir in dataset_dirs.items():
        parser = DATASET_PARSERS.get(dataset_name)
        if parser is None:
            print(f"Unknown dataset: {dataset_name}, skipping.")
            continue

        wav_files = glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True)
        print(f"[{dataset_name}] Found {len(wav_files)} files in {root_dir}")

        for file_path in wav_files:
            label = parser(file_path)
            if label is None or label not in target_emotions:
                continue

            features = extract_features(file_path, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
            if features is None:
                continue

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage - update paths to where you've downloaded the datasets
    dataset_dirs = {
        "ravdess": "data/RAVDESS",
        "tess": "data/TESS",
        "emodb": "data/EMODB",
    }

    X, y = build_dataset(dataset_dirs)
    print("Feature matrix shape:", X.shape)
    print("Labels shape:", y.shape)

    np.save("data/X_features.npy", X)
    np.save("data/y_labels.npy", y)
