# 🎙️ Emotion Recognition from Speech

Recognize human emotions (happy, angry, sad, neutral, fearful, disgust, surprised) from speech audio using deep learning and speech signal processing.

## 📌 Objective

Build a system that analyzes raw speech audio and classifies the speaker's emotional state.

## 🛠️ Approach

Combines **speech signal processing** (feature extraction with MFCCs, chroma, mel-spectrogram, spectral contrast, tonnetz) with **deep learning** (CNN, LSTM, or CNN+LSTM) to classify emotions.

## ✨ Key Features

- **Feature Extraction**: MFCCs (Mel-Frequency Cepstral Coefficients) plus chroma, mel-spectrogram, spectral contrast, and tonnetz features, computed with `librosa`.
- **Deep Learning Models**: Choose between three architectures via a single flag:
  - `cnn` – 2D CNN over stacked spectral features
  - `lstm` – pure LSTM over time-sequenced features
  - `cnn_lstm` – CNN feature extractor followed by LSTM for temporal modeling
- **Datasets supported**:
  - [RAVDESS](https://zenodo.org/record/1188976)
  - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
  - [EMO-DB](http://emodb.bilderbar.info/start.html)

## 🧰 Tech Stack

- Python 3.9+
- librosa (audio processing & feature extraction)
- TensorFlow / Keras (model building & training)
- scikit-learn (label encoding, train/test split)
- NumPy, Pandas, Matplotlib, Seaborn

## 📂 Project Structure

```
emotion-recognition/
├── data/                     # Place RAVDESS/TESS/EMO-DB folders here
│   ├── RAVDESS/
│   ├── TESS/
│   └── EMODB/
├── models/                    # Saved models, label encoder, norm stats
├── src/
│   ├── preprocessing.py      # Audio loading, feature extraction, dataset builder
│   ├── model.py              # CNN / LSTM / CNN+LSTM architectures
│   ├── train.py               # Training script
│   └── predict.py             # Inference on new audio files
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/emotion-recognition-from-speech.git
cd emotion-recognition-from-speech
pip install -r requirements.txt
```

## ▶️ Usage

### 1. Prepare the data

Download RAVDESS, TESS, and/or EMO-DB and place them under `data/` (e.g. `data/RAVDESS`, `data/TESS`, `data/EMODB`).

### 2. Extract features and build the dataset

```bash
python src/preprocessing.py
```

This walks through the dataset folders, extracts MFCC + chroma + mel + contrast + tonnetz features for each clip, and saves:
- `data/X_features.npy` — feature matrices
- `data/y_labels.npy` — emotion labels

### 3. Train a model

```bash
python src/train.py --architecture cnn --epochs 50 --batch_size 32
```

Available `--architecture` options: `cnn`, `lstm`, `cnn_lstm`.

This trains the model with early stopping and learning-rate scheduling, evaluates it on a held-out test set, and saves to `models/`:
- `final_<architecture>_model.h5`
- `best_<architecture>_model.h5` (best validation checkpoint)
- `label_encoder.pkl`
- `norm_stats.pkl`

### 4. Predict emotion from a new audio file

```bash
python src/predict.py --file path/to/audio.wav --architecture cnn
```

Example output:

```
Predicted emotion: HAPPY

Class probabilities:
  happy       : 0.7421
  surprised   : 0.1203
  neutral     : 0.0812
  ...
```

## 📊 Results

> Add your model's accuracy, confusion matrix, and sample predictions here once training is complete.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).

## 📄 License

This project is licensed under the MIT License.
