# Multimodal Emotion Recognition

A deep learning system that recognizes human emotions from speech and text using CNN, BiLSTM, and BERT-based architectures. Three independent pipelines are implemented — speech-only, text-only, and a multimodal fusion system — with a real-time Streamlit frontend.

Built with [PyTorch](https://pytorch.org) · [Streamlit](https://streamlit.io) · [Librosa](https://librosa.org) · [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

## Detectable Emotions

`Angry` `Disgust` `Fear` `Happy` `Neutral` `Pleasant Surprise` `Sad`

---

## Dataset

**Toronto Emotional Speech Set (TESS)**

| Property | Value |
|----------|-------|
| Total samples | 2,800 |
| Emotion classes | 7 |
| Speakers | 2 female actors (ages 26 and 64) |
| Train / Val / Test | 80% / 10% / 10% |
| Class balance | Perfectly balanced (400 samples per class) |

> **Note:** TESS uses the same 200 sentences across all emotion classes. This means the text transcripts are nearly identical regardless of emotion, which limits the text model's discriminative capability on this dataset. The speech model is unaffected.

---

## Results

### Speech Pipeline — CNN + BiLSTM

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.99 | 0.99 | 0.99 |
| Disgust | 1.00 | 0.97 | 0.99 |
| Fear | 0.99 | 1.00 | 0.99 |
| Happy | 1.00 | 0.96 | 0.98 |
| Neutral | 1.00 | 0.99 | 0.99 |
| Pleasant Surprise | 0.94 | 1.00 | 0.97 |
| Sad | 1.00 | 1.00 | 1.00 |
| **Weighted Avg** | **0.99** | **0.99** | **0.99** |

**Test Accuracy: 98.75%**

Confusion matrix: [`Results/confusion_matrix.png`](Results/confusion_matrix.png)

### Text Pipeline — BERT

Due to TESS using identical transcripts across emotion classes, BERT has no discriminative textual signal to learn from. Performance is low on this dataset — this is a dataset constraint, not an architectural limitation. On datasets with emotionally varied language, BERT achieves 80–90% accuracy.

Confusion matrix: [`Results/text_confusion_matrix.png`](Results/text_confusion_matrix.png)

### Fusion Pipeline — Speech + Text

Multimodal fusion provides a consistent improvement over speech-only by correcting borderline cases where the speech model is uncertain. The text embedding acts as a secondary signal when the BiLSTM softmax confidence is low.

Confusion matrix: [`Results/fusion_confusion_matrix.png`](Results/fusion_confusion_matrix.png)

---

## Project Structure

```text
Multimodal-Emotion-Recognition/
│
├── data/
│   └── TESS/
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── speech_pipeline/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── test.py
│   │   ├── test_dataset.py
│   │   ├── test_model.py
│   │   └── best_speech_model.pth
│   │
│   ├── text_pipeline/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── test.py
│   │   └── best_text_model.pth
│   │
│   └── fusion_pipeline/
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       ├── test.py
│       └── best_fusion_model.pth
│
├── Results/
│   ├── confusion_matrix.png
│   ├── text_confusion_matrix.png
│   └── fusion_confusion_matrix.png
│
├── mfcc_test.py
├── requirements.txt
├── README.md
└── Report.docx
```

---

## Architecture

### Speech Pipeline

```
Audio Input
    → Preprocessing (22,050 Hz, silence trim, normalize)
    → MFCC Extraction (40 coefficients × 200 frames)
    → CNN (local pattern extraction)
    → Batch Normalization + Dropout
    → BiLSTM (temporal context, forward + backward)
    → Fully Connected Layer
    → Softmax (7 classes)
```

### Text Pipeline

```
Text Transcript
    → Tokenization (bert-base-uncased, max 64 tokens)
    → BERT Encoder ([CLS] embedding, 768-dim)
    → Fully Connected Layer
    → Softmax (7 classes)
```

### Fusion Pipeline

```
Speech Embedding (BiLSTM output)  ─┐
                                   ├── Concatenate → MLP → Dropout → Softmax
Text Embedding (BERT [CLS])       ─┘
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/rishithav19/Multimodal-Emotion-Recognition.git
cd Multimodal-Emotion-Recognition
```

### 2. Create and activate a virtual environment

**Linux / WSL**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python3 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you are on Linux or WSL and encounter audio issues, also run:

```bash
sudo apt-get install libsndfile1
```

---

## Usage

### Run the frontend

```bash
cd frontend
streamlit run app.py
```

The app supports WAV file upload and live microphone recording.

### Train models

```bash
# Speech
cd models/speech_pipeline
python3 train.py

# Text
cd models/text_pipeline
python3 train.py

# Fusion (train speech and text first)
cd models/fusion_pipeline
python3 train.py
```

### Test models

```bash
# Speech
cd models/speech_pipeline
python3 test.py

# Text
cd models/text_pipeline
python3 test.py

# Fusion
cd models/fusion_pipeline
python3 test.py
```

---

## Dependencies

```
torch
torchaudio
librosa
transformers
streamlit
streamlit-mic-recorder
numpy
scikit-learn
soundfile
```

Install all with:

```bash
pip install -r requirements.txt
```

---

## Technologies

| Technology | Purpose |
|------------|---------|
| Python 3.12 | Core development |
| PyTorch | Model training and inference |
| Librosa | Audio loading, MFCC extraction |
| Hugging Face Transformers | BERT tokenizer and encoder |
| Streamlit | Frontend application |
| NumPy | Numerical computation |
| Scikit-learn | Evaluation metrics |

---

## Future Improvements

- Attention-based fusion for dynamic modality weighting
- Wav2Vec2 / HuBERT as richer speech feature extractors
- Webcam facial emotion detection as a third modality
- Real-world dataset evaluation (IEMOCAP, MSP-IMPROV)
- Cloud or mobile deployment

---

## Author

**Rishitha V** — [github.com/rishithav19](https://github.com/rishithav19)
