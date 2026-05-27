# Multimodal Emotion Recognition

A deep learning-based multimodal emotion recognition system that detects human emotions from speech and text using CNN, BiLSTM, and Transformer-based architectures.

Built using:

* [PyTorch](https://pytorch.org)
* [Streamlit](https://streamlit.io)
* [Librosa](https://librosa.org)
* [Transformers](https://huggingface.co/docs/transformers/index)

---

# Features

* Speech Emotion Recognition
* Text Emotion Recognition
* Fusion Pipeline
* Real-time Voice Recording
* Audio Upload Support
* Interactive Streamlit Frontend
* CNN + BiLSTM Speech Model
* BERT-based Text Model
* Emotion Confidence Scores
* MFCC Feature Extraction

---

# Detectable Emotions

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Pleasant Surprise
* Sad

---

# Project Structure

```text
Multimodal-Emotion-Recognition/
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   └── fusion_pipeline/
│
├── Results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Speech Pipeline

## Architecture

* MFCC Feature Extraction
* CNN Layers
* BiLSTM Layers
* Fully Connected Classifier

## Dataset

Speech model trained using:

* Toronto Emotional Speech Set (TESS)

## Speech Model Performance

| Metric      | Score    |
| ----------- | -------- |
| Accuracy    | 98.75%   |
| Classes     | 7        |
| Features    | 40 MFCC  |
| Sample Rate | 22050 Hz |

---

# Text Pipeline

## Architecture

* BERT Encoder
* Dense Classification Head

## Libraries

* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

# Fusion Pipeline

Combines:

* Speech Predictions
* Text Predictions

to improve overall emotion understanding.

---

# Frontend

Interactive frontend built using:

* Streamlit

Features:

* Upload WAV / MP3 / M4A files
* Record live audio
* Emotion visualization
* Confidence scores
* Real-time prediction

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/rishithav19/Multimodal-Emotion-Recognition.git
```

```bash
cd Multimodal-Emotion-Recognition
```

---

## 2. Create Virtual Environment

```bash
python3 -m venv venv
```

Activate environment:

### Linux / WSL

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Frontend

```bash
cd frontend
```

```bash
streamlit run app.py
```

---

# Train Speech Model

```bash
cd models/speech_pipeline
```

```bash
python3 train.py
```

---

# Test Speech Model

```bash
python3 test.py
```

---

# Technologies Used

| Technology   | Purpose               |
| ------------ | --------------------- |
| Python       | Core development      |
| PyTorch      | Deep learning         |
| Librosa      | Audio processing      |
| Streamlit    | Frontend              |
| Transformers | NLP models            |
| NumPy        | Numerical computation |

---

# Future Improvements

* Webcam Facial Emotion Detection
* Real-time Multimodal Fusion
* Attention-based Fusion Networks
* Deployment on Cloud
* Mobile Application Support

---

# Results

* High speech emotion recognition accuracy
* Real-time inference capability
* Interactive frontend
* Multimodal extensibility

---

# Author

Rishitha V

GitHub:
[rishithav19](https://github.com/rishithav19)
