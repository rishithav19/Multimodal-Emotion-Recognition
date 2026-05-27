import streamlit as st
import torch
import librosa
import numpy as np
import tempfile
import sys
from streamlit_mic_recorder import mic_recorder

sys.path.append("../models/speech_pipeline")
from model import SpeechEmotionModel

st.set_page_config(
    page_title="Multimodel Emotion Recognition",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background-color: #111110;
    color: #e8e4dc;
}

[data-testid="stSidebar"] {
    background-color: #1a1917;
    border-right: 1px solid #2e2c28;
}
[data-testid="stSidebar"] * {
    color: #9c9890 !important;
    font-family: 'Syne', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #d4a843 !important;
    font-size: 13px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

h1, h2, h3, h4 { color: #e8e4dc !important; }
p, label, div { color: #9c9890; }

[data-testid="stTabs"] button {
    color: #6b6860 !important;
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    background: transparent;
    border: none;
    padding: 10px 16px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #d4a843 !important;
    border-bottom: 2px solid #d4a843;
}

[data-testid="stFileUploader"] {
    background: #161513;
    border: 1px dashed #2e2c28;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #d4a843;
}
[data-testid="stFileUploader"] label {
    color: #9c9890 !important;
}

.stButton > button {
    background: #d4a843 !important;
    color: #111110 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 14px 24px !important;
    width: 100%;
    transition: opacity 0.15s !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    background: #d4a843 !important;
}

.stProgress > div > div {
    background: #2e2c28;
    border-radius: 4px;
}
.stProgress > div > div > div {
    background: linear-gradient(90deg, #d4a843, #e8c060) !important;
    border-radius: 4px;
}

.stSuccess {
    background: #1a1c17 !important;
    border: 1px solid #3a3a20 !important;
    border-radius: 10px !important;
    color: #c8d87a !important;
}
.stInfo {
    background: #161513 !important;
    border: 1px solid #2e2c28 !important;
    border-radius: 10px !important;
    color: #d4a843 !important;
}
.stWarning {
    background: #1c1710 !important;
    border: 1px solid #3e2e10 !important;
    border-radius: 10px !important;
    color: #d4a843 !important;
}

audio {
    width: 100%;
    border-radius: 8px;
    background: #161513;
    margin-top: 8px;
}

hr { border-color: #2e2c28 !important; }

[data-testid="stMetric"] {
    background: #161513;
    border: 1px solid #2e2c28;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricLabel"] { color: #5a5850 !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="stMetricValue"] { color: #d4a843 !important; font-size: 24px !important; }

.stSpinner > div { border-top-color: #d4a843 !important; }
</style>
""", unsafe_allow_html=True)

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"]
EMOTION_ICONS = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨", "Happy": "😊",
    "Neutral": "😐", "Pleasant Surprise": "😲", "Sad": "😢"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = SpeechEmotionModel().to(device)
    model.load_state_dict(torch.load(
        "../models/speech_pipeline/best_speech_model.pth",
        map_location=device
    ))
    model.eval()
    return model

model = load_model()

def extract_mfcc(audio_path):
    signal, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < 200:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, 200 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :200]
    return mfcc

def predict_emotion(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(mfcc_t)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    all_probs = probs.squeeze().cpu().numpy()
    return EMOTIONS[predicted.item()], confidence.item() * 100, all_probs

with st.sidebar:
    st.markdown("## 🎙 Emotion Recognition")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
Multimodal emotion recognition from speech using deep learning.

**Model:** CNN + BiLSTM  
**Features:** MFCC (40 coefficients)  
**Sample rate:** 22,050 Hz  
**Frames:** 200  
    """)
    st.markdown("---")
    st.markdown("### Detectable Emotions")
    for e in EMOTIONS:
        st.markdown(f"{EMOTION_ICONS[e]} {e}")

st.markdown("# 〜 Emotion Recognition")
st.markdown("Detect emotions from speech · CNN + BiLSTM · 7 classes")
st.markdown("---")

audio_path = None
tab1, tab2 = st.tabs(["  📁  Upload audio", "  🎙  Record voice"])

with tab1:
    uploaded_file = st.file_uploader("Drop a WAV file or click to browse", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(uploaded_file.read())
            audio_path = f.name
        st.session_state["recorded_audio_path"] = None

with tab2:
    st.markdown("Click **Start Recording** and speak into your microphone.")
    st.caption("Best results with clear speech and low background noise.")
    audio = mic_recorder(
        start_prompt="⏺ Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=False,
        key="recorder"
    )
    if audio and audio != st.session_state.get("last_audio"):
        st.session_state["last_audio"] = audio
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio["bytes"])
        tmp.close()
        st.session_state["recorded_audio_path"] = tmp.name

    if st.session_state.get("last_audio"):
        st.audio(st.session_state["last_audio"]["bytes"], format="audio/wav")

if audio_path is None and st.session_state.get("recorded_audio_path"):
    audio_path = st.session_state["recorded_audio_path"]

st.markdown("")
predict_button = st.button("Analyze Emotion", use_container_width=True)

if predict_button:
    if audio_path is None:
        st.warning("Please upload or record audio first.")
    else:
        with st.spinner("Analyzing emotion..."):
            emotion, confidence, all_probs = predict_emotion(audio_path)

        st.markdown("---")
        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected Emotion", f"{EMOTION_ICONS[emotion]}  {emotion}")
        with col2:
            st.metric("Confidence", f"{confidence:.1f}%")

        st.markdown("")
        st.progress(int(confidence))

        st.markdown("#### All emotion scores")
        for emo, prob in zip(EMOTIONS, all_probs):
            pct = prob * 100
            cols = st.columns([2, 5, 1])
            cols[0].markdown(f"`{EMOTION_ICONS[emo]} {emo}`")
            cols[1].progress(int(pct))
            cols[2].markdown(f"**{pct:.1f}%**")