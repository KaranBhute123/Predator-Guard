import streamlit as st
import os
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import tempfile

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Predator Guard Dashboard", page_icon="🦁", layout="wide")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/375/375048.png", width=100)
st.sidebar.title(" Settings")

# 1. Telegram Settings
st.sidebar.header("Telegram Bot")
TOKEN = st.sidebar.text_input("Bot Token", value="48rh4rXXXXXXXXXXXXX", type="password")
CHAT_ID = st.sidebar.text_input("Chat ID", value="23m4n4XXXXXXX")
ENABLE_ALERTS = st.sidebar.checkbox("Enable Real Telegram Alerts", value=False)

# 2. Thresholds (Live Tuning)
st.sidebar.header("Decision Thresholds")
CONF_THRESH = st.sidebar.slider("Confidence Threshold (%)", 0.0, 1.0, 0.60)
FLAT_THRESH = st.sidebar.slider("Flatness (Aggression)", 0.001, 0.1, 0.05, format="%.3f")
ROLL_THRESH = st.sidebar.slider("Rolloff (Distance)", 500, 5000, 2500)

# --- PATHS ---
MODEL_PATH = r"D:\animal detection\models\predator_guard_model.h5"
LABEL_PATH = r"D:\animal detection\dataset\processed\label_map.npy"

# --- LOAD MODEL & LABELS (Cached for Speed) ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f" Model not found at {MODEL_PATH}")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    try:
        loaded = np.load(LABEL_PATH, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.ndim == 0:
            loaded = loaded.item()
        return list(loaded)
    except:
        return ['Ambient', 'Aslan', 'Bear', 'Cow', 'Dog', 'Elephant', 'Horse', 'Monkey']

model = load_model()
CLASSES = load_labels()

# --- ANALYSIS LOGIC ---
def send_telegram(species, behavior, proximity, conf):
    msg = (f"*PREDATOR ALERT* \n"
           f"Animal: {species}\n"
           f" Status: {behavior}\n"
           f" Proximity: {proximity}\n"
           f"Conf: {conf*100:.1f}%")
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'}, timeout=3)
        return True
    except Exception as e:
        return False

def analyze_audio(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050, duration=3)
    
    # 2. Preprocess (Magic Width 63)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_fixed = librosa.util.fix_length(mel_db, size=63, axis=1)
    input_data = mel_fixed[np.newaxis, ..., np.newaxis]

    # 3. Predict
    preds = model.predict(input_data, verbose=0)
    class_idx = np.argmax(preds)
    conf = np.max(preds)
    species = CLASSES[class_idx]

    # 4. Physics Logic
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))

    behavior = "Aggressive" if flatness > FLAT_THRESH else "Passive"
    proximity = "Near" if rolloff > ROLL_THRESH else "Far"

    # 5. Safety Logic
    final_species = species
    is_threat = False

    # False Alarm Filter
    if species in ["Aslan", "Lion"] and behavior == "Passive":
        final_species = f"{species} (False Alarm)"
    else:
        # Threat List: Aslan, Lion, Bear, Elephant
        if species in ["Aslan", "Lion", "Bear", "Elephant"] and behavior == "Aggressive":
            is_threat = True
    
    return final_species, behavior, proximity, conf, is_threat, y, sr, mel_db, flatness, rolloff

# --- MAIN DASHBOARD UI ---
st.title(" Alert System: Localhost Command Center")
st.markdown("---")

# File Uploader
uploaded_file = st.file_uploader("Upload an Audio File (.wav)", type=["wav"])

if uploaded_file is not None:
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Run Analysis
    with st.spinner("Analyzing Audio Signature..."):
        species, behav, dist, conf, threat, y, sr, mel, flat, roll = analyze_audio(tmp_path)
    
    # Delete temp file
    os.remove(tmp_path)

    # --- RESULTS SECTION ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Detected Animal", species)
    with col2:
        st.metric("Behavior", behav, delta=f"{flat:.4f} Flatness")
    with col3:
        st.metric("Proximity", dist, delta=f"{int(roll)} Hz Rolloff")
    with col4:
        st.metric("AI Confidence", f"{conf*100:.1f}%")

    # --- ALERT STATUS ---
    st.markdown("### 🚦 System Status")
    if threat and conf > CONF_THRESH:
        st.error(" THREAT CONFIRMED: High Danger Level")
        if ENABLE_ALERTS:
            if send_telegram(species, behav, dist, conf):
                st.success("✅ Telegram Alert Sent!")
            else:
                st.warning("❌ Telegram Failed (Check Internet/Token)")
        else:
            st.info("ℹ️ Telegram Alerts Disabled in Sidebar")
    elif threat:
         st.warning("⚠️ Potential Threat (Ignored due to Low Confidence)")
    else:
        st.success("✅ Safe: No Action Required")

    # --- VISUALIZATION ---
    st.markdown("### 📊 Audio Forensics")
    tab1, tab2 = st.tabs(["Waveform", "Spectrogram"])
    
    with tab1:
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='blue')
        st.pyplot(fig_wave)
        st.audio(uploaded_file, format='audio/wav')
        
    with tab2:
        fig_spec, ax_spec = plt.subplots(figsize=(10, 3))
        img = librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=sr, ax=ax_spec, cmap='inferno')
        fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
        st.pyplot(fig_spec)

else:
    st.info("👋 Upload a .wav file above to start detection.")
