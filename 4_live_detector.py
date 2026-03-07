import os
import librosa
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime

# --- CONFIGURATION ---
TOKEN = "8310524934:AAHF7VF-WsOHm6lWaIMzYAbGHS1UT_ruATM"
CHAT_ID = "7039154615" 

# PATHS
MODEL_PATH = r"D:\animal detection\models\predator_guard_model.h5"
LABEL_PATH = r"D:\animal detection\dataset\processed\label_map.npy" 

# AUDIO SETTINGS
SR = 22050
DURATION = 3 

# THRESHOLDS
CONFIDENCE_THRESHOLD = 0.60 
FLATNESS_THRESHOLD = 0.05 
ROLLOFF_THRESHOLD = 2500  # <--- This controls the Distance logic

# 1. Load Model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
print("Loading AI Model...")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Load Labels
try:
    loaded_labels = np.load(LABEL_PATH, allow_pickle=True)
    if isinstance(loaded_labels, np.ndarray) and loaded_labels.ndim == 0:
        loaded_labels = loaded_labels.item()
    CLASS_LABELS = list(loaded_labels)
    print(f"✅ Loaded {len(CLASS_LABELS)} classes.")
except:
    CLASS_LABELS = ['Ambient', 'Aslan', 'Bear', 'Cow', 'Dog', 'Elephant', 'Horse', 'Monkey']

print("System Active.")

def send_telegram_alert(species, behavior, proximity, confidence):
    print(">>>  SENDING TELEGRAM ALERT...")
    message = (
        f" PREDATOR ALERT \n"
        f"--------------------------------\n"
        f" Animal: {species}\n"
        f" Status: {behavior}\n"
        f" Proximity: {proximity}\n"
        f" Confidence: {confidence*100:.1f}%\n"
        f" Time: {datetime.now().strftime('%H:%M:%S')}"
    )
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.post(url, data={'chat_id': CHAT_ID, 'text': message}, timeout=5)
        print(">>> Alert Sent Successfully")
    except Exception as e:
        print(f">>> Telegram Failed: {e}")

def analyze_audio(file_path):
    print(f"\nAnalyzing: {os.path.basename(file_path)}...")

    # Audio Input
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return

    # Feature Extraction
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_fixed = librosa.util.fix_length(mel_spec_db, size=63, axis=1)
    input_data = mel_spec_fixed[np.newaxis, ..., np.newaxis]
    
    # Prediction
    prediction = model.predict(input_data, verbose=0)
    confidence = np.max(prediction)
    class_index = np.argmax(prediction)
    
    if class_index >= len(CLASS_LABELS):
        return

    species_id = CLASS_LABELS[class_index]
    
    # Decision Module (Distance Calculation)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85))

    # LOGIC: High Rolloff = Loud/High Freq = "Near"
    behavior = "Aggressive" if flatness > FLATNESS_THRESHOLD else "Passive"
    proximity = "Near" if rolloff > ROLLOFF_THRESHOLD else "Far"
    
    # Logic Layer
    final_species = species_id
    is_high_threat = False

    if species_id == "Aslan" and behavior == "Passive":
        final_species = "False Alarm (Dog/Cow)"
        is_high_threat = False
    else:
        is_high_threat = (species_id in ["Aslan", "Bear", "Lion", "Elephant"]) and behavior == "Aggressive"

    # --- OUTPUT FIX ---
    # Now explicitly printing Distance/Proximity
    print(f"   -> Result: {final_species} ({behavior})")
    print(f"   -> Distance: {proximity} (Score: {int(rolloff)})") 
    print(f"   -> Confidence: {confidence:.2f}")

    if is_high_threat and confidence > CONFIDENCE_THRESHOLD:
        print("   -> THREAT CONFIRMED. SENDING ALERT...")
        send_telegram_alert(final_species, behavior, proximity, confidence)
    else:
        print("   -> No Action Required.")

if __name__ == "__main__":
    file = r'output video\Elephant_7.wav' # r'output video\Bear_50.wav'
    if os.path.exists(file):
        analyze_audio(file)
    else:
        print(f"File not found: {file}")