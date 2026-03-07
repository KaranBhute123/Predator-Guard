import os
import wave
import librosa
import numpy as np
import pandas as pd
from datetime import datetime

# --- CONFIGURATION ---
DATASET_PATH = r"dataset\raw"
OUTPUT_DIR = r"output video"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# THRESHOLDS (Matches your Live Detector)
FLATNESS_THRESHOLD = 0.05
THREAT_SPECIES = ["Aslan", "Lion", "Bear", "Elephant","Horse",]

def get_technical_metadata(file_path, filename):
    """
    Extracts low-level file properties using the 'wave' library.
    Target Columns: name, path, channels, sample_width, frame_rate, nframes, duration, size
    """
    try:
        file_size_kb = os.path.getsize(file_path) / 1024  # Size in KB
        
        with wave.open(file_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth() # Bytes per sample
            frame_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / frame_rate
            
        return {
            "name": filename,
            "path": file_path,
            "channels": channels,
            "sample_width": sample_width,
            "frame_rate": frame_rate,
            "nframes": n_frames,
            "duration": round(duration, 3),
            "size_kb": round(file_size_kb, 2)
        }
    except Exception as e:
        print(f"Error reading metadata for {filename}: {e}")
        return None

def analyze_behavior_features(file_path, filename, label):
    """
    Extracts AI features and applies 'Predator Guard' logic.
    Target Columns: file_name, label, mean_frequency, flatness_score, rms_energy, behavior, danger_level
    """
    try:
        y, sr = librosa.load(file_path, sr=None) # Load at native sampling rate
        
        # 1. Feature Extraction
        # Mean Frequency (Spectral Centroid)
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        # Flatness (Texture: Noisy vs Tonal)
        flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        # RMS Energy (Loudness)
        rms = np.mean(librosa.feature.rms(y=y))
        
        # 2. Project Logic (Behavior)
        behavior = "Aggressive" if flatness > FLATNESS_THRESHOLD else "Passive"
        
        # 3. Project Logic (Danger Level)
        # Danger = High if (Species is Dangerous AND Behavior is Aggressive)
        is_threat_species = label in THREAT_SPECIES
        
        if is_threat_species and behavior == "Aggressive":
            danger_level = "High"
        elif is_threat_species and behavior == "Passive":
            danger_level = "Moderate (Passive Threat)"
        else:
            danger_level = "Low (Safe)"

        return {
            "file_name": filename,
            "label": label,
            "mean_frequency_hz": round(centroid, 2),
            "flatness_score": round(flatness, 4),
            "rms_energy": round(rms, 4),
            "behavior": behavior,
            "danger_level": danger_level
        }
    except Exception as e:
        print(f"Error analyzing features for {filename}: {e}")
        return None

def main():
    print(f"Scanning Dataset at: {DATASET_PATH}")
    
    technical_data = []
    behavior_data = []
    
    # Iterate through all Class Folders (Aslan, Bear, etc.)
    for label in os.listdir(DATASET_PATH):
        class_dir = os.path.join(DATASET_PATH, label)
        
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Processing Class: {label}...")
        
        for audio_file in os.listdir(class_dir):
            if not audio_file.lower().endswith('.wav'):
                continue
                
            full_path = os.path.join(class_dir, audio_file)
            
            # 1. Get Technical Metadata
            meta = get_technical_metadata(full_path, audio_file)
            if meta:
                technical_data.append(meta)
            
            # 2. Get Behavioral Analysis
            features = analyze_behavior_features(full_path, audio_file, label)
            if features:
                behavior_data.append(features)

    # --- SAVE REPORTS ---
    # Report 1: Technical Specs
    df_tech = pd.DataFrame(technical_data)
    tech_path = os.path.join(OUTPUT_DIR, "1_dataset_technical_report.csv")
    df_tech.to_csv(tech_path, index=False)
    
    # Report 2: Behavior & Risk Analysis
    df_behavior = pd.DataFrame(behavior_data)
    behavior_path = os.path.join(OUTPUT_DIR, "2_behavior_risk_report.csv")
    df_behavior.to_csv(behavior_path, index=False)
    
    print("\n" + "="*40)
    print("ANALYSIS COMPLETE")
    print("="*40)
    print(f"Files Processed: {len(df_tech)}")
    print(f"1. Technical Report saved to: {tech_path}")
    print(f"2. Risk Report saved to: {behavior_path}")
    print("="*40)

if __name__ == "__main__":
    main()