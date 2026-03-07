import os
import numpy as np
import librosa
import splitfolders
from glob import glob

# CONFIGURATION
SOURCE_FOLDER = "dataset/raw"
OUTPUT_FOLDER = "dataset/processed"
IMG_HEIGHT = 64   # Mel-bins (Y-axis of the image)
IMG_WIDTH = 128   # Time-steps (X-axis, depends on duration)
DURATION = 3      # Seconds
SAMPLE_RATE = 22050

def create_dataset():
    print("Step 1: Splitting data into Train/Val/Test...")
    splitfolders.ratio(SOURCE_FOLDER, output=OUTPUT_FOLDER, 
                       seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
    
    # define classes
    subfolders = sorted(os.listdir(f"{OUTPUT_FOLDER}/train"))
    class_map = {name: i for i, name in enumerate(subfolders)}
    print(f"Classes found: {class_map}")

    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} set...")
        X_spec = []      
        y_class = []     
        y_agg = []       
        y_dist = []      

        path = f"{OUTPUT_FOLDER}/{split}"
        
        for class_name, label_id in class_map.items():
            files = glob(f"{path}/{class_name}/*.wav")
            
            for file_path in files:
                try:
                 
                    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                    if len(y) < SAMPLE_RATE * DURATION:
                        y = librosa.util.fix_length(y, size=SAMPLE_RATE * DURATION)
                    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_HEIGHT)
                    mel_db = librosa.power_to_db(mel, ref=np.max)
                    
                    
                    if mel_db.shape[1] > IMG_WIDTH: mel_db = mel_db[:, :IMG_WIDTH]
                    else: mel_db = np.pad(mel_db, ((0,0), (0, IMG_WIDTH - mel_db.shape[1])))

                    # Physics Features (Aggression)
                    
                    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
                    
                    X_spec.append(mel_db)
                    y_class.append(label_id)
                    y_agg.append(flatness)
                    y_dist.append(0)

                    
                    if split == 'train' and class_name != 'Ambient':
                        y_far = librosa.effects.preemphasis(y, coef=0.97) 
                        y_far = y_far * 0.4 
                        
                        mel_far = librosa.feature.melspectrogram(y=y_far, sr=sr, n_mels=IMG_HEIGHT)
                        mel_far_db = librosa.power_to_db(mel_far, ref=np.max)
                        
                        if mel_far_db.shape[1] > IMG_WIDTH: mel_far_db = mel_far_db[:, :IMG_WIDTH]
                        else: mel_far_db = np.pad(mel_far_db, ((0,0), (0, IMG_WIDTH - mel_far_db.shape[1])))
                        
                        X_spec.append(mel_far_db)
                        y_class.append(label_id)
                        y_agg.append(flatness * 0.5)
                        y_dist.append(1) 

                except Exception as e:
                    print(f"Error file {file_path}: {e}")

        X_spec = np.array(X_spec)[..., np.newaxis] 
        y_class = np.array(y_class)
        y_agg = np.array(y_agg)
        y_dist = np.array(y_dist)

        print(f"Saving {split} data: {len(X_spec)} samples.")
        np.save(f"{OUTPUT_FOLDER}/X_{split}.npy", X_spec)
        np.save(f"{OUTPUT_FOLDER}/y_class_{split}.npy", y_class)
        np.save(f"{OUTPUT_FOLDER}/y_agg_{split}.npy", y_agg)
        np.save(f"{OUTPUT_FOLDER}/y_dist_{split}.npy", y_dist)

    np.save(f"{OUTPUT_FOLDER}/label_map.npy", class_map)
    print("\n[SUCCESS] All data processed and saved!")

if __name__ == "__main__":
    create_dataset()