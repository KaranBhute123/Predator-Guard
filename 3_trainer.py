import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
# Use raw string 'r' for Windows paths to avoid errors
DATASET_PATH = r"D:\animal detection\dataset\processed"
MODEL_SAVE_PATH = r"D:\animal detection\models\predator_guard_model.h5"

# Hardcoded Regularization Parameters
L2_REG = 0.01      # Penalty for complex weights
DROPOUT_RATE = 0.5 # Drop 50% of neurons to force learning

def build_hardened_model(input_shape, num_classes):
    """
    Robust Architecture with Dropout and L2 Regularization
    """
    model = Sequential([
        # Explicit Input Layer
        Input(shape=input_shape),

        # Layer 1
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2), 

        # Layer 2
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Layer 3
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(L2_REG)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        # Flatten & Dense
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE), 
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_labels_safely(path):
    """
    Safely loads the label map and handles 0-d array errors
    """
    try:
        class_names = np.load(path, allow_pickle=True)
        
        # FIX: Check if it is a 0-d array (scalar) and extract content
        if isinstance(class_names, np.ndarray) and class_names.ndim == 0:
            class_names = class_names.item()
            
        # Convert to standard list
        return list(class_names)
    except Exception as e:
        print(f"Warning: Could not load labels from file ({e}). Using Hardcoded list.")
        # Fallback Hardcoded List (Matches your dataset folders)
        return ['Ambient', 'Bear', 'Cow', 'Dog', 'Donkey', 'Elephant', 'Horse', 'Lion', 'Monkey']

def main():
    print("Loading Data...")
    # Load Data Arrays
    X_train = np.load(os.path.join(DATASET_PATH, "X_train.npy"))
    y_train = np.load(os.path.join(DATASET_PATH, "y_class_train.npy"))
    X_val = np.load(os.path.join(DATASET_PATH, "X_val.npy"))
    y_val = np.load(os.path.join(DATASET_PATH, "y_class_val.npy"))
    X_test = np.load(os.path.join(DATASET_PATH, "X_test.npy"))
    y_test = np.load(os.path.join(DATASET_PATH, "y_class_test.npy"))
    
    # Robust Label Loading
    class_names = load_labels_safely(os.path.join(DATASET_PATH, "label_map.npy"))
    
    print(f"Data Loaded. Training on {len(X_train)} samples.")
    print(f"Classes: {class_names}")

    # Build Model
    input_shape = (128, 63, 1) # Hardcoded to match your CNN architecture
    num_classes = len(class_names)
    
    model = build_hardened_model(input_shape, num_classes)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    # Train
    print("\nStarting Training (Robust Mode)...")
    history = model.fit(
        X_train, y_train,
        epochs=50, 
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # --- EVALUATION ---
    print("\n--- GENERATING REPORTS ---")
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report
    print(classification_report(y_test, y_pred_classes, target_names=class_names))

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names, 
                cmap='Blues')
    plt.title('Final Robust Confusion Matrix')
    plt.ylabel('Actual Animal')
    plt.xlabel('Predicted Animal')
    plt.show()

    # Training Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()