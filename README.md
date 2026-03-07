# PredatorGuard - Animal Detection System

An intelligent animal detection and monitoring system that uses machine learning to identify and classify animal sounds in real-time. This project combines deep learning with audio processing to detect potential predators and provide alerts for wildlife protection.

## Overview

PredatorGuard is a comprehensive solution for wildlife monitoring that:
- **Downloads** animal sound recordings from Xeno-Canto (an online database)
- **Preprocesses** audio data using MFCC (Mel-Frequency Cepstral Coefficients)
- **Trains** a deep learning model to classify animal species from their sounds
- **Detects** animals in real-time audio streams
- **Analyzes** detection patterns and statistics
- **Visualizes** insights through an interactive dashboard

## Project Structure

```
├── 1_downloader.py          # Downloads animal sound recordings
├── 2_preprocessor.py        # Processes raw audio data into ML-ready format
├── 3_trainer.py             # Trains deep learning classification model
├── 4_live_detector.py       # Real-time animal detection from audio streams
├── 5_analysis.py            # Analyzes detection results and generates reports
├── 6_dashboard.py           # Interactive dashboard for visualization
├── requirements.txt         # Python dependencies
├── dataset/
│   ├── raw/                 # Raw audio files organized by animal species
│   ├── processed/           # Preprocessed data (MFCC features, labels, splits)
│   └── Animal_Sound.csv     # Audio metadata
├── models/
│   └── predator_guard_model.h5  # Trained ML model
├── logs/                    # Training logs and outputs
└── output video/            # Generated video outputs
```

## Features

### 1. **Sound Recording Download** (`1_downloader.py`)
- Automatically downloads animal sound recordings from Xeno-Canto API
- Fetches multiple species including: Lion, Tiger, Leopard, Elephant, Hyena, Bear, Hippo, Wolf
- Organizes recordings by animal species

### 2. **Audio Preprocessing** (`2_preprocessor.py`)
- Converts raw audio to MFCC (Mel-Frequency Cepstral Coefficients) format
- Normalizes audio features for ML model compatibility
- Creates train/validation/test splits
- Generates label mappings for classification

### 3. **Model Training** (`3_trainer.py`)
- Builds hardened CNN (Convolutional Neural Network) architecture
- Implements L2 regularization and dropout for robustness
- Uses early stopping and learning rate reduction callbacks
- Validates performance with classification metrics

### 4. **Real-Time Detection** (`4_live_detector.py`)
- Monitors live audio feeds for animal sounds
- Classifications in real-time
- Low-latency detection suitable for field deployment

### 5. **Analysis & Reporting** (`5_analysis.py`)
- Generates detailed detection statistics
- Analyzes temporal patterns and confidence scores
- Creates performance reports

### 6. **Interactive Dashboard** (`6_dashboard.py`)
- Web-based visualization using Streamlit
- Real-time monitoring interface
- Detection history and analytics

## Dependencies

- **Deep Learning**: TensorFlow, Keras
- **Audio Processing**: librosa, PyAudio
- **Data Processing**: NumPy, Pandas, Matplotlib
- **Visualization**: Altair, Streamlit, Seaborn
- **Utilities**: scikit-learn

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/PredatorGuard.git
cd PredatorGuard
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv animal
.\animal\Scripts\activate  # Windows
source animal/bin/activate # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Download Animal Sound Data
```bash
python 1_downloader.py
```
This downloads sound recordings from Xeno-Canto for various animal species.

### Step 2: Preprocess Audio Data
```bash
python 2_preprocessor.py
```
Converts raw audio to MFCC features and creates train/test splits.

### Step 3: Train the Model
```bash
python 3_trainer.py
```
Trains the deep learning model on preprocessed audio features.

### Step 4: Run Live Detection
```bash
python 4_live_detector.py
```
Monitors audio input for real-time animal detection.

### Step 5: Analyze Results
```bash
python 5_analysis.py
```
Generates detailed performance analysis and statistics.

### Step 6: Launch Dashboard
```bash
streamlit run 6_dashboard.py
```
Opens interactive web dashboard for visualization and monitoring.

## Machine Learning Model

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Shape**: MFCC features from audio
- **Output**: Animal species classification
- **Regularization**: L2 regularization + Dropout
- **Callbacks**: Early stopping, Model checkpointing, Learning rate reduction
- **Performance**: Validated on separate test set

## Supported Animals

The system can detect and classify the following species:
- Lion
- Tiger
- Leopard
- Elephant
- Hyena
- Bear
- Hippo
- Wolf

## Data Sources

- **Audio Recordings**: Xeno-Canto (https://www.xeno-canto.org)
- **Metadata**: Processed from sound recordings

## System Requirements

- Python 3.8+
- 4GB RAM (minimum)
- Audio input device (for live detection)
- Microphone or audio line-in

## Performance Metrics

The trained model achieves classification accuracy on a held-out test set. Performance metrics include:
- Precision, Recall, and F1-score per animal species
- Confusion matrix analysis
- ROC curves and AUC scores

## Future Improvements

- [ ] Multi-species simultaneous detection
- [ ] Integration with ecosystem sensors
- [ ] Mobile app deployment
- [ ] Cloud-based real-time monitoring
- [ ] Improved model with transfer learning
- [ ] Support for additional animal species
- [ ] Wildlife protection alerting system

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add some feature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Authors

- Your Name - Initial development

## Acknowledgments

- Xeno-Canto for providing the animal sound dataset
- TensorFlow/Keras for deep learning framework
- Librosa for audio processing
- Streamlit for dashboard visualization

## Contact & Support

For issues, questions, or suggestions, please open an issue on GitHub or contact the development team.

---

**Note**: This project is designed for wildlife protection and educational purposes. Ensure compliance with local wildlife protection laws when using audio recording equipment.
