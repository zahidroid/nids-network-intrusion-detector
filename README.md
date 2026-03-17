# 🛡️ AI Network Intrusion Detection System (NIDS)

> Real-time network traffic analysis — detects Port Scans, DDoS, and Brute Force attacks instantly

🔗 **[Live Demo → https://huggingface.co/spaces/zahidmohd/nids-network-intrusion-detector](https://huggingface.co/spaces/zahidmohd/nids-network-intrusion-detector)**

## What it does
- Classifies network traffic into: Normal / Port Scan / DDoS / Brute Force
- Shows probability distribution across all attack types
- Provides real Linux commands to mitigate each attack
- Neural Network trained on 8000 network traffic samples
- 100% accuracy across all 4 classes

## Model Architecture
- Input: 8 network features (packet size, packets/sec, bytes, ports, etc.)
- Hidden layers: 256 → 128 → 64 neurons with BatchNorm + Dropout
- Output: 4-class softmax classifier
- Training: 8000 samples | Test Accuracy: 100% | All classes F1: 1.00

## Tech Stack
- TensorFlow / Keras — model training
- Scikit-learn — preprocessing
- Gradio 6 — UI
- HuggingFace Hub — model hosting
- HuggingFace Spaces — deployment

## Run Locally
pip install tensorflow gradio huggingface_hub scikit-learn numpy
python app.py

## Built By
Zahid Mohammed 
