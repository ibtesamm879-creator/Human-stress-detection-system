CalmSphere AI — Facial Expression Stress Analytics & Wellness Suite
CalmSphere AI is a state-of-the-art, production-ready Final Year Project (FYP) application developed to detect cognitive and physiological stress from human facial expressions. It combines a deep learning backend with an immersive, premium glassmorphic frontend built entirely using modern native web standards.
The model is a transfer-learned MobileNetV2 neural network trained using Keras on the Roboflow Stress Detection dataset, yielding highly accurate, binary classification predictions ("Stress" vs "Non Stress").
🚀 Key Features
1. Visual Diagnosis Console
Drag & Drop Face Uploader: Supports drag-and-drop face portrait scanning with an active loading state.
Webcam Real-time Snapshot Scanner: Integrated camera snapshot capabilities using HTML5 navigator.mediaDevices.getUserMedia with a glowing laser scanner animation.
Sample Portrait Testing Panel: One-click test cards powered by professional portraits linked to the backend via URL analysis.
2. Comprehensive Stress Analytics
Historical Charting (Chart.js): Line charts tracking users' daily stress scan levels.
Persistence: Data is saved to the browser's localStorage so stress history persists across sessions.
Dynamic AI Wellness Advice: Renders tailored clinical suggestions and coping tips depending on the detected stress level (Normal, Moderate, Severe).
3. Integrated Stress Mitigation Hub
Interactive Breathing Pacer: Syncs respiration with an animated expanding and contracting neon bubble (Inhale, Hold, Exhale).
Ambient Soundscapes (Web Audio API): Real-time browser-native synthesizer for 528Hz Solfeggio frequencies (DNA repair harmonic), Alpha Binaural brainwaves, and calming brown noise—operates 100% locally with zero external audio assets.
Daily Resilience Log: Checklists for tracking sleep, hydration, exercise, and mindfulness that dynamically adjust a custom stress resilience score.
4. Transfer Learning Visualizer
Embedded graphical walkthrough in the frontend displaying the complete MobileNetV2 architecture.
🛠️ Architecture & Preprocessing Pipeline
graph LR
    A[Upload Face Image] --> B[Resize to 224x224 RGB]
    B --> C[Normalize Pixels by 1/255.0]
    C --> D[Expand Dimensions to 1,224,224,3]
    D --> E[MobileNetV2 Base Weights: ImageNet, Frozen]
    E --> F[GlobalAveragePooling2D]
    F --> G[Dropout]
    G --> H[Dense Sigmoid Layer]
    H --> I[Inference Output: 0.0 - 1.0]
📦 Prerequisites & Installation
To run this project locally, ensure you have Python 3.10+ installed.
1. Install Dependencies
Run the following command to install the required Python packages:
pip install tensorflow fastapi uvicorn pillow python-multipart numpy
🏃 How to Run the App
Ensure your trained model file stress_detection_model.h5 is in the same directory as backend_app.py.
Start the FastAPI server by running:
python backend_app.py
Open your browser and navigate to:
http://localhost:8000
The server automatically serves the premium glassmorphic frontend at the root URL (/) and handles predictions at /predict and /predict_url.
📁 Project Directory Structure
