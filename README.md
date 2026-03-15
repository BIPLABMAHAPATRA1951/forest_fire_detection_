# 🔥 Forest Fire Detection System

A real-time forest fire and smoke detection system built using YOLOv11 and Streamlit.

## 🎯 Problem Statement
Forest fires cause massive destruction every year. Early detection is critical to minimize damage. This project uses computer vision to automatically detect fire and smoke in forest images.

## 🧠 How It Works
1. User uploads a forest image
2. YOLOv11 model analyzes the image
3. Fire/smoke is detected with bounding boxes
4. Risk level is shown (Critical/High/Medium/Low)

## 🏗️ Architecture
```
Input Image → YOLOv11 Model → Bounding Box Detection → Risk Assessment → Output
```

## 🔍 Detection Classes
| Class | Risk Level |
|-------|-----------|
| Large Fire | 🔴 Critical |
| Medium Fire | 🟠 High |
| Small Fire | 🟡 Medium |
| Heavy Smoke | ⚫ High |
| Low Smoke | 🔵 Low |

## 🛠️ Tech Stack
- **Model:** YOLOv11 (Ultralytics)
- **Training:** PyTorch + CUDA (RTX 3050)
- **Dataset:** 1376 images, 5 classes (Roboflow)
- **Web App:** Streamlit
- **Language:** Python 3.12

## 📊 Model Performance
- Epochs: 100
- mAP50: ~0.46
- Precision: ~0.50
- Recall: ~0.50
- Training Time: ~2 hours on RTX 3050

## 🚀 Live Demo
[Click here to try the app](https://forestfiredetection-juyz8gbvnluz2hxwptj5op.streamlit.app)

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/BIPLABMAHAPATRA1951/forest_fire_detection_.git
cd forest_fire_detection_
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

## 📁 Project Structure
```
forest_fire_detection/
├── app.py              ← Streamlit web app
├── train.py            ← Model training script
├── predict.py          ← Prediction script
├── best.pt             ← Trained YOLOv11 model
├── requirements.txt    ← Dependencies
└── dataset/            ← Training data
    ├── train/
    ├── valid/
    └── test/
```

## 👨‍💻 Author
Biplab Mahapatra