<div align="center">

# 🤟 SignSenseLive ✨

**Train your own real-time hand sign recognizer using a webcam — no pretrained sign-language dataset required.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-0A84FF)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 🌟 Highlights

- 🖐️ 21-point MediaPipe hand landmark tracking
- 🌲 RandomForest classifier for fast CPU inference
- 🎯 Train with your own custom signs
- ⚡ Real-time prediction with temporal smoothing
- 📦 Lightweight and beginner friendly

## 🧠 Pipeline

```text
Camera
   │
MediaPipe (21 landmarks)
   │
Feature Normalization
   │
RandomForest Training
   │
Live Prediction
   │
Temporal Voting
```
---

## 🔄 How It Works

```
 ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
 │ 1. `collect` │ ───► │  2. `train`  │ ───► │   3. `live`  │
 └──────────────┘      └──────────────┘      └──────────────┘

```

1. **`collect` — Extraction & Normalization**

* MediaPipe extracts 21 3D hand landmarks per frame.
* Each sample is normalized (wrist-centered, scale-normalized by the wrist→middle-knuckle bone vector).
* Hand placement and camera distance do not affect accuracy — only the hand *shape* matters.


2. **`train` — Model Fitting**

* Reads normalized landmark vectors along with user-assigned labels to train a `RandomForestClassifier` (scikit-learn).
* Evaluates performance on a held-out test split to ensure real accuracy over training-set memorization.


3. **`live` — Real-Time Inference & Smoothing**

* Predicts labels frame-by-frame.
* Smooths out noisy frame-to-frame flicker via **temporal voting** (majority vote across the last 8 frames) before committing a reading — analogous to debouncing a noisy hardware sensor.

---
## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

Download the MediaPipe task model:

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -O models/hand_landmarker.task
```

Run:

```bash
python main.py collect
python main.py train
python main.py live
```


---

## 🎮 Collect Mode Controls

| Key | Action |
| --- | --- |
| L | Type a new label (e.g. `fist`, `peace`, `thumbs_up`)|
| SPACE | Toggle capture on/off (records frames periodically while active)|
| Q / Esc | Exit application|

> [!TIP]
> **Dataset Recommendations:** Collect each sign from various distances, angles, and camera frame positions. Although normalization handles scale and translation, natural variety in hand posture builds model robustness and prevents overfitting. Aim for **40+ samples per sign** as a baseline.
> 

---

## 📁 Project Layout

```text
SignSense/
├── main.py                    # CLI entry point: collect / train / live
├── requirements.txt
├── signsense/
│   ├── tracker.py             # Single-hand MediaPipe wrapper
│   ├── features.py            # Landmark -> invariant feature vector transformer
│   ├── dataset.py             # CSV storage for labeled samples
│   ├── model.py               # RandomForest classifier train/predict/save/load
│   ├── collect.py             # Data collection camera application
│   ├── live.py                # Live recognition app with temporal voting
│   └── ui.py                  # UI design system (Poppins, glass panels, glow effect)
├── assets/fonts/              # Poppins font family (OFL-licensed)
├── models/
│   ├── hand_landmarker.task   # MediaPipe task model
│   └── classifier.pkl         # Saved trained classifier
└── data/
    └── samples.csv            # Labeled feature samples dataset

```

---

## 🛠️ Extending SignSense

* **➕ Additional Signs:** Add new signs by simply running `collect` and `train` — zero code modifications required. The classifier adapts dynamically to labels in `data/samples.csv`.

* **👐 Two-Handed Gestures:** Update `tracker.py`'s `max_hands` configuration (default is `1`) and update `features.py` to concatenate feature vectors from both hands.

* **🏃 Motion-Based Signs:** For dynamic signs requiring movement (e.g., dynamic letters like J or Z), extend the single-frame architecture to process sequences over time using landmark history buffers combined with a sequence model or hand-engineered velocity features.

* **🔄 Custom Classifiers:** `SignClassifier` in `model.py` provides a standard train/predict/save/load interface. Swap out `RandomForestClassifier` for any scikit-learn compatible estimator without breaking `collect.py` or `live.py`.

<div align="center">

**⭐ If you found this project useful, consider starring the repository.**

</div>