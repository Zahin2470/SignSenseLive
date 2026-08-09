<div align="center">

# 🤟 SignSenseLive ✨

**Train your own real-time hand sign recognizer using a webcam - no pretrained sign-language dataset required.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Tracking-0A84FF?style=for-the-badge&logo=google&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-4CAF50?style=for-the-badge)

<br>

⭐ **If this project is useful to you, a star helps a lot!** ⭐

</div>

---

## 🗺️ Jump To

<div align="center">

| 🌟 | 🧠 | 🎮 | 🚀 | 📁 | 🛠 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| [Highlights](#-highlights) | [How It Works](#-how-it-works) | [Controls](#-controls) | [Quick Start](#-quick-start) | [Project Layout](#-project-layout) | [Extending It](#️-extending-signsenselive) |

</div>

---

## 🌟 Highlights

<table>
<tr>
<td width="50%" valign="top">

### 🖐️ Real-Time Hand Tracking
21-point MediaPipe hand landmark tracking, normalized to be translation- and scale-invariant — hand placement and camera distance never affect accuracy, only the hand's *shape* does.

### 🌲 Train-It-Yourself Classifier
No pretrained sign-language dataset. A `RandomForestClassifier` trains on *your* collected signs in under a second on CPU — fast, explainable, and realistic for the few-hundred-samples-per-class datasets a webcam session actually produces.

### 🎯 Temporal-Smoothed Prediction
Raw per-frame predictions flicker on live video — that's normal, not a bug. Majority-vote smoothing across the last 8 frames debounces it into a stable, confident reading.

</td>
<td width="50%" valign="top">

### 🎮 Practice / Match Mode
A quiz mode that prompts a random sign, tracks your streak, accuracy, and reaction time — turns the project from a demo into something you can actually train yourself with.

### 🔊 Text-to-Speech Output
Every recognized sign is spoken aloud in `live` mode — non-blocking (a background thread + queue, so it never freezes the camera feed) and fails silently if no TTS voice is available.

### 📊 Confusion Matrix Dashboard
Every `train` run renders a heatmap showing exactly which signs get mixed up with which — overall accuracy hides that a model can be 90% accurate while two specific signs are nearly indistinguishable to it.

</td>
</tr>
</table>

---

## 🧠 How It Works

```mermaid
flowchart LR
    A[📷 Camera] --> B[🖐️ MediaPipe<br/>21 landmarks]
    B --> C[📐 Feature<br/>Normalization]
    C --> D[🌲 RandomForest<br/>Training]
    D --> E[🎯 Live<br/>Prediction]
    E --> F[⏱️ Temporal<br/>Voting]
    F --> G[🔊 Speech +<br/>🎮 Practice Mode]

    style A fill:#3776AB,color:#fff,stroke:#fff
    style B fill:#0A84FF,color:#fff,stroke:#fff
    style C fill:#845EF7,color:#fff,stroke:#fff
    style D fill:#F7931E,color:#fff,stroke:#fff
    style E fill:#20C997,color:#fff,stroke:#fff
    style F fill:#FF6B6B,color:#fff,stroke:#fff
    style G fill:#E64980,color:#fff,stroke:#fff
```

<div align="center">

| Step | Command | What Happens |
|:---|:---:|:---|
| 1️⃣ **Collect** | `python main.py collect` | MediaPipe extracts 21 3D hand landmarks per frame. Each sample is normalized (wrist-centered, scale-normalized by the wrist→middle-knuckle bone) and saved under a label you type. |
| 2️⃣ **Train** | `python main.py train` | Normalized landmark vectors + your labels train a `RandomForestClassifier`. Accuracy is reported on a **held-out test split** — real generalization, not memorization — and a confusion-matrix heatmap is saved automatically. |
| 3️⃣ **Live** | `python main.py live` | Predicts every frame, smooths flicker with temporal voting, and **speaks each newly-recognized sign aloud**. |
| 4️⃣ **Practice** | `python main.py practice` | Quiz mode — it names a sign, you make it, it tracks your streak, accuracy, and reaction time. |

</div>

> [!TIP]
> **Why RandomForest, not a neural net?** Hand-collected datasets here are realistically a few hundred samples per sign, not thousands. A forest of shallow trees on well-normalized features generalizes better at that scale, trains instantly on a laptop CPU, and needs no GPU — a deliberate choice, not a shortcut.

---

## 🎮 Controls

### Collect Mode

| Key | Action |
|:---:|:---|
| `L` | Type a new label (e.g. `fist`, `peace`, `thumbs_up`) |
| `SPACE` | Toggle capture on/off — records a sample every couple of frames while active |
| `Q` / `Esc` | Exit |

> [!TIP]
> **Dataset recommendations:** collect each sign from a few different distances, angles, and positions in frame. Normalization handles scale/translation, but natural variety in *how* you hold the sign builds robustness instead of memorizing one exact pose. Aim for **40+ samples per sign** as a baseline — and if the confusion matrix later shows two signs mixed up, collect more of *those two specifically* rather than padding every class equally.

### Live & Practice Mode

| Key | Action |
|:---:|:---|
| `M` | Mute / unmute text-to-speech |
| `N` | *(Practice only)* Skip to a new random sign |
| `Q` / `Esc` | Exit — Practice mode prints a session summary (accuracy, best streak, avg reaction time) |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
```

Download the MediaPipe hand landmark model:

```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -O models/hand_landmarker.task
```

Run:

```bash
python main.py collect    # record a few signs — 40+ samples each
python main.py train      # train + evaluate, saves a confusion-matrix heatmap
python main.py live       # recognize signs live, speaks each one aloud
python main.py practice   # quiz mode — it prompts, you sign, it scores you
```

---

## 📁 Project Layout

```text
SignSenseLive/
├── main.py                    # CLI entry point: collect / train / live / practice
├── requirements.txt
├── signsense/
│   ├── tracker.py             # Single-hand MediaPipe wrapper
│   ├── features.py            # Landmark -> invariant feature vector transformer
│   ├── dataset.py             # CSV storage for labeled samples
│   ├── model.py                # RandomForest classifier train/predict/save/load
│   ├── voting.py               # Temporal majority-vote prediction smoothing (shared)
│   ├── confusion.py            # Confusion-matrix heatmap rendering
│   ├── speech.py               # Fail-soft, non-blocking text-to-speech
│   ├── collect.py             # Data collection camera application
│   ├── live.py                 # Live recognition app (voting + speech)
│   ├── practice.py             # Quiz / match mode
│   └── ui.py                   # UI design system (Poppins, glass panels, glow effect)
├── assets/fonts/               # Poppins font family (OFL-licensed)
├── models/
│   ├── hand_landmarker.task    # MediaPipe task model (you provide this)
│   ├── classifier.pkl          # Saved trained classifier
│   └── confusion_matrix.png    # Saved after every `train` run
└── data/
    └── samples.csv             # Labeled feature samples dataset
```

---

## 🛠️ Extending SignSenseLive

<table>
<tr>
<td width="50%" valign="top">

**➕ More signs**
Just run `collect` then `train` again — zero code changes. The classifier adapts to whatever labels exist in `data/samples.csv`.

**👐 Two-handed signs**
Bump `tracker.py`'s `max_hands` (default `1`) and extend `features.py` to concatenate both hands' feature vectors.

</td>
<td width="50%" valign="top">

**🏃 Motion-based signs**
Letters like J or Z involve movement, not a static pose — this architecture sees one frame at a time. You'd need a short landmark *sequence* per sample and a sequence model (or hand-engineered velocity features) instead of a single-frame classifier.

**🔄 Swap the classifier**
`SignClassifier` in `model.py` wraps a standard train/predict/save/load API — swap `RandomForestClassifier` for any scikit-learn-compatible estimator without touching `collect.py`, `live.py`, or `practice.py`.

</td>
</tr>
</table>

<div align="center">

**⭐ If you found this project useful, consider starring the repository. ⭐**

</div>
