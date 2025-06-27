# 🎵 Music Genre Classifier (React + FastAPI + PyTorch)

This project is an end-to-end **music genre classification system** built with:

- **React + Next.js + TypeScript** frontend for uploading `.mp3`/`.wav` files
- **FastAPI** backend that processes the audio and runs inference
- A custom-trained **CNN model in PyTorch** that classifies songs into one of 10 genres
- Fully containerized with **Docker** (optional)

---

## 🔍 Features

- Upload `.wav` or `.mp3` files
- Get predicted genre instantly on the UI
- View genre description and sample image
- Trained with the GTZAN dataset
- Clean, modern UI using TailwindCSS
- Fast, responsive, and easy to deploy

---

## 🧠 Supported Genres

- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## 🖼️ Demo



---

## 🛠 Tech Stack

| Layer        | Technology                  |
|-------------|-----------------------------|
| Frontend     | React, Next.js, TypeScript, TailwindCSS |
| Backend      | FastAPI, Python, Librosa, Torch |
| ML Model     | PyTorch CNN trained on Mel spectrograms |
| Audio Tools  | Librosa, ffmpeg             |
| Deployment   | Docker (optional)           |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music-genre-classifier.git
cd music-genre-classifier
```

### 2. Backend Setup (FastAPI)

```bash
cd backend
pip install -r requirements.txt
```

Run FastAPI server:

```bash
uvicorn main:app --reload
```

### 3. Frontend Setup (React + Next.js)

```bash
cd frontend
npm install
npm run dev
```