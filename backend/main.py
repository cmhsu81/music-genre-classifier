import torch
import torchaudio
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from model import CNNGenreClassifier
import torchvision.transforms as transforms
from PIL import Image

app = FastAPI()

# Allow CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and class names
MODEL_PATH = "best_cnn_model.pth"
CLASS_NAMES = ['blues', 'classical', 'country', 'disco', 'hiphop',
               'jazz', 'metal', 'pop', 'reggae', 'rock']

model = CNNGenreClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
def preprocess_wav(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # Use longer segment, up to 30s
    if len(y) < sr * 30:
        y = np.pad(y, (0, sr * 30 - len(y)))  # pad if shorter
    else:
        y = y[:sr * 30]  # truncate

    # Generate log-scaled mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 255] and resize
    S_norm = (S_DB - S_DB.min()) / (S_DB.max() - S_DB.min())
    S_img = Image.fromarray((S_norm * 255).astype(np.uint8)).convert("L")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    tensor = transform(S_img).unsqueeze(0)  # Shape: [1, 1, 128, 128]
    return tensor


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        input_tensor = preprocess_wav(temp_path)
        with torch.no_grad():
            output = model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
            genre = CLASS_NAMES[predicted]
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(temp_path)

    return {"genre": genre}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
