import librosa
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Feature Extraction
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    features = np.concatenate((
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.min(mfcc, axis=1),
        np.max(mfcc, axis=1),
    ))
    return features

# 2. Dataset Preparation
def prepare_dataset(dataset_path):
    genres = os.listdir(dataset_path)
    features, labels = [], []
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    audio_path = os.path.join(genre_path, file)
                    try:
                        feature = extract_features(audio_path)
                        features.append(feature)
                        labels.append(genre)
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")
    return np.array(features), np.array(labels)

# Load data
dataset_path = 'dataset/genres_original'
X, y = prepare_dataset(dataset_path)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. Define Model
class GenreClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GenreClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# Model parameters
input_dim = X_train.shape[1]  # e.g., 13 MFCCs
hidden_dim = 128
output_dim = len(np.unique(y_encoded))  # 10 genres

# Initialize model
model = GenreClassifier(input_dim, hidden_dim, output_dim)

# 4. Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
epochs = 200

# 5. Training Loop
best_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    val_outputs = model(X_test_tensor)
    val_loss = criterion(val_outputs, y_test_tensor).item()

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pth")  # save best model
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# 6. Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_tensor.numpy(), predicted.numpy(), target_names=le.classes_))


# Save the model and label encoder
MODEL_PATH = "genre_classifier.pth"
ENCODER_PATH = "label_encoder.npy"

# Save model state dict
torch.save(model.state_dict(), MODEL_PATH)

# Save label encoder classes
np.save(ENCODER_PATH, le.classes_)

print(f"\n✅ Model saved to {MODEL_PATH}")
print(f"✅ Label encoder saved to {ENCODER_PATH}")