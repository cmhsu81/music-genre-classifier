import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np

def save_mel_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=30)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(2, 2))
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Modify these paths
source_dir = 'dataset/genres_original'
output_dir = 'spectrograms'

os.makedirs(output_dir, exist_ok=True)

for genre in os.listdir(source_dir):
    genre_path = os.path.join(source_dir, genre)
    if not os.path.isdir(genre_path):
        continue
    output_genre_dir = os.path.join(output_dir, genre)
    os.makedirs(output_genre_dir, exist_ok=True)

    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):
            input_path = os.path.join(genre_path, filename)
            output_path = os.path.join(output_genre_dir, filename.replace('.wav', '.png'))
            try:
                save_mel_spectrogram(input_path, output_path)
                print(f"Saved {output_path}")
            except Exception as e:
                print(f"Failed {input_path}: {e}")
