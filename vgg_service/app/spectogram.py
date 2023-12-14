import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram_for_file(audio_path, output_path):
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)

    # Créer le spectrogramme mel
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    # Tracer et sauvegarder le spectrogramme en tant qu'image PNG
    plt.figure(figsize=(8, 6))
    librosa.display.specshow(spec_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {os.path.basename(audio_path)}")
    plt.savefig(output_path, format='png')
    plt.close()

def create_spectrograms(dataset_path, output_images_dir):
    # Parcourir tous les sous-répertoires (genres musicaux)
    for genre in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre)

        # S'assurer que l'élément est un dossier
        if os.path.isdir(genre_path):
            # Créer un sous-répertoire pour les images de ce genre
            genre_output_dir = os.path.join(output_images_dir, genre)
            os.makedirs(genre_output_dir, exist_ok=True)

            # Parcourir tous les fichiers dans le sous-dossier
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    # Construire le chemin complet vers le fichier audio
                    audio_path = os.path.join(genre_path, file)

                    # Construire le chemin complet pour sauvegarder le spectrogramme
                    output_path = os.path.join(genre_output_dir, f"{os.path.splitext(file)[0]}_spectrogram.png")

                    # Créer le spectrogramme et le sauvegarder
                    create_spectrogram_for_file(audio_path, output_path)

# Exemple d'utilisation
dataset_path = '/Users/wiem/Desktop/Data/genres_original'
output_images_dir = '/Users/wiem/Desktop/Data/img'

create_spectrograms(dataset_path, output_images_dir)