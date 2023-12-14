from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa

app = Flask(__name__)

# Charger le modèle VGG
vgg_service_url = 'http://vgg_service:7000'
model_path = '/app/vgg_service'  
model = tf.saved_model.load(model_path)

classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

@app.route('/predict_vgg', methods=['POST'])
def predict_vgg():
    try:
        # Assurez-vous que 'audio' est la clé correcte dans la requête multipart
        file = request.files['audio']
        waveform, sr = librosa.load(file, sr=16000)

        # ... Traitement des caractéristiques audio ...

        # Créer le tenseur d'entrée
        inp = tf.constant(np.array([waveform]), dtype='float32')

        # Faire la prédiction
        class_scores = model(inp)[0].numpy()
        predicted_class = classes[class_scores.argmax()]

        # Retourner le résultat
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)
