from flask import Flask, request, jsonify, send_file
import torch
from flask_cors import CORS
import torchaudio
import torchaudio.transforms as T
import pickle
import os
import tempfile
import traceback
from pydub import AudioSegment
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# Load trained model
model_path = "pet_behavior_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

with open(model_path, "rb") as f:
    model = pickle.load(f)

model.eval()

behavior_precautions = {
    "Barking": "Your pet is barking. This may indicate excitement, fear, or alertness. Ensure they have enough exercise and check for any disturbances.",
    "Whining": "Your pet is whining. They may be anxious, in pain, or seeking attention. Comfort them and check if they need food, water, or medical care.",
    "Meowing": "Your cat is meowing. This could mean they are hungry, lonely, or seeking attention. Ensure they have food, water, and interaction.",
    "Purring": "Your cat is purring. This usually means they are happy and relaxed. However, excessive purring might indicate stress or pain. Observe their behavior closely."
}

def convert_webm_to_wav(webm_audio):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as webm_file:
            webm_file.write(webm_audio)
            webm_path = webm_file.name
        
        wav_path = webm_path.replace(".webm", ".wav")
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio.export(wav_path, format="wav", codec="pcm_s16le")
        os.remove(webm_path)
        return wav_path
    except Exception as e:
        app.logger.error(f"[ERROR] Audio conversion failed: {str(e)}")
        raise

def extract_features(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=40)
        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1)
        return mfcc.unsqueeze(0)
    except Exception as e:
        app.logger.error(f"[ERROR] Feature extraction failed: {str(e)}")
        raise

def generate_response_audio(response_text, response_audio_path):
    try:
        tts = gTTS(text=response_text, lang="en")
        tts.save(response_audio_path)
    except Exception as e:
        app.logger.error(f"[ERROR] TTS generation failed: {str(e)}")
        raise

@app.route("/predict", methods=["POST"])
def predict_behavior():
    try:
        app.logger.debug("[DEBUG] Received request for prediction.")
        
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        app.logger.debug("[DEBUG] Saving uploaded file...")
        audio_data = file.read()
        wav_path = convert_webm_to_wav(audio_data)

        app.logger.debug(f"[DEBUG] Extracting features from {wav_path}...")
        features = extract_features(wav_path)
        os.remove(wav_path)

        app.logger.debug(f"[DEBUG] Running model prediction with input shape: {features.shape}")
        with torch.no_grad():
            output = model(features)
            prediction = torch.argmax(output, dim=1).item()

        if prediction >= len(behavior_precautions):
            raise ValueError("Invalid prediction index.")

        predicted_behavior = list(behavior_precautions.keys())[prediction]
        precaution = behavior_precautions[predicted_behavior]

        app.logger.debug(f"[DEBUG] Predicted behavior: {predicted_behavior}")

        response_audio_path = "response.mp3"
        generate_response_audio(precaution, response_audio_path)

        return jsonify({
            "behavior": predicted_behavior,
            "precaution": precaution,
            "audio_url": "http://127.0.0.1:5000/get_audio"
        })
    except Exception as e:
        error_message = f"Internal Server Error: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(f"[ERROR] {error_message}")
        return jsonify({"error": error_message}), 500

@app.route("/get_audio", methods=["GET"])
def get_audio():
    response_audio_path = "response.mp3"
    return send_file(response_audio_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)
