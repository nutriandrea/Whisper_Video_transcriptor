import os
import torch
import subprocess
from flask import Flask, request, render_template, send_from_directory
from moviepy.editor import VideoFileClip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_to_mp3(video_path):
    try:
        clip = VideoFileClip(video_path)
        mp3_path = video_path.rsplit('.', 1)[0] + ".mp3"
        clip.audio.write_audiofile(mp3_path)
        return mp3_path
    except Exception as e:
        print("Errore conversione audio:", e)
        return None

def transcribe(mp3_path):
    model_id = "openai/whisper-large-v3-turbo"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    asr = pipeline("automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device
    )

    result = asr(mp3_path)
    return result["text"]

def correct_text(text):
    prompt = f"SISTEMA GLI ERRORI ORTOGRAFICI, DI SINTASSI O PUNTEGGIATURA, SENZA CAMBIARE IL TESTO:\n{text}"
    try:
        result = subprocess.run(
            ["llama-run", "--model", "qwen", "--prompt", prompt],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Errore nella correzione: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_file = request.files["video"]
        if video_file:
            filepath = os.path.join(UPLOAD_FOLDER, video_file.filename)
            video_file.save(filepath)

            mp3_path = convert_to_mp3(filepath)
            if not mp3_path:
                return "Errore nella conversione audio."

            trascrizione = transcribe(mp3_path)
            correzione = correct_text(trascrizione)

            return render_template("result.html", originale=trascrizione, corretto=correzione)

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
