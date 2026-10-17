import io
import base64
import torch
import soundfile as sf
import runpod
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

MODEL_ID = "Tanmoyshome/bn-parler-tts"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

VOICES = {
    "female": "A Bengali female speaker speaks at a moderate pace in a very clear recording.",
    "male": "A Bengali male speaker speaks at a moderate pace in a very clear recording.",
}

def handler(job):
    inp = job.get("input", {})
    text = inp.get("text", "").strip()
    voice = inp.get("voice", "female")
    if not text:
        return {"error": "text is required"}
    if len(text) > 500:
        return {"error": "text too long (max 500 characters)"}
    if voice not in VOICES:
        return {"error": "unknown voice"}

    desc = desc_tokenizer(VOICES[voice], return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").
