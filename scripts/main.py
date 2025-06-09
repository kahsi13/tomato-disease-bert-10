from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pickle
import requests
import os

# 🌐 Hugging Face model adı
HF_MODEL_ID = "Kahsi13/tomato-bert-10"
LABEL_ENCODER_URL = "https://huggingface.co/Kahsi13/tomato-bert-10/resolve/main/label_encoder.pkl"

# 🚀 Uygulama
app = FastAPI()

# 🧠 Model ve tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
model.eval()

# 🏷️ Label encoder'ı indir ve yükle
encoder_path = "label_encoder.pkl"
if not os.path.exists(encoder_path):
    with open(encoder_path, "wb") as f:
        f.write(requests.get(LABEL_ENCODER_URL).content)

with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

# 📥 API giriş formatı
class InputText(BaseModel):
    text: str

# 🏠 Ana endpoint
@app.get("/")
def read_root():
    return {"message": "Tomato Disease BERT API is running."}

# 🔍 Tahmin endpoint'i
@app.post("/predict")
def predict(input: InputText):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = torch.max(probs).item()
        label = label_encoder.inverse_transform([pred_id])[0]
    return {"label": label, "confidence": round(confidence, 4)}