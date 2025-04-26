# streamlit_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache

# ---------- configuration ----------
# MODEL_PATH = "your-username/your-model"  # Option B (Hub)

LABELS = {0: "Human", 1: "Ai"}   # customise

# ---------- helpers ----------
# streamlit_app.py
MODEL_PATH = "nawafalomari/test606"   # <— your HF repo

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    tok  = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl  = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    mdl.eval()
    return tok, mdl


def predict(code: str) -> str:
    tokenizer, model = load_model()
    with torch.inference_mode():
        inputs = tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = model(**inputs).logits
        idx = int(logits.argmax(-1))
    return LABELS[idx]

# ---------- UI ----------
st.title("CodeBERT commit classifier")
code_snippet = st.text_area("Paste a code diff or snippet:", height=200)

if st.button("Classify") and code_snippet.strip():
    label = predict(code_snippet)
    st.success(f"Predicted label → **{label}**")
