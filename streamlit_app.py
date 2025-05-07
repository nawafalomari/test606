# streamlit_app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from functools import lru_cache  # still unused but harmless

# ---------- configuration ----------
MODEL_PATH = "nawafalomari/ICS606"      # 💾 your HF repo
LABELS     = {0: "Human 🙋🏻‍♂️", 1: "AI 🤖"}       # 🏷️ customise if you add more classes

# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    mdl.eval()                                      # we’re only doing inference
    return tok, mdl


def predict(code: str) -> dict[str, float]:
    """
    Returns a dict {label: prob} with probabilities in the range [0, 1].
    """
    tokenizer, model = load_model()

    with torch.inference_mode():
        inputs = tokenizer(
            code,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        logits = model(**inputs).logits             # shape [1, num_labels]
        probs  = torch.softmax(logits, dim=-1)[0]   # shape [num_labels]

    # map to readable labels and convert to standard Python floats
    return {LABELS[i]: float(probs[i]) for i in range(len(probs))}

# ---------- UI ----------
st.title("GPTPointerException 🤖")
code_snippet = st.text_area("Paste a code diff or snippet:", height=200)

if st.button("Classify") and code_snippet.strip():
    probs = predict(code_snippet)

    # 1️⃣ show the arg‑max label + confidence
    best_label = max(probs, key=probs.get)
    best_pct   = probs[best_label] * 100
    st.success(f"**{best_label}** · {best_pct:.2f}% confidence")

    # 2️⃣ (optional) show a full breakdown
    st.subheader("Class probabilities")
    for lbl, p in probs.items():
        st.write(f"- {lbl}: {p*100:.2f}%")
        # or:  st.metric(lbl, f"{p*100:.2f}%")
