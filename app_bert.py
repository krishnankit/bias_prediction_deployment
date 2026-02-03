import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

idx2bias = {
    0: "Right",
    1: "Right-Center",
    2: "Neutral",
    3: "Left-Center",
    4: "Left"
}

MAX_LEN = 128   # MUST match training
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "bert_model.pth")

class BERTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BERTClassifier(num_classes=5).to(device)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

def encode(text: str):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

st.title("📰 News Bias Prediction")

article_title = st.text_input("Enter news title:")
article_content = st.text_area("Enter news article:", height=250)

if st.button("Predict"):
    if not article_title.strip() or not article_content.strip():
        st.warning("Please enter some text.")
    else:
        inputs = encode(article_title + ' ' + article_content)

        with torch.no_grad():
            logits = model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )

            probs = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(probs).item()
            conf = probs[pred].item()

        st.success(f"This new articles is politically **{idx2bias[pred]}** biased")
        st.write(f"Confidence: **{conf*100:.2f}%**")
