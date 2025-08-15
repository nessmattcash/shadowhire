from flask import Flask, request, jsonify
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
import re, unicodedata
import sentencepiece

# --- Helper: clean text ---
def clean_text(text):
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()

# --- Load dataset ---
with open("chatbot.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)
# If wrapped in {"dataset": [...]}
if isinstance(raw_data, dict) and "dataset" in raw_data:
    raw_data = raw_data["dataset"]

questions = [clean_text(item['question']) for item in raw_data]
answers = [item['answer'] for item in raw_data]

# --- Load embeddings and build FAISS index ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embs = embedder.encode(questions, convert_to_numpy=True)
dim = embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embs)

# --- Load T5 model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5_rag_tokenizer")
model = T5ForConditionalGeneration.from_pretrained("t5_rag_model").to(device)

# --- Translation pipelines ---
fr2en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
en2fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

# --- Flask app ---
app = Flask(__name__)

def rag_generate(user_text, top_k=5, max_length=200):
    try:
        lang = detect(user_text)
    except:
        lang = "en"
    query_en = user_text
    if lang.startswith("fr"):
        query_en = fr2en(user_text)[0]['translation_text']
    q_emb = embedder.encode([clean_text(query_en)], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    context = " ".join([answers[idx] for idx in I[0] if idx < len(answers)])
    input_text = f"question: {query_en} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=6,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,
        length_penalty=1.0,
        early_stopping=True
    )
    answer_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if lang.startswith("fr"):
        return en2fr(answer_en)[0]['translation_text']
    return answer_en

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_text = data.get("message", "")
    if not user_text:
        return jsonify({"error": "No message provided"}), 400
    answer = rag_generate(user_text)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
