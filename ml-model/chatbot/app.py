from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from sentence_transformers import SentenceTransformer
import faiss
from langdetect import detect
import re
import unicodedata
import logging
import sentencepiece


# Setup logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper: clean text ---
def clean_text(text):
    try:
        t = unicodedata.normalize("NFC", str(text))
        t = re.sub(r"[\x00-\x1f\x7f]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t.lower()
    except Exception as e:
        logging.error(f"Error cleaning text: {str(e)}")
        return text.lower()

# --- Load dataset ---
try:
    with open("chatbot.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    # If wrapped in {"dataset": [...]}
    if isinstance(raw_data, dict) and "dataset" in raw_data:
        raw_data = raw_data["dataset"]
    questions = [clean_text(item['question']) for item in raw_data]
    answers = [item['answer'] for item in raw_data]
except Exception as e:
    logging.error(f"Error loading dataset: {str(e)}")
    raise

# --- Load embeddings and build FAISS index ---
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embs = embedder.encode(questions, convert_to_numpy=True, show_progress_bar=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
except Exception as e:
    logging.error(f"Error building FAISS index: {str(e)}")
    raise

# --- Load T5 model ---
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5_rag_tokenizer")
    model = T5ForConditionalGeneration.from_pretrained("t5_rag_model").to(device)
except Exception as e:
    logging.error(f"Error loading T5 model: {str(e)}")
    raise

# --- Translation pipelines ---
try:
    fr2en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    en2fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
except Exception as e:
    logging.error(f"Error loading translation pipelines: {str(e)}")
    raise

# --- Flask app ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:4200"}})

# --- Chatbot logic ---
def rag_generate(user_text, top_k=5, max_length=200):
    try:
        lang = detect(user_text)
    except Exception as e:
        logging.warning(f"Language detection failed: {str(e)}. Defaulting to English.")
        lang = "en"
    
    query_en = user_text
    if lang.startswith("fr"):
        query_en = fr2en(user_text)[0]['translation_text']
    
    try:
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
    except Exception as e:
        logging.error(f"Error in rag_generate: {str(e)}")
        return "Sorry, an error occurred while generating the response."

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            logging.warning("Missing 'question' in JSON payload")
            return jsonify({"error": "Missing 'question' in JSON payload"}), 400
        user_text = data["question"]
        if not user_text:
            logging.warning("Empty question provided")
            return jsonify({"error": "Empty question provided"}), 400
        answer = rag_generate(user_text)
        logging.info(f"Question: {user_text} | Answer: {answer}")
        return jsonify({"question": user_text, "answer": answer})
    except Exception as e:
        logging.error(f"Error in /chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)