from flask import Flask, request, jsonify
from utils import load_ner_model, extract_text_from_pdf, parse_cv
import logging

app = Flask(__name__)
nlp = load_ner_model("./fine_tuned_resume_ner/fine_tuned_resume_ner")

@app.route('/parse_resume', methods=['POST'])
def parse_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
        parsed = parse_cv(text, file.filename, nlp=nlp)
        return jsonify(parsed)
    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True)