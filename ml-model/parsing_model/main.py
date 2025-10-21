import logging
from utils import load_ner_model, process_pdfs, extract_text_from_pdf, parse_cv
import os
import json

logging.basicConfig(level=logging.INFO)

def test_model():
    model_path = "./fine_tuned_resume_ner/fine_tuned_resume_ner"
    nlp = load_ner_model(model_path)
    logging.info("Model loaded successfully!")

    # Test on sample text
    sample_text = "John Doe is a Python developer with skills in Java and AWS. Email: john@example.com"
    ner_results = nlp(sample_text)
    print("NER Results on sample text:", ner_results)

    # Test on PDF
    pdf_path = "./cvs_test/sana.pdf"  # Put your test PDF here
    if os.path.exists(pdf_path):
        text = extract_text_from_pdf(pdf_path)
        parsed = parse_cv(text, pdf_path, nlp=nlp)
        print("Parsed CV:", json.dumps(parsed, indent=2))
    else:
        logging.warning("No sample PDF found. Add one to ./samples/")

if __name__ == "__main__":
    test_model()
