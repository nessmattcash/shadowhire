import logging
import os
import json
import numpy as np
from utils import load_ner_model, extract_text_from_pdf, parse_cv

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

# Custom JSON serializer for numpy types
def numpy_to_python(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def test_model():
    print("Starting test_model...")
    model_path = "./fine_tuned_resume_ner/fine_tuned_resume_ner"
    print(f"Loading model from {model_path}")
    try:
        nlp = load_ner_model(model_path)
        logging.info("Model loaded successfully!")
        print("Model loaded successfully!")

        # Test on sample text
        sample_text = "John Doe is a Python developer with skills in Java and AWS. Email: john@example.com Projects: Resume Parser Experience "
        print("Running NER on sample text...")
        ner_results = nlp(sample_text)
        print("NER Results on sample text:", json.dumps(ner_results, indent=2, default=numpy_to_python))

        # Test on PDF
        pdf_path =  [
            
           
            "./cvs_test/aziz.pdf",
            "./cvs_test/walid.pdf",
            "./cvs_test/sana.pdf",
            "./cvs_test/tam.pdf",
            "./cvs_test/Aziz-Mehdi.pdf",
            "./cvs_test/yasmine.pdf",
            "./cvs_test/frosty.pdf",
            
        ]
        for pdf in pdf_path:
            print(f"Processing PDF: {pdf}")
            if os.path.exists(pdf):
                text = extract_text_from_pdf(pdf)
                print("Extracted text from PDF (first 100 chars):", text[:100])
                parsed = parse_cv(text, pdf, nlp=nlp)
                print("Parsed CV:", json.dumps(parsed, indent=2, default=numpy_to_python))
                output_file = f"parsed2_{os.path.basename(pdf).replace('.pdf', '.json')}"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2, default=numpy_to_python)    
            else:
                logging.warning(f"No sample PDF found at {pdf}")
                print(f"No sample PDF found at {pdf}")
               
    except Exception as e:
        logging.error(f"Error in test_model: {str(e)}")
        print(f"Error in test_model: {str(e)}")
        raise

if __name__ == "__main__":
    print("Running main.py...")
    test_model()