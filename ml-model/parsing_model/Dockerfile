FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm
RUN python -c "import nltk; nltk.download('punkt')"

COPY . .

# Copy your model folder
COPY fine_tuned_resume_ner ./fine_tuned_resume_ner/fine_tuned_resume_ner

EXPOSE 5005

CMD ["gunicorn", "--bind", "0.0.0.0:5005", "app:app"]