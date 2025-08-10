# Put this in a Colab cell and run it
# !pip install -q sentence-transformers==2.2.2 transformers datasets scikit-learn faiss-cpu langdetect nltk huggingface_hub==0.13.4


#2 Load and inspect chatbot.json
#import json, os
fn = "chatbot.json"
assert os.path.exists(fn), f"{fn} not found in working directory. Upload it via Colab UI (Files -> Upload)."
with open(fn, "r", encoding="utf-8") as f:
    raw = json.load(f)

# The dataset you showed uses a top-level key "dataset" -> a list of dicts
if isinstance(raw, dict) and "dataset" in raw:
    data_list = raw["dataset"]
else:
    # If the file is already a list, handle that
    if isinstance(raw, list):
        data_list = raw
    else:
        raise ValueError("Unexpected JSON structure. It should be either {'dataset':[...]} or a list.")

print("Total items:", len(data_list))
print("Sample item 0:", data_list[0])



#3 Convert to DataFrame
import pandas as pd
df = pd.DataFrame(data_list)

# Ensure columns exist
assert "question" in df.columns and "answer" in df.columns, "Each item must have 'question' and 'answer'."

# Basic overview
print("Columns:", df.columns.tolist())
print("Number of rows:", len(df))
display(df.head(8))

# Quick stats
df["q_len"] = df["question"].astype(str).apply(lambda s: len(s.split()))
df["a_len"] = df["answer"].astype(str).apply(lambda s: len(s.split()))
print("Question length — mean / min / max:", df["q_len"].mean(), df["q_len"].min(), df["q_len"].max())
print("Answer length  — mean / min / max:", df["a_len"].mean(), df["a_len"].min(), df["a_len"].max())

# Duplicates
dups = df[df.duplicated(subset=["question"], keep=False)]
print("Number of duplicated questions:", len(dups))
if len(dups)>0:
    display(dups)


#4 Language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # deterministic

def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unknown"

df["q_lang_guess"] = df["question"].astype(str).apply(safe_detect)
print(df["q_lang_guess"].value_counts())
display(df[["question","q_lang_guess"]].head(10))

#5 Text cleaning
import re, unicodedata

def clean_text(text, lowercase=True):
    if text is None:
        return ""
    # Normalize unicode
    text = unicodedata.normalize("NFC", str(text))
    # Remove control chars
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()
    return text

# Apply cleaning
df["question_clean"] = df["question"].apply(lambda t: clean_text(t, lowercase=True))
df["answer_clean"]   = df["answer"].apply(lambda t: clean_text(t, lowercase=False))  # keep answer case to preserve readability

display(df[["question","question_clean","answer","answer_clean"]].head(8))


#6 divide into train/val/test
from sklearn.model_selection import train_test_split

# Split dataset into 80% train, 10% val, 10% test
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# Save to disk (optional but good practice)
train_df.to_json("train.json", orient="records", force_ascii=False)
val_df.to_json("val.json", orient="records", force_ascii=False)
test_df.to_json("test.json", orient="records", force_ascii=False)



#7 Vector search with faiss
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load pretrained multilingual model (handles English & French)
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(model_name)

# Encode all training questions to vectors
train_questions = train_df["question_clean"].tolist()
train_embeddings = embedder.encode(train_questions, convert_to_numpy=True)

print("Encoded", len(train_embeddings), "questions into embeddings.")

# Build FAISS index (vector search index)
dimension = train_embeddings.shape[1]  # should be 384 for this model
index = faiss.IndexFlatL2(dimension)
index.add(train_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

#8 Define a function to retrieve similar questions
def retrieve_similar_questions(query, top_k=3):
    # Encode user query
    query_emb = embedder.encode([query], convert_to_numpy=True)
    # Search top_k nearest neighbors
    distances, indices = index.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        q = train_questions[idx]
        a = train_df.iloc[idx]["answer_clean"]
        results.append({"question": q, "answer": a, "distance": dist})
    return results

# Try it:
query = "How do I sign up?"
results = retrieve_similar_questions(query)
for i, res in enumerate(results):
    print(f"Top {i+1}: Q: {res['question']}")
    print(f"        A: {res['answer']}")
    print(f"        Distance: {res['distance']:.4f}\n")

#9 create a custom dataset class for training
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.questions = df["question_clean"].tolist()
        self.answers = df["answer_clean"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        input_encoding = self.tokenizer(
            question,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            answer,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding for loss

        return {
            "input_ids": input_encoding.input_ids.squeeze(),
            "attention_mask": input_encoding.attention_mask.squeeze(),
            "labels": labels,
        }



# 10 Load T5 tokenizer and model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)




#11 Create DataLoader for training and validation
train_dataset = QADataset(train_df, tokenizer)
val_dataset = QADataset(val_df, tokenizer)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)



#12 Training loop
def train_epoch(model, dataloader, optimizer, device):
    model.train()  # set model to training mode
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()  # compute gradients
        optimizer.step()  # update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



#13 Initialize optimizer

def eval_epoch(model, dataloader, device):
    model.eval()  # eval mode disables dropout, etc.
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



#14 Training loop
epochs = 3
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = eval_epoch(model, val_loader, device)
    print(f"Epoch {epoch+1} / {epochs} — Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")





#15 fuction to generate answers test
def generate_answer(question, model, tokenizer, device, max_length=64):
    model.eval()
    input_text = "question: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=64).to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if len(answer.strip()) < 5:
        return "Sorry, I can only answer questions related to Shadowhire."
    return answer
#testing 
question = "How do I sign up?"
answer = generate_answer(question, model, tokenizer, device)
print("Q:", question)
print("A:", answer)


#16 generating questions using rag 
#option1
def rag_answer(user_question, 
               faiss_index, 
               embedder, 
               data_answers, 
               model, 
               tokenizer, 
               device, 
               top_k=1,
               max_length=64):
    # Step 1: Retrieve nearest question(s)
    q_emb = embedder.encode([user_question.lower()], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, top_k)
    
    # Get retrieved context/answers
    retrieved_texts = [data_answers[idx] for idx in indices[0]]
    
    # Create a combined prompt for T5, e.g. prefix + retrieved answer
    context = " ".join(retrieved_texts)
    input_text = f"question: {user_question} context: {context}"
    
    # Step 2: Generate answer with T5 using retrieved context
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=128).to(device)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if len(answer.strip()) < 5:
        return "Sorry, I can only answer questions related to Shadowhire."
    return answer


#testing
answers = train_df["answer_clean"].tolist()  # or val_df/test_df/df depending on what you want

user_q = "How do I sign up?"
answer = rag_answer(
    user_question=user_q,
    faiss_index=index,
    embedder=embedder,
    data_answers=answers,
    model=model,
    tokenizer=tokenizer,
    device=device
)
print(answer)



#option2
def rag_answer(user_question, faiss_index, embedder, data_answers, top_k=1):
    # Encode user question
    q_emb = embedder.encode([user_question], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, top_k)
    # Return the stored answer of the top match
    idx = indices[0][0]
    answer = data_answers[idx]
    return answer




#testing
answers = train_df["answer_clean"].tolist()  # or val_df/test_df/df depending on what you want

user_q = "How do I sign up?"
answer = rag_answer(
    user_question=user_q,
    faiss_index=index,
    embedder=embedder,
    data_answers=answers
)
print(answer)

