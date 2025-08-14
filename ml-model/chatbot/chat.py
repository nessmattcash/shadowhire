# Colab cell
#!pip install -q sentence-transformers==2.2.2 transformers datasets scikit-learn faiss-cpu langdetect nltk huggingface_hub==0.13.4
#!pip install -U sentence-transformers

#2 load the dataset
import json, os
fn = "chatbot.json"
assert os.path.exists(fn), f"{fn} not found. Upload via Colab Files."
with open(fn, "r", encoding="utf-8") as f:
    raw = json.load(f)
data_list = raw["dataset"] if isinstance(raw, dict) and "dataset" in raw else raw
print("Total items:", len(data_list))
# quick sample
print(data_list[0])




#3clean up the data
# Colab cell
import pandas as pd, re, unicodedata
from sklearn.model_selection import train_test_split

df = pd.DataFrame(data_list)
# simple cleaning function
def clean_text(text, lowercase=True):
    if text is None: return ""
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower() if lowercase else t

df["question_clean"] = df["question"].apply(lambda t: clean_text(t, True))
df["answer_clean"]   = df["answer"].apply(lambda t: clean_text(t, False))

#new step augment dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

paraphraser_model = "ramsrigouthamg/t5_paraphraser"
tokenizer_para = AutoTokenizer.from_pretrained(paraphraser_model)
model_para = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_model)

def paraphrase(text, num_return=2):
    input_text = f"paraphrase: {text}"
    encoding = tokenizer_para.encode_plus(input_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model_para.generate(
        **encoding,
        max_length=256,
        num_beams=5,
        num_return_sequences=num_return,
        temperature=1.0  # Lower temperature for better quality
    )
    paras = [tokenizer_para.decode(o, skip_special_tokens=True) for o in outputs]
    # Filter bad paraphrases (e.g., repetitive or too short)
    filtered = [p for p in paras if len(p) > 10 and not re.search(r'(.)\1{3,}', p)]  # No 3+ repeats
    return filtered or paras  # Fallback to original paras if all filtered

# Generate augmented QA pairs
augmented_qas = []
for i, row in df.iterrows():
    paras = paraphrase(row["question_clean"], num_return=2)
    for q in paras:
        augmented_qas.append({"question_clean": q, "answer_clean": row["answer_clean"]})

df_aug = pd.DataFrame(augmented_qas)
print("Augmented dataset size:", len(df_aug))
df.to_json("UPDATEDDATASET.json", orient="records", lines=True)



#split the dataset

from sklearn.model_selection import train_test_split

df["question_clean"] = df["question"].apply(lambda t: clean_text(t, True))
df["answer_clean"] = df["answer"].apply(lambda t: clean_text(t, False))

train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("Train / Val / Test:", len(train_df), len(val_df), len(test_df))
train_df.to_json("train.json", orient="records", force_ascii=False)
val_df.to_json("val.json", orient="records", force_ascii=False)
test_df.to_json("test.json", orient="records", force_ascii=False)



#4 vector embedding
# Colab cell
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Better for English
train_questions = train_df["question_clean"].tolist()
train_answers = train_df["answer_clean"].tolist()
train_embs = embedder.encode(train_questions, convert_to_numpy=True, show_progress_bar=True)

dim = train_embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(train_embs)
print("FAISS index built with", index.ntotal, "vectors.")  # Expected ~1,842

#5 retrieve similar questions

def make_context_for_train(df_inputs, embedder, index, all_answers, top_k=5):
    new_inputs = []
    new_targets = []
    
    for i, row in df_inputs.iterrows():
        q_emb = embedder.encode([row["question_clean"]], convert_to_numpy=True)
        distances, indices = index.search(q_emb, top_k + 1)  # +1 to allow skipping same item
        
        retrieved = []
        for idx in indices[0]:
            if idx < len(all_answers):  # ✅ prevent out-of-range error
                retrieved.append(all_answers[idx])
            if len(retrieved) >= top_k:
                break
        
        context = " ".join(retrieved)
        new_inputs.append(f"question: {row['question_clean']} context: {context}")
        new_targets.append(row["answer_clean"])
    
    return new_inputs, new_targets
train_inputs, train_targets = make_context_for_train(train_df, embedder, index, train_answers, top_k=5)
val_inputs, val_targets = make_context_for_train(val_df, embedder, index, train_answers, top_k=5)    

#6 dataloader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader

model_name = "t5-small"   # change to larger later (e.g., t5-base, flan-t5-small) if you have GPU
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.config.dropout_rate = 0.5       # default ~0.1
model.config.attention_dropout_rate = 0.5


class RAGDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_len_input=256, max_len_target=200):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        inp = self.inputs[idx]
        tgt = self.targets[idx]
        enc_inp = self.tokenizer(inp, max_length=self.max_len_input, truncation=True, padding="max_length", return_tensors="pt")
        enc_tgt = self.tokenizer(tgt, max_length=self.max_len_target, truncation=True, padding="max_length", return_tensors="pt")
        labels = enc_tgt.input_ids.squeeze()
        labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc_inp.input_ids.squeeze(),
            "attention_mask": enc_inp.attention_mask.squeeze(),
            "labels": labels
        }
train_inputs, train_targets = make_context_for_train(train_df, embedder, index, train_answers, top_k=3)
train_dataset = RAGDataset(train_inputs, train_targets, tokenizer)
val_inputs, val_targets = make_context_for_train(val_df, embedder, index, train_answers, top_k=3)  # use val similarly
val_dataset = RAGDataset(val_inputs, val_targets, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 7 check the device
# Colab cell
from torch.optim import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)


# 9 trainning loop
# Colab cell
#!pip install rouge_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rouge_score import rouge_scorer
import time

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device, tokenizer):
    model.eval()
    total_loss = 0.0
    rouge_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            gen_outputs = model.generate(input_ids=input_ids, max_length=200, num_beams=6, do_sample=True, top_p=0.9, length_penalty=1.2)
            for pred, ref in zip(gen_outputs, labels):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()
                ref_text = tokenizer.decode(ref[ref != -100], skip_special_tokens=True).strip()
                if pred_text and ref_text:
                    scores = scorer.score(ref_text, pred_text)
                    rouge_scores.append(scores['rougeL'].fmeasure)
                else:
                    rouge_scores.append(0.0)
                if len(rouge_scores) <= 5:
                    print(f"Pred: {pred_text} | Ref: {ref_text} | ROUGE-L: {scores['rougeL'].fmeasure}")
    return total_loss / len(loader), sum(rouge_scores) / len(rouge_scores)

epochs = 10
best_val_loss = float("inf")
patience = 5
patience_counter = 0
train_losses, val_losses, val_rouges = [], [], []

for epoch in range(1, epochs+1):
    t0 = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_rouge = eval_epoch(model, val_loader, device, tokenizer)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_rouges.append(val_rouge)
    print(f"Epoch {epoch}/{epochs} — train_loss: {train_loss:.4f} val_loss: {val_loss:.4f} val_rouge: {val_rouge:.4f} time: {time.time()-t0:.0f}s")
    scheduler.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save_pretrained("t5_rag_model")
        tokenizer.save_pretrained("t5_rag_tokenizer")
        print("✅ Model saved.")
    else:
        patience_counter += 1
        print(f"Patience counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("⏹ Early stopping triggered.")
            break

# result Epoch 1/6 — train_loss: 4.7961 val_loss: 5.0115 time: 213s,Epoch 2/6 — train_loss: 4.5286 val_loss: 4.6988 time: 231s
# result Epoch 3/6 — train_loss: 4.4076 val_loss: 4.5418 time: 208s
# result Epoch 4/6 — train_loss: 4.3078 val_loss: 4.4712 time: 212s
# result Epoch 5/6 — train_loss: 4.2474 val_loss: 4.4123 time: 210s
# result Epoch 6/6 — train_loss: 4.2114 val_loss: 4.3533 time: 209s



#9 french english 
# Colab cell
from transformers import pipeline
# FR -> EN
fr2en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# EN -> FR
en2fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

#10 inference 
# Colab cell
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

def rag_generate(user_text, embedder, index, train_answers, model, tokenizer, device, top_k=3, max_length=180):
    # 1) detect language
    try:
        lang = detect(user_text)
    except:
        lang = "en"
    # 2) translate to English if French
    query_en = user_text
    if lang.startswith("fr"):
        # keep short; pipeline may need the raw string
        query_en = fr2en(user_text)[0]['translation_text']
    # 3) retrieve top_k contexts
    q_emb = embedder.encode([query_en], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    contexts = [train_answers[idx] for idx in I[0]]
    context = " ".join(contexts)
    # 4) build prompt for T5 (instruction style)
    input_text = f"instruction: You are a helpful bilingual assistant. Use the context to answer the question in English. Context: {context} Question: {query_en}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    # 5) generate with tuned decoding for long, professional answers
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_beams=6,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    answer_en = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 6) translate back to French if needed
    if lang.startswith("fr"):
        answer_fr = en2fr(answer_en)[0]['translation_text']
        return answer_fr
    return answer_en
