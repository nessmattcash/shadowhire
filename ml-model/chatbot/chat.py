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
print(len(data_list))




#3clean up the data
# Colab cell
import pandas as pd, re, unicodedata
from sklearn.model_selection import train_test_split

df = pd.DataFrame(data_list)
# simple cleaning function
import json
import pandas as pd
import re
import unicodedata
from sklearn.model_selection import train_test_split

# Load dataset
with open("chatbot.json", "r", encoding="utf-8") as f:
    raw = json.load(f)
data_list = raw["dataset"] if isinstance(raw, dict) and "dataset" in raw else raw
print("Total items:", len(data_list))  # Expected: 877

df = pd.DataFrame(data_list)

# Enhanced clean_text function
def clean_text(text, lowercase=True):
    if text is None or pd.isna(text):
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"[\x00-\x1f\x7f]+", " ", t)  # Remove control characters
    t = re.sub(r"\s+", " ", t).strip()  # Normalize spaces
    t = re.sub(r"(.)\1{3,}", r"\1", t)  # Remove excessive repeats
    t = re.sub(r"[\/]{2,}", "", t)  # Remove excessive slashes
    return t.lower() if lowercase else t

# Clean questions and answers
df["question_clean"] = df["question"].apply(lambda t: clean_text(t, True))
df["answer_clean"] = df["answer"].apply(lambda t: clean_text(t, False))

# Remove invalid or duplicate entries
df = df[df["question_clean"].str.len() > 5]  # Min question length
df = df[df["answer_clean"].str.len() > 10]  # Min answer length
df = df.drop_duplicates(subset=["question_clean", "answer_clean"])
print("Cleaned dataset size:", len(df))  # Expected: ~800-850

# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
print("Train / Val / Test:", len(train_df), len(val_df), len(test_df))  # Expected: ~560, ~120, ~120

# Save splits
train_df.to_json("train.json", orient="records", force_ascii=False)
val_df.to_json("val.json", orient="records", force_ascii=False)
test_df.to_json("test.json", orient="records", force_ascii=False)
#new step augment dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        temperature=0.7,  # Lower for coherence
        repetition_penalty=1.5  # Prevent repetitions
    )
    paras = [tokenizer_para.decode(o, skip_special_tokens=True) for o in outputs]
    # Stricter filtering
    filtered = [
        p for p in paras
        if len(p) > 10 and
        len(p.split()) > 3 and
        not re.search(r'(.)\1{3,}', p) and
        p.lower() != text.lower()
    ]
    return filtered or paras[:1]  # Fallback to one paraphrase

# Augment train_df
augmented_qas = []
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # For similarity filtering
for i, row in train_df.iterrows():
    paras = paraphrase(row["question_clean"], num_return=2)
    q_emb = embedder.encode([row["question_clean"]], convert_to_numpy=True)
    for p in paras:
        p_emb = embedder.encode([p], convert_to_numpy=True)
        similarity = cosine_similarity(q_emb, p_emb)[0][0]
        if similarity < 0.9:  # Ensure diversity
            augmented_qas.append({
                "question": p,
                "answer": row["answer_clean"],
                "question_clean": clean_text(p, True),
                "answer_clean": row["answer_clean"]
            })

df_aug = pd.DataFrame(augmented_qas)
train_df = pd.concat([train_df, df_aug], ignore_index=True)
train_df = train_df.drop_duplicates(subset=["question_clean", "answer_clean"])
print("Augmented train size:", len(train_df))  # Expected: ~560 + ~1,120 = ~1,680

# Save augmented train set
train_df.to_json("train_augmented.json", orient="records", force_ascii=False)




#4 vector embedding
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Better for English
train_questions = train_df["question_clean"].tolist()
train_answers = train_df["answer_clean"].tolist()
train_embs = embedder.encode(train_questions, convert_to_numpy=True, show_progress_bar=True)

dim = train_embs.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(train_embs)
print("FAISS index built with", index.ntotal, "vectors.")  # Expected ~1,680

#5 retrieve similar questions

def make_context_for_train(df_inputs, embedder, index, all_answers, top_k=5):
    new_inputs, new_targets = [], []
    for i, row in df_inputs.iterrows():
        q_emb = embedder.encode([row["question_clean"]], convert_to_numpy=True)
        distances, indices = index.search(q_emb, top_k + 1)
        retrieved = []
        for idx in indices[0]:
            if idx != i and idx < len(all_answers):
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
            gen_outputs = model.generate(
                input_ids=input_ids,
                max_length=200,
                num_beams=6,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
            for pred, ref in zip(gen_outputs, labels):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip()
                ref_text = tokenizer.decode(ref[ref != -100], skip_special_tokens=True).strip()
                if pred_text and ref_text and len(pred_text.split()) > 3 and len(ref_text.split()) > 3:
                    scores = scorer.score(ref_text, pred_text)
                    rouge_scores.append(scores['rougeL'].fmeasure)
                    if len(rouge_scores) <= 5:
                        print(f"Pred: {pred_text} | Ref: {ref_text} | ROUGE-L: {scores['rougeL'].fmeasure}")
                else:
                    rouge_scores.append(0.0)
    return total_loss / len(loader), sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

epochs = 10
best_val_loss = float("inf")
patience = 5
patience_counter = 0
train_losses, val_losses, val_rouges = [], [], []

for epoch in range(1, epochs + 1):
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

#9 french english 
# Colab cell
from transformers import pipeline
# FR -> EN
fr2en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
# EN -> FR
en2fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

#10 inference 
# Colab cell
def rag_generate(user_text, embedder, index, train_answers, model, tokenizer, device, top_k=5, max_length=200):
    try:
        lang = detect(user_text)
    except:
        lang = "en"
    query_en = user_text
    if lang.startswith("fr"):
        query_en = fr2en(user_text)[0]['translation_text']
    q_emb = embedder.encode([query_en], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    contexts = [train_answers[idx] for idx in I[0] if idx < len(train_answers)]
    context = " ".join(contexts)
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
        answer_fr = en2fr(answer_en)[0]['translation_text']
        return answer_fr
    return answer_en


# Example usage for multiple questions
# Combine old + new questions
quest = [
    # OLD QUESTIONS
    "Hello", "Hi", "What is Shadowhire?", "Who are you?", "How do I sign up?",
    "Is Shadowhire free?", "What file formats can I upload?", "How does job matching work?",
    "What tech stack does Shadowhire use?", "How to reset password?",
    "Can recruiters see my resume?", "How to delete account?", "What's the development roadmap?",
    "How accurate is the resume score?", "Where is Shadowhire hosted?",
    "How to contact support?", "Is there a mobile app?", "What's the difference between candidate/recruiter views?",
    "How to report a bug?", "Is my data secure?", "Quel est le format maximum d'un CV ?",
    "Comment modifier mon profil ?", "Pourquoi ne puis-je pas télécharger mon CV ?",
    "Comment fonctionnent les recommandations d'emploi ?", "Puis-je postuler directement ?",
    "Qu'est-ce que le 'Resume Score' ?", "Comment améliorer mon score de CV ?", "Qu'est-ce que le TF-IDF ?",
    "À quelle fréquence les offres d'emploi sont-elles mises à jour ?", "Puis-je sauvegarder des offres d'emploi ?",

    # NEW DATASET QUESTIONS
    "What's the max resume size?", "How to edit profile?", "Why can't I upload my resume?",
    "How do job recommendations work?", "Can I apply directly?", "What's the 'Resume Score'?",
    "How to improve my resume score?", "What's TF-IDF?", "How often are jobs updated?",
    "Can I save jobs?", "How to unsubscribe from emails?", "What's the privacy policy?",
    "Can I use without account?", "How to change email?", "Does Shadowhire support dark mode?",
    "What languages supported?", "How to report a job posting?", "Can I delete my resume?",
    "What industries covered?", "How to log out?", "Why is my match score low?",
    "Can I upload multiple resumes?", "How do I know if recruiter viewed me?", "Average response time?",
    "How to withdraw application?", "Can I edit resume after uploading?", "What's the 'AI Feedback' panel?",
    "How to search jobs?", "Can I message recruiters?", "What's 'Skills Gap'?",
    "What is Sofrecom?", "Tell me about InstaDeep", "Capgemini Tunisia info",
    "Does Actia hire in Tunisia?", "Picosoft overview", "What is Defency?",
    "Tell me about Sphra", "Does Shadowhire have jobs at STMicroelectronics?"
]

# Run RAG on each question
answers = [rag_generate(q, embedder, index, train_answers, model, tokenizer, device) for q in quest]

# Print results
for q, a in zip(quest, answers):
    print(f"Q: {q}\nA: {a}\n{'-'*80}")

# result 