#1 install required packages
#!unzip -q /content/archive.zip -d /content/
BASE = "/content/Resume"
#!ls -R "$BASE"
#!pip install -q PyPDF2 python-docx pandas numpy transformers datasets seqeval torch torchvision torchaudio
#!pip install -q sentencepiece sacremoses langdetect evaluate
#!pip install -q scikit-learn tqdm
#!pip install docx2txt 

 



#2 import libraries
import os, re, json, math, random
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pdfminer.high_level import extract_text
import docx2txt
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
from evaluate import load  # Updated import
from collections import defaultdict



#3 load data
import pandas as pd
import os
from pathlib import Path

# Define paths
BASE_PATH = "/content/Resume"  # Adjust to your folder path
CSV_PATH = f"{BASE_PATH}/Resume.csv"

# Load CSV
df = pd.read_csv(CSV_PATH, encoding='utf-8', dtype=str, low_memory=False)
df = df.fillna('')  # Replace NaN with empty strings
print(f"Number of rows: {len(df)}")
print(df.head(2).T)

# Output columns: ID, Resume_str, Resume_html, Category




#3 cleanning data 
def clean_text(s):
    s = s.replace('\r',' ').replace('\n',' ').replace('\t',' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

df['resume_text'] = df['Resume_str'].astype(str).apply(clean_text)



#4 bigger skills list for better annotation
# ==================== CORE TECHNICAL SKILLS ====================

# --- Programming Languages ---
PROGRAMMING_LANGUAGES = [
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "php", "ruby", "swift",
    "kotlin", "go", "golang", "rust", "scala", "r", "dart", "perl", "haskell", "elixir",
    "clojure", "lua", "matlab", "objective-c", "bash", "shell", "powershell", "groovy",
    "julia", "cobol", "fortran", "lisp", "scheme", "erlang", "f#", "assembly", "vb.net", "vba"
]

# --- Web Development (Frontend) ---
WEB_FRONTEND = [
    "html", "html5", "css", "css3", "sass", "scss", "less", "bootstrap", "tailwind css",
    "react", "angular", "vue", "vue.js", "svelte", "next.js", "nuxt.js", "gatsby", "remix",
    "redux", "context api", "vuex", "rxjs", "webpack", "vite", "parcel", "babel", "npm", "yarn",
    "pnpm", "jquery", "web components", "stencil", "three.js", "d3.js", "chart.js", "jest", "cypress",
    "storybook", "material ui", "ant design", "chakra ui", "figma", "sketch", "xd", "webgl"
]

# --- Web Development (Backend) & Frameworks ---
WEB_BACKEND_FRAMEWORKS = [
    "node.js", "express.js", "nest.js", "koa", "django", "flask", "fastapi", "ruby on rails",
    "spring", "spring boot", "micronaut", "quarkus", "laravel", "symfony", "codeigniter",
    "asp.net", "asp.net core", "blazor", "phoenix", "gin", "echo", "fiber", "actix", "rocket"
]

# --- Mobile Development ---
MOBILE_DEVELOPMENT = [
    "android", "ios", "react native", "flutter", "xamarin", "ionic", "kotlin multiplatform",
    "swiftui", "jetpack compose", "objective-c", "android sdk", "xcode", "appium"
]

# --- Database Technologies ---
DATABASES = [
    "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "microsoft sql server", "sql server",
    "mongodb", "cassandra", "redis", "elasticsearch", "dynamodb", "cosmos db", "firebase firestore",
    "realm", "couchbase", "couchdb", "neo4j", "arangodb", "influxdb", "snowflake", "bigquery",
    "redshift", "table design", "database normalization", "indexing", "query optimization", "etl", "elt"
]

# --- Cloud & DevOps Platforms ---
CLOUD_PLATFORMS = [
    "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud platform",
    "oracle cloud", "oci", "ibm cloud", "digitalocean", "linode", "akamai", "cloudflare",
    "heroku", "netlify", "vercel", "firebase", "openshift", "vmware", "openstack"
]

# --- DevOps & Infrastructure Tools ---
DEVOPS_TOOLS = [
    "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef", "jenkins",
    "github actions", "gitlab ci", "circleci", "travis ci", "argo cd", "flux", "helm",
    "prometheus", "grafana", "datadog", "splunk", "new relic", "elk stack", "elastic stack",
    "istio", "linkerd", "envoy", "vagrant", "packer", "consul", "vault", "nomad",
    "nginx", "apache", "iis", "linux", "ubuntu", "debian", "centos", "red hat", "rhel", "suse",
    "bash scripting", "shell scripting", "powershell scripting"
]

# --- Specific AWS Services ---
AWS_SERVICES = [
    "ec2", "s3", "lambda", "rds", "dynamodb", "iam", "vpc", "route 53", "cloudfront",
    "sns", "sqs", "eventbridge", "api gateway", "elastic beanstalk", "ecs", "eks", "fargate",
    "cloudformation", "cloudwatch", "codebuild", "codepipeline", "codedeploy", "systems manager",
    "secrets manager", "kms", "cognito", "amplify", "appsync", "glue", "athena", "quickSight"
]

# --- Specific Azure Services ---
AZURE_SERVICES = [
    "azure vm", "azure app service", "azure functions", "azure sql database", "cosmos db",
    "azure active directory", "aad", "azure devops", "arm templates", "azure resource manager",
    "azure kubernetes service", "aks", "azure container instances", "azure storage", "blob storage",
    "azure monitor", "application insights", "azure pipeline", "key vault", "azure CDN",
    "azure event grid", "service bus", "azure data factory", "synapse analytics", "power bi"
]

# --- Specific GCP Services ---
GCP_SERVICES = [
    "compute engine", "app engine", "cloud functions", "cloud run", "bigquery", "bigtable",
    "cloud spanner", "cloud sql", "firestore", "cloud storage", "gcs", "vertex ai",
    "google kubernetes engine", "gke", "cloud build", "cloud deployment manager",
    "iam", "cloud identity", "vpc", "cloud load balancing", "stackdriver", "operations",
    "pub/sub", "dataflow", "dataproc", "looker", "looker studio"
]

# --- AI/ML/Data Science ---
AI_ML_DATA_SCIENCE = [
    "machine learning", "ml", "deep learning", "neural networks", "natural language processing", "nlp",
    "computer vision", "cv", "generative ai", "llm", "large language models", "tensorflow", "pytorch",
    "keras", "scikit-learn", "opencv", "hugging face", "transformers", "langchain", "llama index",
    "apache spark", "pyspark", "hadoop", "hdfs", "mapreduce", "hive", "pig", "kafka", "kafka streams",
    "airflow", "prefect", "dagster", "dbt", "data analysis", "data visualization", "tableau", "power bi",
    "qlik", "matplotlib", "seaborn", "plotly", "pandas", "numpy", "jupyter", "rstudio", "mlops"
]

# --- Networking & Security ---
NETWORKING_SECURITY = [
    "tcp/ip", "dns", "dhcp", "http", "https", "ssl", "tls", "vpn", "ipsec", "ssh",
    "network security", "cybersecurity", "application security", "appsec", "devsecops",
    "owasp", "penetration testing", "pentesting", "vulnerability assessment", "siem",
    "soc", "firewalls", "waf", "ids", "ips", "zero trust", "pki", "cryptography", "encryption",
    "iso 27001", "soc 2", "pci dss", "gdpr", "compliance", "risk management", "incident response"
]

# --- Software Development Practices ---
SOFTWARE_PRACTICES = [
    "agile", "scrum", "kanban", "devops", "devsecops", "gitops", "ci/cd", "continuous integration",
    "continuous delivery", "continuous deployment", "test driven development", "tdd", "bdd",
    "pair programming", "code review", "refactoring", "clean code", "design patterns",
    "solid principles", "microservices", "monolith", "serverless", "rest", "restful api",
    "graphql", "grpc", "soap", "api design", "event driven architecture", "eda", "domain driven design", "ddd",
    "twelve factor app", "object oriented programming", "oop", "functional programming", "fp"
]

# --- QA & Testing ---
QA_TESTING = [
    "unit testing", "integration testing", "end-to-end testing", "e2e testing", "ui testing",
    "regression testing", "performance testing", "load testing", "stress testing", "security testing",
    "accessibility testing", "a11y", "manual testing", "automated testing", "selenium",
    "cypress", "playwright", "puppeteer", "jest", "mocha", "jasmine", "karma", "phpunit",
    "junit", "testng", "cucumber", "specflow", "soapui", "postman", "jmeter", "gatling"
]

# --- Enterprise & Other Tech ---
ENTERPRISE_TECH = [
    "salesforce", "servicenow", "sap", "oracle ebs", "microsoft dynamics", "sharepoint",
    "sitecore", "adobe experience manager", "aem", "workday", "mainframe", "cobol",
    "soa", "esb", "tibco", "webmethods", "mulesoft", "ibm mq", "active directory", "ldap",
    "windows server", "exchange server", "vmware vsphere", "citrix", "san", "nas", "raid"
]

# --- IT Support & Administration ---
IT_SUPPORT = [
    "itil", "ticketing systems", "service desk", "help desk", "technical support", "troubleshooting",
    "hardware", "software installation", "network administration", "system administration",
    "windows", "macos", "active directory", "group policy", "mdm", "intune", "jamf", "sccm",
    "backup", "disaster recovery", "bcdr", "patch management", "remote support", "voip"
]

# --- Low-Level & Embedded ---
LOW_LEVEL_EMBEDDED = [
    "embedded systems", "arduino", "raspberry pi", "iot", "internet of things", "fpga", "verilog", "vhdl",
    "rtos", "real-time operating system", "device drivers", "kernel development", "assembly", "cpp"
]

# --- Game Development ---
GAME_DEVELOPMENT = [
    "unity", "unreal engine", "cryengine", "godot", "directx", "opengl", "vulkan", "shaders", "hlsl", "glsl",
    "game design", "3d modeling", "blender", "maya", "3ds max", "physics engine", "vr", "ar", "virtual reality", "augmented reality"
]

# --- Blockchain ---
BLOCKCHAIN = [
    "blockchain", "smart contracts", "solidity", "web3", "ethereum", "bitcoin", "hyperledger", "fabric",
    "cryptocurrency", "nft", "defi", "distributed ledger", "consensus algorithms"
]

# ==================== GENERAL & SOFT SKILLS ====================
# Categorized for better analysis

# --- Communication & Collaboration ---
SOFT_SKILLS_COMMUNICATION = [
    "communication", "written communication", "verbal communication", "presentation", "public speaking",
    "active listening", "storytelling", "negotiation", "influencing", "collaboration", "teamwork",
    "conflict resolution", "mediation", "customer service", "client facing", "stakeholder management",
    "relationship building", "networking"
]

# --- Leadership & Management ---
SOFT_SKILLS_LEADERSHIP = [
    "leadership", "team leadership", "technical leadership", "mentoring", "coaching", "people management",
    "project management", "program management", "product management", "agile coaching", "scrum mastery",
    "decision making", "strategic thinking", "vision", "delegation", "change management", "risk management",
    "resource allocation", "budgeting", "forecasting", "kanban", "prioritization", "time management"
]

# --- Cognitive & Analytical ---
SOFT_SKILLS_ANALYTICAL = [
    "problem solving", "critical thinking", "analytical skills", "data analysis", "research", "troubleshooting",
    "root cause analysis", "debugging", "log analysis", "business analysis", "requirements gathering",
    "systems thinking", "innovation", "creativity", "design thinking", "curiosity", "attention to detail",
    "quality assurance", "process improvement", "lean", "six sigma"
]

# --- Business & Domain Knowledge ---
BUSINESS_DOMAIN_KNOWLEDGE = [
    "business acumen", "domain knowledge", "finance", "accounting", "marketing", "digital marketing", "seo",
    "sem", "social media", "content marketing", "ecommerce", "retail", "healthcare", "healthtech", "fintech",
    "edtech", "supply chain", "logistics", "manufacturing", "hr", "human resources", "recruiting", "sales",
    "business development", "partner management", "go-to-market", "gtm", "product marketing"
]

# --- Tools & Productivity ---
PRODUCTIVITY_TOOLS = [
    "microsoft office", "excel", "word", "powerpoint", "outlook", "google workspace", "gsuite", "sheets",
    "docs", "slides", "jira", "confluence", "trello", "asana", "monday.com", "notion", "slack",
    "microsoft teams", "zoom", "sharepoint", "salesforce", "sap", "quickbooks", "xero", "hubspot", "zendesk"
]

# --- Personal Attributes ---
PERSONAL_ATTRIBUTES = [
    "adaptability", "flexibility", "resilience", "persistence", "self motivation", "initiative",
    "proactive", "work ethic", "reliability", "accountability", "ownership", "independence",
    "autonomy", "stress management", "patience", "empathy", "cultural fit", "growth mindset"
]

# ==================== COMBINED & NORMALIZED LISTS ====================

# Combine all IT skills into one massive list
IT_SKILLS = (
    PROGRAMMING_LANGUAGES +
    WEB_FRONTEND +
    WEB_BACKEND_FRAMEWORKS +
    MOBILE_DEVELOPMENT +
    DATABASES +
    CLOUD_PLATFORMS +
    DEVOPS_TOOLS +
    AWS_SERVICES +
    AZURE_SERVICES +
    GCP_SERVICES +
    AI_ML_DATA_SCIENCE +
    NETWORKING_SECURITY +
    SOFTWARE_PRACTICES +
    QA_TESTING +
    ENTERPRISE_TECH +
    IT_SUPPORT +
    LOW_LEVEL_EMBEDDED +
    GAME_DEVELOPMENT +
    BLOCKCHAIN
)

# Combine all General/Soft skills into one massive list
GENERAL_SKILLS = (
    IT_SKILLS +  # Includes all technical skills
    SOFT_SKILLS_COMMUNICATION +
    SOFT_SKILLS_LEADERSHIP +
    SOFT_SKILLS_ANALYTICAL +
    BUSINESS_DOMAIN_KNOWLEDGE +
    PRODUCTIVITY_TOOLS +
    PERSONAL_ATTRIBUTES
)

# Normalize all to lowercase and remove duplicates by converting to a set and back to a list
IT_SKILLS = list(set([s.lower().strip() for s in IT_SKILLS]))
GENERAL_SKILLS = list(set([s.lower().strip() for s in GENERAL_SKILLS]))

# Optional: Sort the lists for readability if needed
# IT_SKILLS.sort()
# GENERAL_SKILLS.sort()

# Print the count to see how massive the list is
print(f"Number of unique IT skills: {len(IT_SKILLS)}")
print(f"Number of unique general skills: {len(GENERAL_SKILLS)}")




#5 weaklabeling function
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{6,}\d)')  # rough

SECTION_PATTERNS = {
    'summary': r'^(Summary|Professional Summary|Profile):?$',
    'experience': r'^(Experience|Work Experience|Professional Experience):?$',
    'education': r'^(Education|Academic Background):?$',
    'skills': r'^(Skills|Technical Skills|Competencies):?$',
    'projects': r'^(Projects|Personal Projects):?$',
    'certifications': r'^(Certifications|Achievements|Highlights):?$',
}

def weak_label_text(text):
    lines = text.splitlines()
    tokens = []
    labels = []
    current_section = None
    previous_label = 'O'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect section
        for sec, pattern in SECTION_PATTERNS.items():
            if re.match(pattern, line, re.I):
                current_section = sec
                # Skip labeling the header line itself
                continue
        
        words = re.split(r'\s+', line)
        for word_idx, word in enumerate(words):
            tokens.append(word)
            label = 'O'
            
            # Email and Phone (single token, B- only)
            if EMAIL_RE.match(word):
                label = 'B-EMAIL'
            elif PHONE_RE.match(word):
                label = 'B-PHONE'
            # Name (first line, multi-word)
            elif current_section is None and len(words) <= 5 and all(w.istitle() for w in words if w.isalpha()):
                label = 'B-NAME' if word_idx == 0 else 'I-NAME'
            # Section-based labeling
            elif current_section:
                if current_section == 'skills':
                    # Match multi-word skills from dict
                    matched = False
                    for skill in GENERAL_SKILLS:
                        skill_words = skill.split()
                        if ' '.join(words[word_idx:word_idx+len(skill_words)]).lower() == skill:
                            for i in range(len(skill_words)):
                                labels.append('B-SKILL' if i == 0 else 'I-SKILL')
                            matched = True
                            break
                    if not matched:
                        label = 'B-SKILL' if previous_label != 'I-SKILL' else 'I-SKILL'
                elif current_section == 'experience':
                    label = 'B-EXP' if previous_label != 'I-EXP' else 'I-EXP'
                elif current_section == 'education':
                    label = 'B-EDU' if previous_label != 'I-EDU' else 'I-EDU'
                elif current_section == 'projects':
                    label = 'B-PROJECT' if previous_label != 'I-PROJECT' else 'I-PROJECT'
                elif current_section == 'certifications':
                    label = 'B-CERT' if previous_label != 'I-CERT' else 'I-CERT'
            
            labels.append(label)
            previous_label = label
    
    return tokens, labels


#6 weaklabeling
MAX_SAMPLES = len(df)
examples = []
for idx, row in tqdm(df.head(MAX_SAMPLES).iterrows(), total=min(MAX_SAMPLES, len(df))):
    text = row['resume_text']
    tokens, labels = weak_label_text(text)
    examples.append({'id': str(row['ID']), 'tokens': tokens, 'labels': labels, 'raw_text': text})
print("examples:", len(examples))

#7 testing weaklabeling
for ex in examples[:3]:
    print("ID:", ex['id'])
    for t,l in list(zip(ex['tokens'][:120], ex['labels'][:120]))[:80]:
        print(f"{t} -> {l}")
    print("-----\n")

#7encode fuction 

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Build full label list with B/I/O
entity_types = ['NAME', 'EMAIL', 'PHONE', 'SKILL', 'EXP', 'EDU', 'PROJECT', 'CERT']
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]
label_list = sorted(set(label_list))  # Dedup if needed

unique_labels = set(l for ex in examples for l in ex['labels'])
print("Unique labels found:", unique_labels)  # Debug

def encode_example(ex):
    tokens = ex['tokens']
    labels = ex['labels']
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=512)
    word_ids = encoding.word_ids()
    label_ids = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            label_ids.append(-100)
        elif word_id != previous_word_id:  # New word
            label = labels[word_id]
            label_ids.append(label_list.index(label))
        else:
            # Subword: Convert B to I if applicable
            label = labels[word_id]
            if label.startswith('B-'):
                label = 'I-' + label[2:]
            label_ids.append(label_list.index(label) if label in label_list else -100)
        previous_word_id = word_id
    encoding['labels'] = label_ids
    encoding['id'] = ex['id']
    return encoding

hf_dataset = Dataset.from_list(examples)
hf_dataset_enc = hf_dataset.map(encode_example, batched=False, remove_columns=hf_dataset.column_names)
hf_dataset_enc = hf_dataset_enc.train_test_split(test_size=0.1)
train_ds = hf_dataset_enc['train']
eval_ds = hf_dataset_enc['test']

print(train_ds[0].keys())


#8 model training
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
training_args = TrainingArguments(
    output_dir="/content/models/resume-ner-v2",  # New dir
    num_train_epochs=5,  # Increased
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # For low memory
    do_eval=True,
    eval_strategy="epoch",  # Eval each epoch
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=50,
    push_to_hub=False,
    report_to="none"
)

#9 define metrics
metric = load("seqeval")

def align_predictions(predictions, label_ids):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = []
    out_pred_list = []
    for i in range(batch_size):
        example_labels = []
        example_preds = []
        for j in range(seq_len):
            if label_ids[i,j] != -100:
                example_labels.append(label_list[label_ids[i,j]])
                example_preds.append(label_list[preds[i,j]])
        out_label_list.append(example_labels)
        out_pred_list.append(example_preds)
    return out_pred_list, out_label_list

def compute_metrics(p):
    preds, labels = p
    preds_list, labels_list = align_predictions(preds, labels)
    results = metric.compute(predictions=preds_list, references=labels_list)
    return {"overall_f1": results.get("overall_f1",0)}

#9 trainning model
import os
os.environ["WANDB_DISABLED"] = "true"
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("/content/models/resume-ner-v2")
tokenizer.save_pretrained("/content/models/resume-ner-v2")


#10 inference function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_resume(text, model_path="/content/models/resume-ner-v2", chunk_size=400):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    model.eval()
    
    # Chunk text if long
    words = re.split(r'\s+', text)
    all_entities = defaultdict(list)
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        enc = tokenizer(chunk_words, is_split_into_words=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        word_ids = enc.word_ids(0)
        
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()
        
        current_ent = None
        current_tokens = []
        previous_wid = None
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != previous_wid:
                if current_tokens:
                    all_entities[current_ent].append(" ".join(current_tokens))
                current_tokens = []
                current_ent = None
            label = label_list[preds[idx]]
            token = tokenizer.convert_ids_to_tokens(enc['input_ids'][0][idx])
            if token.startswith("##"):
                if current_tokens:
                    current_tokens[-1] += token[2:]
            else:
                current_tokens.append(token)
            
            if label == 'O':
                current_ent = None
            elif label.startswith('B-'):
                current_ent = label[2:]
            elif label.startswith('I-') and current_ent == label[2:]:
                pass
            else:
                current_ent = None
            
            previous_wid = wid
        
        if current_tokens:
            all_entities[current_ent].append(" ".join(current_tokens))
    
    # Post-process
    out_json = {}
    for ent, spans in all_entities.items():
        if ent == 'SKILL':
            skills = set()
            for span in spans:
                low = span.lower()
                matched = [sk for sk in GENERAL_SKILLS if sk == low or sk in low.split()]
                skills.update(matched or [span])
            out_json['skills'] = list(skills)
        else:
            out_json[ent.lower()] = list(set(spans))  # Dedup spans
    
    return out_json

#11 inferecnce with translation
# Load translation model
mt_model_name = "Helsinki-NLP/opus-mt-fr-en"
mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name, use_fast=True)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name).to(device)

def translate_to_en(text):
    try:
        if detect(text[:500]) == 'fr':
            inputs = mt_tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            translated = mt_model.generate(**inputs)
            return mt_tokenizer.decode(translated[0], skip_special_tokens=True)
        return text
    except:
        return text

# Example
sample_text = df['resume_text'].iloc[0]
translated_text = translate_to_en(sample_text)
resume_json = infer_resume(translated_text)
print(resume_json)

#12 extract from pdf and docx
def extract_text_from_file(file_path):
    if file_path.lower().endswith('.pdf'):
        text = extract_text(file_path)
    elif file_path.lower().endswith('.docx'):
        text = docx2txt.process(file_path)
    else:
        raise ValueError("Use PDF or DOCX.")
    text = re.sub(r'\s+', ' ', text).strip()  # Clean
    return text

# Test with your CV
pdf_path = "/content/aziz.pdf"
pdf_text = extract_text_from_file(pdf_path)
print("First 500 chars of extracted text:\n", pdf_text[:500], "...\n")

translated_pdf = translate_to_en(pdf_text)
resume_json = infer_resume(translated_pdf)
print("Extracted resume info:\n", resume_json)
