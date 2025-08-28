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



#3 load  and clean resume data
# Define paths
BASE_PATH = "/content/Resume"
CSV_PATH = f"{BASE_PATH}/Resume.csv"

# Load CSV
df = pd.read_csv(CSV_PATH, encoding='utf-8', dtype=str, low_memory=False)
df = df.fillna('')  # Replace NaN with empty strings
print(f"Number of rows: {len(df)}")
print(df.head(2).T)

# Clean text function
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
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_RE = re.compile(r'\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
NAME_RE = re.compile(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)?)?\b', re.I)
DATE_RE = re.compile(r'\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})\b', re.I)
AGE_RE = re.compile(r'\b(?:age\s*\d{1,2}|\d{1,2}\s*years?\s*old|born\s*(?:\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}))\b', re.I)

SECTION_PATTERNS = {
    'summary': [re.compile(r'\b(summary|professional summary|profile|about|objective|career objective)\b', re.I)],
    'experience': [re.compile(r'\b(experience|work experience|professional experience|work history|employment|career history|job history)\b', re.I)],
    'education': [re.compile(r'\b(education|academic background|qualifications|academic qualifications|degrees)\b', re.I)],
    'skills': [re.compile(r'\b(skills|technical skills|competencies|expertise|technical proficiencies|abilities)\b', re.I)],
    'projects': [re.compile(r'\b(projects|personal projects|portfolio|key projects|work projects)\b', re.I)],
    'certifications': [re.compile(r'\b(certifications|achievements|highlights|accomplishments|certificates|credentials|awards)\b', re.I)],
}

def weak_label_text(text, resume_id):
    lines = text.replace('\r', '\n').split('\n')
    tokens = []
    labels = []
    current_section = None
    line_count = 0
    stop_words = {'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'profile', 'objective'}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        lower_line = line.lower()
        is_header = False
        for sec, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(lower_line) and len(line.split()) <= 10:
                    current_section = sec
                    print(f"ID: {resume_id} - Detected section: {sec} in line: {line}")
                    is_header = True
                    break
            if is_header:
                break
        
        if is_header:
            continue
        
        words = re.split(r'\s*,\s*|\s+', line.strip(',.:;'))
        word_idx = 0
        while word_idx < len(words):
            word = words[word_idx].strip()
            if not word:
                word_idx += 1
                continue
            
            matched = False
            if current_section == 'skills':
                for skill in GENERAL_SKILLS:
                    skill_words = skill.split()
                    if word_idx + len(skill_words) <= len(words):
                        candidate = ' '.join(words[word_idx:word_idx+len(skill_words)]).lower()
                        candidate_normalized = re.sub(r'[^\w\s]', '', candidate)
                        skill_normalized = re.sub(r'[^\w\s]', '', skill)
                        if (candidate_normalized == skill_normalized or 
                            fuzz.ratio(candidate_normalized, skill_normalized) > 90 or
                            skill_normalized in candidate_normalized.split()):
                            for i in range(len(skill_words)):
                                tokens.append(words[word_idx + i])
                                labels.append('B-SKILL' if i == 0 else 'I-SKILL')
                            print(f"ID: {resume_id} - Matched skill: {candidate} (score: {fuzz.ratio(candidate_normalized, skill_normalized)})")
                            word_idx += len(skill_words)
                            matched = True
                            break
            
            if matched:
                continue
            
            tokens.append(word)
            label = 'O'
            
            if EMAIL_RE.match(word):
                label = 'B-EMAIL'
            elif PHONE_RE.match(word):
                label = 'B-PHONE'
            elif AGE_RE.match(' '.join(words[word_idx:word_idx+3]).lower()):
                label = 'B-AGE'
                tokens.append(word)
                labels.append(label)
                word_idx += 1
                if word_idx < len(words) and ('years' in words[word_idx].lower() or 'old' in words[word_idx].lower() or 'born' in words[word_idx].lower()):
                    tokens.append(words[word_idx])
                    labels.append('I-AGE')
                    word_idx += 1
                    if word_idx < len(words) and 'old' in words[word_idx].lower():
                        tokens.append(words[word_idx])
                        labels.append('I-AGE')
                        word_idx += 1
                continue
            elif (current_section is None and line_count < 3 and 
                  NAME_RE.match(' '.join(words[word_idx:word_idx+3])) and 
                  word.lower() not in stop_words):
                label = 'B-NAME' if word_idx == 0 else 'I-NAME'
            elif current_section:
                if current_section == 'skills':
                    label = 'B-SKILL' if word_idx == 0 or words[word_idx-1].endswith((',', '.', ':', ';')) else 'I-SKILL'
                elif current_section == 'experience':
                    label = 'B-EXP' if word_idx == 0 or words[word_idx-1].endswith((',', '.', ':', ';')) else 'I-EXP'
                elif current_section == 'education':
                    label = 'B-EDU' if word_idx == 0 or words[word_idx-1].endswith((',', '.', ':', ';')) else 'I-EDU'
                elif current_section == 'projects':
                    label = 'B-PROJECT' if word_idx == 0 or words[word_idx-1].endswith((',', '.', ':', ';')) else 'I-PROJECT'
                elif current_section == 'certifications':
                    label = 'B-CERT' if word_idx == 0 or words[word_idx-1].endswith((',', '.', ':', ';')) else 'I-CERT'
            
            labels.append(label)
            word_idx += 1
        
        line_count += 1
    
    return tokens, labels
#6 weaklabeling
examples = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    resume_id = str(row['ID'])
    text = row['Resume_str']
    tokens, labels = weak_label_text(text, resume_id)
    cleaned_text = clean_text(' '.join(tokens))
    examples.append({'id': resume_id, 'tokens': tokens, 'labels': labels, 'raw_text': cleaned_text})

print("Number of examples:", len(examples))

# Test weak labeling
for ex in examples[:2]:
    print("ID:", ex['id'])
    for t, l in list(zip(ex['tokens'][:20], ex['labels'][:20])):
        print(f"{t} -> {l}")
    print("-----\n")

unique_labels = set(l for ex in examples for l in ex['labels'])
print("Unique labels found:", unique_labels)


# 7 encode function

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

entity_types = ['NAME', 'EMAIL', 'PHONE', 'SKILL', 'EXP', 'EDU', 'PROJECT', 'CERT', 'AGE']
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]
label_list = sorted(set(label_list))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}
print("Label list:", label_list)

def encode_example(ex):
    tokens = ex['tokens']
    labels = ex['labels']
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors=None
    )
    word_ids = encoding.word_ids()
    label_ids = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[labels[word_idx]])
        else:
            current_label = labels[word_idx]
            if current_label.startswith('B-'):
                label_ids.append(label2id['I-' + current_label[2:]])
            else:
                label_ids.append(label2id[current_label])
        previous_word_idx = word_idx
    
    encoding['labels'] = label_ids
    encoding['id'] = ex['id']
    return encoding

hf_dataset = Dataset.from_list(examples)
hf_dataset_enc = hf_dataset.map(encode_example, batched=False, remove_columns=hf_dataset.column_names)
hf_dataset_enc = hf_dataset_enc.train_test_split(test_size=0.1, seed=42)
train_ds = hf_dataset_enc['train']
eval_ds = hf_dataset_enc['test']
print("Train dataset size:", len(train_ds), "Eval dataset size:", len(eval_ds))






#8  weightedtrainer 
all_labels = [l for ex in examples for l in ex['labels']]
label_counts = Counter(all_labels)
total = len(all_labels)
weights = [total / (len(label_list) * max(label_counts.get(label, 1), 10)) for label in label_list]  # Avoid division by small counts
weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", {label: w.item() for label, w in zip(label_list, weights)})

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, len(label_list)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss






#9 model training
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="/content/models/resume-ner-v5",
    num_train_epochs=15,  # Increased epochs
    per_device_train_batch_size=16,  # Larger batch size
    per_device_eval_batch_size=16,
    learning_rate=3e-5,  # Slightly higher
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none",
    warmup_ratio=0.1
)

metric = load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
#10 training model
import os
os.environ["WANDB_DISABLED"] = "true"
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,  # Fix FutureWarning
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("/content/models/resume-ner-v5")
tokenizer.save_pretrained("/content/models/resume-ner-v5")


#11 translation and extraction functions
mt_model_name = "facebook/m2m100_418M"
mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name).to(device)

def translate_to_en(text, pdf_path):
    try:
        if not text.strip():
            print("Warning: Empty input text for translation")
            return text
        lang = 'fr' if 'sana.pdf' in pdf_path or 'yasmine.pdf' in pdf_path else detect(text[:500])
        print(f"Detected/Forced language: {lang}")
        if lang == 'fr':
            words = text.split()
            chunks = [' '.join(words[i:i+150]) for i in range(0, len(words), 150)]  # Smaller chunks
            translated_chunks = []
            for chunk in chunks:
                inputs = mt_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                translated = mt_model.generate(**inputs, max_length=512, num_beams=5, forced_bos_token_id=mt_tokenizer.get_lang_id("en"))
                translated_text = mt_tokenizer.decode(translated[0], skip_special_tokens=True)
                translated_chunks.append(translated_text)
            translated_text = ' '.join(translated_chunks)
            print(f"Translated text (first 100 chars): {translated_text[:100]}")
            return translated_text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# 14. Extract text from file
def extract_text_from_file(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text(file_path)
        elif file_path.lower().endswith('.docx'):
            text = docx2txt.process(file_path)
        else:
            raise ValueError("Use PDF or DOCX.")
        text = re.sub(r'\s+', ' ', text).strip()
        print(f"Extracted text (first 100 chars): {text[:100]}")
        return text
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return ""



#11 inference function
def infer_resume(text, model_path="/content/models/resume-ner-v5", chunk_size=150):
    if not text.strip():
        print("Error: Input text is empty")
        return {}
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        model.eval()
        print(f"Loaded model and tokenizer from {model_path}")
        
        words = re.split(r'\s+', text.strip())
        print(f"Input words count: {len(words)}")
        if len(words) == 0:
            print("Error: No valid words after splitting")
            return {}
        
        all_entities = defaultdict(list)
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i+chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}: {len(chunk_words)} words")
            if not chunk_words:
                continue
                
            enc = tokenizer(chunk_words, is_split_into_words=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            word_ids = enc.word_ids(batch_index=0)
            
            if not word_ids or all(wid is None for wid in word_ids):
                print("Warning: No valid word IDs in chunk, skipping")
                continue
                
            with torch.no_grad():
                logits = model(**enc).logits
            print(f"Logits shape: {logits.shape}")
            
            if logits.dim() != 3 or logits.shape[1] <= 1:
                print(f"Warning: Invalid logits shape {logits.shape}, skipping chunk")
                continue
                
            preds = torch.argmax(logits, dim=2)[0].cpu().numpy()
            print(f"Preds length: {len(preds)}, Word IDs length: {len(word_ids)}")
            print(f"Sample preds: {preds[:10]}")
            
            if len(preds) != len(word_ids):
                print(f"Warning: Mismatch between preds ({len(preds)}) and word_ids ({len(word_ids)}), skipping")
                continue
            
            current_ent = None
            current_tokens = []
            previous_wid = None
            for idx, wid in enumerate(word_ids):
                if wid is None:
                    continue
                if wid != previous_wid and current_tokens:
                    if current_ent and current_tokens:
                        all_entities[current_ent].append(" ".join(current_tokens))
                    current_tokens = []
                    current_ent = None
                label = id2label[preds[idx]]
                token = tokenizer.convert_ids_to_tokens(enc['input_ids'][0][idx].item())
                if token.startswith("##"):
                    if current_tokens:
                        current_tokens[-1] += token[2:]
                else:
                    if token not in ['[CLS]', '[SEP]', '[PAD]']:
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
            
            if current_tokens and current_ent:
                all_entities[current_ent].append(" ".join(current_tokens))
        
        # Post-process entities
        out_json = {}
        stop_words = {'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'profile', 'objective'}
        for ent, spans in all_entities.items():
            if ent is None:
                continue
            if ent == 'SKILL':
                skills = set()
                for span in spans:
                    span_lower = span.lower()
                    matched = False
                    for skill in GENERAL_SKILLS:
                        if (fuzz.ratio(span_lower, skill) > 90 or 
                            skill in span_lower.split() or 
                            any(fuzz.ratio(word, skill) > 90 for word in span_lower.split())):
                            skills.add(skill)
                            matched = True
                    if not matched and span_lower not in stop_words:
                        skills.add(span)
                out_json['skills'] = list(skills)
            elif ent == 'NAME':
                filtered_spans = [span for span in spans if span.lower() not in stop_words and len(span.split()) <= 3]
                out_json['name'] = list(set(filtered_spans))
            else:
                out_json[ent.lower()] = list(set(spans))
        
        print(f"Extracted entities: {out_json}")
        return out_json
    
    except Exception as e:
        print(f"Inference error: {e}")
        return {}

# 16. Test on CVs
pdf_paths = [
    "/content/aziz.pdf",
    "/content/frosty.pdf",
    "/content/sana.pdf",
    "/content/yasmine.pdf"
]

for pdf_path in pdf_paths:
    print(f"\n{'='*60}")
    print(f"Processing {pdf_path}")
    print(f"{'='*60}")
    
    text = extract_text_from_file(pdf_path)
    if not text:
        print("Could not extract text")
        continue
    
    translated_text = translate_to_en(text, pdf_path)
    result = infer_resume(translated_text)
    
    print(f"\nExtracted entities from {pdf_path}:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")