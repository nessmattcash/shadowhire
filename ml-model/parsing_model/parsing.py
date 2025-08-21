#1 unziping the dataset
import zipfile
#!unzip -q /content/archive.zip -d /content/
BASE = "/content/Resume"
#!ls -R "$BASE"


#2 install the dependecies
# Core ML & parsing libs
#!pip install -q pdfminer.six python-docx transformers datasets seqeval holmes
#!pip install -q sentencepiece sacremoses
#!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
#!pip install -q scikit-learn tqdm
#!pip install pdfminer.six --quiet


#3imports
import os, re, json, math, random
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pdfminer.high_level import extract_text


#4 testing the imports 
import numpy as np
import pandas as pd
import tensorflow as tf

print(np.__version__)
print(pd.__version__)
print(tf.__version__)


#5 read the annotatited file

CSV_PATH = "/content/Resume/Resume.csv"  
df = pd.read_csv(CSV_PATH, encoding='utf-8', dtype=str, low_memory=False)
df = df.fillna('')  # replace NaN
print("rows:", len(df))
display(df.head(2).T)
#result 
#rows: 2484
#0	1
#ID	16852973	22323967
#Resume_str	HR ADMINISTRATOR/MARKETING ASSOCIATE\...	HR SPECIALIST, US HR OPERATIONS ...
#Resume_html	<div class="fontsize fontface vmargins hmargin...	<div class="fontsize fontface vmargins hmargin...
#Category	HR	HR


#6 cleanning function 
def clean_text(s):
    s = s.replace('\r',' ').replace('\n',' ').replace('\t',' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    return s

df['resume_text'] = df['Resume_str'].astype(str).apply(clean_text)

#7 BUILD SKILL DICTIONARY
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

#8 Weak labeling functions (regex + sections)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s]{6,}\d)')  # rough
YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')

SECTION_HEADERS = ['summary','experience','work experience','education','skills','projects','certifications','achievements','highlights','professional summary','profile']
SECTION_HEADERS = [h.lower() for h in SECTION_HEADERS]

def weak_label_text(text):
    text_lower = text.lower()
    # naive split into sentences/tokens
    tokens = re.split(r'(\s+)', text)  # keep whitespace tokens so we can join later
    labels = ['O'] * len(tokens)
    
    # label emails
    for m in EMAIL_RE.finditer(text):
        start, end = m.span()
        # mark overlapping tokens
        i = 0
        pos = 0
        while i < len(tokens):
            pos_next = pos + len(tokens[i])
            if pos_next > start and pos < end:
                labels[i] = 'B-EMAIL' if labels[i]=='O' else labels[i]
                # don't overcomplicate I- tags for whitespace tokens
            pos = pos_next
            i += 1
    
    # label phones
    for m in PHONE_RE.finditer(text):
        start, end = m.span()
        i = 0; pos = 0
        while i < len(tokens):
            pos_next = pos + len(tokens[i])
            if pos_next > start and pos < end:
                labels[i] = 'B-PHONE' if labels[i]=='O' else labels[i]
            pos = pos_next; i += 1

    # label skills by substring matching (best-effort)
    low = text_lower
    for skill in GENERAL_SKILLS:
        for m in re.finditer(re.escape(skill), low):
            start, end = m.span()
            i=0; pos=0
            while i < len(tokens):
                pos_next = pos + len(tokens[i])
                if pos_next > start and pos < end:
                    labels[i] = 'B-SKILL' if labels[i]=='O' else labels[i]
                pos = pos_next; i += 1

    # label sections by headers - mark subsequent sentences until next header as that section
    # simple approach: find header keywords positions
    for header in SECTION_HEADERS:
        for m in re.finditer(header, text_lower):
            start = m.start()
            # find next 500 chars as section content (heuristic)
            sec_end = min(len(text), m.end() + 800)
            i=0; pos=0
            while i < len(tokens):
                pos_next = pos + len(tokens[i])
                if pos_next > m.start() and pos < sec_end:
                    # mark as SECTION (helper) - we will later map into EDU/EXP/PROJECT by header
                    if header in ('education','educations','education:'):
                        labels[i] = 'B-EDU' if labels[i]=='O' else labels[i]
                    elif header in ('experience','work experience'):
                        labels[i] = 'B-EXP' if labels[i]=='O' else labels[i]
                    elif header in ('projects','personal projects'):
                        labels[i] = 'B-PROJECT' if labels[i]=='O' else labels[i]
                    elif header in ('skills',):
                        labels[i] = 'B-SKILL' if labels[i]=='O' else labels[i]
                    elif header in ('certifications','achievements'):
                        labels[i] = 'B-CERT' if labels[i]=='O' else labels[i]
                pos = pos_next; i += 1

    # name detection heuristic: often first non-empty line up to first newline
    first_line = text.splitlines()
    if first_line:
        candidate = first_line[0].strip()
        if len(candidate.split()) <= 5 and len(candidate) > 2 and len(candidate) < 80 and '@' not in candidate and any(c.isalpha() for c in candidate):
            # mark first candidate tokens as name
            start = 0
            end = len(candidate)
            i=0; pos=0
            # attempt to match in tokens
            while i < len(tokens):
                pos_next = pos + len(tokens[i])
                if pos_next > start and pos < end:
                    labels[i] = 'B-NAME' if labels[i]=='O' else labels[i]
                pos = pos_next; i += 1

    return tokens, labels



#9 genrate weak labels for all resumes
MAX_SAMPLES = len(df)
examples = []
for idx, row in tqdm(df.head(MAX_SAMPLES).iterrows(), total=min(MAX_SAMPLES,len(df))):
    text = row['resume_text']
    tokens, labels = weak_label_text(text)
    # collapse whitespace tokens (we kept them earlier) to obtain word tokens
    word_tokens = []
    word_labels = []
    for tok, lab in zip(tokens, labels):
        if tok.strip()=='':
            continue
        word_tokens.append(tok)
        word_labels.append(lab)
    examples.append({'id': str(row['ID']), 'tokens': word_tokens, 'labels': word_labels, 'raw_text': text})
print("examples:", len(examples))

#10 inspect some examples
for ex in examples[:3]:
    print("ID:", ex['id'])
    for t,l in list(zip(ex['tokens'][:120], ex['labels'][:120]))[:80]:
        print(f"{t} -> {l}")
    print("-----\n")

#result
#ID: 16852973
#HR -> B-SKILL
#ADMINISTRATOR/MARKETING -> B-SKILL
#ASSOCIATE -> B-SKILL
#HR -> B-SKILL
#ADMINISTRATOR -> B-SKILL
#Summary -> B-SKILL
#Dedicated -> B-SKILL
#Customer -> B-SKILL
#Service -> B-SKILL
#Manager -> B-SKILL
#with -> O
#15+ -> O
#years -> B-SKILL
#of -> O
#experience -> B-SKILL
#in -> B-EXP
#Hospitality -> B-EXP
#and -> B-EXP
#Customer -> B-SKILL
#Service -> B-SKILL
#Management. -> B-EXP
#Respected -> B-SKILL
#builder -> B-SKILL
#and -> B-EXP
#leader -> B-SKILL
#of -> B-EXP
#customer-focused -> B-SKILL
#teams; -> B-EXP
#strives -> B-SKILL
#to -> B-EXP
#instill -> B-EXP
#a -> B-EXP
#shared, -> B-SKILL
#enthusiastic -> B-SKILL
#commitment -> B-SKILL
#to -> B-EXP
#customer -> B-SKILL
#service. -> B-SKILL
#Highlights -> B-EXP
#Focused -> B-SKILL
#on -> B-EXP
#customer -> B-SKILL
#satisfaction -> B-SKILL
#Team -> B-EXP
#management -> B-EXP
#Marketing -> B-SKILL
#savvy -> B-EXP
#Conflict -> B-SKILL
#resolution -> B-SKILL
#techniques -> B-SKILL
#Training -> B-SKILL
#and -> B-EXP
#development -> B-EXP
#Skilled -> B-EXP
#multi-tasker -> B-SKILL
#Client -> B-SKILL
#relations -> B-SKILL
#specialist -> B-SKILL
#Accomplishments -> B-SKILL
#Missouri -> B-SKILL
#DOT -> B-EXP
#Supervisor -> B-SKILL
#Training -> B-SKILL
#Certification -> B-SKILL
#Certified -> B-SKILL
#by -> B-EXP
#IHG -> B-EXP
#in -> B-EXP
#Customer -> B-SKILL
#Loyalty -> B-EXP
#and -> B-EXP
#Marketing -> B-SKILL
#by -> B-EXP
#Segment -> B-EXP
#Hilton -> B-EXP
#Worldwide -> B-SKILL
#General -> B-SKILL
#Manager -> B-SKILL
#Training -> B-SKILL
#Certification -> B-SKILL




#11 convert to huggingface dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# build labels list
unique_labels = set(l for ex in examples for l in ex['labels'])
label_list = sorted(list(unique_labels))
label_list
from datasets import Dataset

def encode_example(ex):
    tokens = ex['tokens']
    labels = ex['labels']
    encoding = tokenizer(tokens, is_split_into_words=True, truncation=True, padding='max_length', max_length=512)
    word_ids = encoding.word_ids()
    label_ids = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            label_ids.append(-100)  # special tokens keep -100
        else:
            label = labels[word_id]
            label_ids.append(label_list.index(label))
    encoding['labels'] = label_ids
    encoding['id'] = ex['id']
    return encoding

hf_dataset = Dataset.from_list(examples)
hf_dataset_enc = hf_dataset.map(lambda x: encode_example(x), batched=False, remove_columns=hf_dataset.column_names)
hf_dataset_enc = hf_dataset_enc.train_test_split(test_size=0.1)
train_ds = hf_dataset_enc['train']
eval_ds = hf_dataset_enc['test']

print(train_ds[0].keys())




#12 define the model and training arguments
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
training_args = TrainingArguments(
    output_dir="/content/models/resume-ner",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_eval=True,
    save_strategy="epoch",
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=50,
    push_to_hub=False,
    report_to="none"   # 🚀 disables wandb/logging

)


#13 metrics function
#!pip install -U --force-reinstall datasets evaluate seqeval
import numpy as np
from datasets import load_metric
metric = load_metric("seqeval")

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
    # return basic f1
    return {"overall_f1": results.get("overall_f1",0)}


#14 define the trainer
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
trainer.save_model("/content/models/resume-ner")
tokenizer.save_pretrained("/content/models/resume-ner")
#result
# Training Loss
# 50	1.026000
# 100	0.802400
# 150	0.632100
# 200	0.581800
# 250	0.443000
# 300	0.331700
# 350	0.253500
# 400	0.250100
# 450	0.217800
# 500	0.187500
# 550	0.161200
# 600	0.146900
# 650	0.144100
# 700	0.132700
# 750	0.126700
# 800	0.121400
# 850	0.120100
# 900	0.116100
# 950	0.102300
# 1000	0.105800
# 1050	0.099000
# 1100	0.099700
# 1150	0.092600
# 1200	0.081900
# 1250	0.091200
# 1300	0.088600
# 1350	0.080300
# 1400	0.075500
# 1450	0.067200
# 1500	0.076000
# 1550	0.081200
# 1600	0.071600
# 1650	0.061600
# Saved model and tokenizer to (('/content/models/resume-ner/tokenizer_config.json', 
# '/content/models/resume-ner/special_tokens_map.json', 
# '/content/models/resume-ner/vocab.txt', 
# '/content/models/resume-ner/added_tokens.json', 
# '/content/models/resume-ner/tokenizer.json'))


#15 inference function
from collections import defaultdict
import torch
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def infer_resume(text):
    # Tokenize words first
    words = [w for w in re.split(r'\s+', text) if w]
    
    # Keep the batch encoding to access word_ids
    enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Save word_ids before moving to tensors
    word_ids = enc.word_ids(batch_index=0)
    
    # Move tensors to device
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model(**enc)
    
    preds = torch.argmax(out.logits, dim=2).squeeze().tolist()
    
    result = defaultdict(list)
    current = None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        label = label_list[preds[idx]]
        token = words[wid]
        
        if label == 'O':
            current = None
            continue
        
        if label.startswith('B-'):
            ent = label.split('-',1)[1]
            result[ent].append(token)
            current = ent
        elif label.startswith('I-') and current is not None:
            result[current].append(token)
        else:
            current = None
    
    out_json = {}
    for ent, toks in result.items():
        joined = " ".join(toks)
        if ent == 'SKILL':
            found = []
            low = joined.lower()
            for sk in GENERAL_SKILLS:
                if sk in low:
                    found.append(sk)
            out_json['skills'] = list(set(found)) if found else [joined]
        else:
            out_json[ent.lower()] = joined

    return out_json

# Example
sample_text = df['resume_text'].iloc[0]
print(infer_resume(sample_text))

# Example output
{'skills': ['hr', 'c', 'conflict resolution', 'customer service', 'go', 'rds', 'lua', 'sem', 'leadership', 'ar', 'marketing', 'r', 'oci', 'time management', 'san', 'sales', 'soc'], 'exp': 'in Hospitality Hospitality and Management. Management. and of teams; teams; to instill instill instill a to Highlights Highlights on Team management savvy savvy savvy and development Skilled Skilled Skilled DOT DOT by IHG IHG IHG in Loyalty Loyalty Loyalty and by Segment Segment Hilton hospitality hospitality systems as Hilton OnQ OnQ , PMS PMS , Fidelio Fidelio System Holidex Holidex Holidex and in 2013 to Name － , State Helps Helps to develop and as employment, employment, benefits, benefits, and employee employee and Keeps Keeps of benefits plans as and pension plan, plan, as and and employee Advises Advises Advises management in of employee issues. issues. benefits as life, life, health, health, dental, dental, pension plans, plans, leave, leave, leave of and employee Designed Designed and meetings, meetings,'}

#16 inference with traduction 
from collections import defaultdict
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import torch
import pandas as pd
import re

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load translation model
mt_model_name = "Helsinki-NLP/opus-mt-fr-en"
mt_tokenizer = AutoTokenizer.from_pretrained(mt_model_name, use_fast=True)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(mt_model_name).to(device)

# Load NER model and tokenizer
ner_model_name = "dslim/bert-base-NER"  # Replace with your specific NER model
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name, use_fast=True)
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)
ner_model.eval()



def translate_fr_to_en(text):
    try:
        if detect(text) == 'fr':
            inputs = mt_tokenizer([text], return_tensors="pt", padding=True).to(device)
            translated_tokens = mt_model.generate(**inputs, max_length=512)
            return mt_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return text
    except:
        return text

def infer_resume(text, ner_tokenizer=ner_tokenizer, ner_model=ner_model, label_list=label_list, general_skills=GENERAL_SKILLS):
    # Tokenize words first
    words = [w for w in re.split(r'\s+', text) if w]
    
    # Keep the batch encoding to access word_ids
    enc = ner_tokenizer(words, is_split_into_words=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Save word_ids before moving to tensors
    word_ids = enc.word_ids(batch_index=0)
    
    # Move tensors to device
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = ner_model(**enc)
    
    preds = torch.argmax(out.logits, dim=2).squeeze().tolist()
    
    result = defaultdict(list)
    current = None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            continue
        label = label_list[preds[idx]]
        token = words[wid]
        
        if label == 'O':
            current = None
            continue
        
        if label.startswith('B-'):
            ent = label.split('-', 1)[1]
            result[ent].append(token)
            current = ent
        elif label.startswith('I-') and current is not None:
            result[current].append(token)
        else:
            current = None
    
    out_json = {}
    for ent, toks in result.items():
        joined = " ".join(toks)
        if ent == 'SKILL':
            found = []
            low = joined.lower()
            for sk in general_skills:
                if sk in low:
                    found.append(sk)
            out_json['skills'] = list(set(found)) if found else [joined]
        else:
            out_json[ent.lower()] = joined

    return out_json

# Example usage
sample_text = df['resume_text'].iloc[0]  # Assuming df is defined
translated_text = translate_fr_to_en(sample_text)
resume_json = infer_resume(translated_text)
print(resume_json)

#safe inference function
def infer_resume_safe(text, chunk_size=200, ner_tokenizer=ner_tokenizer, ner_model=ner_model, label_list=label_list, general_skills=GENERAL_SKILLS):
    lines = re.split(r'[\n\.]', text)  # split by line or sentence
    all_results = defaultdict(list)

    for line in lines:
        words = [w for w in re.split(r'\s+', line) if w]
        if not words:
            continue

        enc = ner_tokenizer(words, is_split_into_words=True, truncation=True, max_length=chunk_size, return_tensors='pt')
        word_ids = enc.word_ids(batch_index=0)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = ner_model(**enc)
        preds = torch.argmax(out.logits, dim=2).squeeze().tolist()

        current = None
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            label = label_list[preds[idx]]
            token = words[wid]

            if label == 'O':
                current = None
                continue
            if label.startswith('B-'):
                ent = label.split('-',1)[1]
                all_results[ent].append(token)
                current = ent
            elif label.startswith('I-') and current is not None:
                all_results[current].append(token)
            else:
                current = None

    # Postprocess results
    out_json = {}
    for ent, toks in all_results.items():
        joined = " ".join(toks)
        if ent == 'SKILL':
            found = [sk for sk in general_skills if sk.lower() in joined.lower()]
            out_json['skills'] = list(set(found)) if found else [joined]
        else:
            out_json[ent.lower()] = joined

    return out_json


#testing with my cv 
#!pip install PyPDF2 -q

import PyPDF2

# Open your PDF file
pdf_path = "/content/aziz.pdf"
pdf_file = open(pdf_path, "rb")
reader = PyPDF2.PdfReader(pdf_file)

# Extract text from all pages
pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text() + "\n"

pdf_file.close()

print("First 500 chars of extracted text:\n", pdf_text[:500], "...\n")

# Then you can feed this to your existing pipeline
resume_json = infer_resume_safe(pdf_text)

print("Extracted resume info:\n", resume_json)
