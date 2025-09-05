#1 install required packages
#!unzip -q /content/archive.zip -d /content/
BASE = "/content/Resume"
#!ls -R "$BASE"
# Install required libraries
#!pip install -q PyPDF2 python-docx pandas numpy transformers datasets seqeval torch torchvision torchaudio
#!pip install -q sentencepiece sacremoses langdetect evaluate
#!pip install -q scikit-learn tqdm
#!pip install docx2txt
#!pip install -q pdfminer.six fuzzywuzzy python-Levenshtein
#!pip install -q unstructured[pdf,docx]  # Better text extraction
#!pip install -q sentence-transformers  # For better similarity matching
 


#2&3 load  and clean resume data
# Define paths
# Remove duplicate imports; keep only one set
import os, re, json, math, random
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from pdfminer.high_level import extract_text
import docx2txt
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
from evaluate import load
from collections import defaultdict

# Define paths with error checking
BASE_PATH = "/content/Resume"
CSV_PATH = f"{BASE_PATH}/Resume.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}. Check unzip.")

# Load CSV with try-except
try:
    df = pd.read_csv(CSV_PATH, encoding='utf-8', dtype=str, low_memory=False)
    df = df.fillna('')  # Replace NaN with empty strings
    print(f"Number of rows: {len(df)}")
    print(df.head(2).T)
except Exception as e:
    raise ValueError(f"Error loading CSV: {e}")

# Clean text function (unchanged, it's fine)
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
import re
import difflib
from tqdm import tqdm
from collections import defaultdict, Counter
import time
from datasets import load_dataset
import json

entity_types = ['NAME', 'EMAIL', 'PHONE', 'SKILL', 'EXP', 'EDU', 'PROJECT', 'CERT', 'AGE']
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]
label_list = sorted(set(label_list))
valid_labels = set(label_list)

EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_RE = re.compile(r'\b(?:\+?\d{1,3}\s?)?(?:\(\d{3}\)\s?|\d{3}[-.\s]?)?\d{3}[-.\s]?\d{4}\b')
NAME_RE = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?\b', re.I)
DATE_RE = re.compile(r'\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})\b', re.I)
AGE_RE = re.compile(r'\b(?:age\s*\d{1,2}|\d{1,2}\s*years?\s*old|born\s*(?:\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}))\b', re.I)
CERT_RE = re.compile(r'\b(?:Nvidia\s+DLI|Microsoft\s+Certified|Azure\s+Data\s+Fundamentals|KodeKloud|Docker\s+Training|Kubernetes\s+Challenges|Workshop\s*/\s*[A-Za-z\s-]+|Certified\s+[A-Za-z\s-]+|Certificate\s+[A-Za-z\s-]+)\b', re.I)

SECTION_PATTERNS = {
    'summary': [re.compile(r'\b(summary|professional summary|profile|about|objective|career objective)\b', re.I)],
    'experience': [re.compile(r'\b(experience|work experience|professional experience|work history|employment|career history|job history|stage|internship)\b', re.I)],
    'education': [re.compile(r'\b(education|academic background|qualifications|academic qualifications|degrees|diplôme|cycle)\b', re.I)],
    'skills': [re.compile(r'\b(skills|technical skills|competencies|expertise|technical proficiencies|abilities|compétences|programming languages|technologies)\b', re.I)],
    'projects': [re.compile(r'\b(projects|personal projects|portfolio|key projects|work projects|projets|pfa)\b', re.I)],
    'certifications': [re.compile(r'\b(certifications|achievements|highlights|accomplishments|certificates|credentials|awards|certificats)\b', re.I)],
}

def estimate_age_from_date(date_str, current_year=2025):
    try:
        year_match = re.search(r'\d{4}', date_str)
        if year_match:
            year = int(year_match.group())
            if 1900 < year < current_year - 15:
                return current_year - year
        return None
    except:
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\ue316\ue258]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

from multiprocessing import Pool

def weak_label_text(args):
    text, resume_id = args
    start_time = time.time()
    if not text or not isinstance(text, str):
        print(f"ID: {resume_id} - Skipping: Invalid or empty text")
        return [], []
    
    if len(text) > 100000:
        text = text[:100000]
    
    lines = re.split(r'\n+', text)
    tokens = []
    labels = []
    current_section = None
    stop_words = set([
        'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'profile',
        'objective', 'dedicated', 'customer', 'service', 'manager', 'hospitality', 'versatile',
        'media', 'professional', 'communications', 'marketing', 'human', 'resources', 'technology',
        'hr', 'us', 'operations', 'engineer', 'intern', 'data', 'cloud', 'web', 'development',
        'computing', 'artificial', 'intelligence', 'ai', 'ml', 'big', 'dashboard', 'keywords',
        'strategically', 'grounded', 'highly', 'experienced', 'across', 'multiple', 'disciplines',
        'including', 'program', 'director', 'office', 'purpose'
    ])
    name_found = False

    for line_idx, line in enumerate(lines):
        line = clean_text(line)
        if not line:
            continue
        
        lower_line = line.lower()
        is_header = False
        for sec, patterns in SECTION_PATTERNS.items():
            if any(pattern.search(lower_line) for pattern in patterns) and len(line.split()) <= 6:
                current_section = sec
                print(f"ID: {resume_id} - Detected section: {current_section}")
                is_header = True
                break
        if is_header:
            continue
        
        words = re.split(r'\s+', line)
        word_idx = 0
        while word_idx < len(words):
            word = words[word_idx].strip()
            if not word or len(word) > 50:
                word_idx += 1
                continue
            
            matched = False
            
            candidate = ' '.join(words[word_idx:word_idx+5])
            if EMAIL_RE.search(candidate):
                email = EMAIL_RE.search(candidate).group()
                tokens.append(email)
                labels.append('B-EMAIL')
                print(f"ID: {resume_id} - Found email: {email}")
                matched = True
                word_idx += len(email.split())
                continue
            elif PHONE_RE.search(candidate) and not DATE_RE.search(candidate):
                phone = PHONE_RE.search(candidate).group()
                tokens.append(phone)
                labels.append('B-PHONE')
                print(f"ID: {resume_id} - Found phone: {phone}")
                matched = True
                word_idx += len(phone.split())
                continue
            
            age_span = ' '.join(words[word_idx:word_idx+4])
            if AGE_RE.match(age_span.lower()):
                age_num = estimate_age_from_date(age_span)
                if age_num:
                    tokens.append(str(age_num))
                    labels.append('B-AGE')
                    if 'years' in age_span.lower() or 'old' in age_span.lower():
                        tokens.append('years old')
                        labels.append('I-AGE')
                    print(f"ID: {resume_id} - Found age: {age_num}")
                    matched = True
                    word_idx += len(age_span.split())
                    continue
            
            if line_idx == 0 and not name_found:
                name_span = ' '.join(words[word_idx:word_idx+3])  # Increase to 3 words
                if NAME_RE.match(name_span):
                    name_words = name_span.split()
                    if 1 <= len(name_words) <= 3 and not any(w.lower() in stop_words for w in name_words):
                        print(f"ID: {resume_id} - Found name: {name_span}")
                        for i, nw in enumerate(name_words):
                            tokens.append(nw)
                            labels.append('B-NAME' if i == 0 else 'I-NAME')
                        name_found = True
                        matched = True
                        word_idx += len(name_words)
                        continue
            
            candidate = ' '.join(words[word_idx:word_idx+3]).lower()
            skill_words = candidate.split()
            matched_skill = None
            for i in range(1, len(skill_words) + 1):
                sub_candidate = ' '.join(skill_words[:i])
                if sub_candidate in IT_SKILLS:
                    matched_skill = sub_candidate
                    break
            if matched_skill:
                skill_words = matched_skill.split()
                for i, sw in enumerate(skill_words):
                    tokens.append(words[word_idx + i] if i < len(words) - word_idx else sw)
                    labels.append('B-SKILL' if i == 0 else 'I-SKILL')
                print(f"ID: {resume_id} - Found skill: {matched_skill}")
                matched = True
                word_idx += len(skill_words)
                continue
            
            if current_section == 'certifications' or CERT_RE.search(lower_line):
                cert_span = ' '.join(words[word_idx:word_idx+10])
                if CERT_RE.search(cert_span.lower()):
                    cert_words = cert_span.split()
                    for i, cw in enumerate(cert_words):
                        tokens.append(words[word_idx + i] if i < len(words) - word_idx else cw)
                        labels.append('B-CERT' if i == 0 else 'I-CERT')
                    print(f"ID: {resume_id} - Found cert: {cert_span}")
                    matched = True
                    word_idx += len(cert_words)
                    continue
            
            if current_section in ['experience', 'education', 'projects']:
                ent_type = current_section.upper()[:4]
                title_end = line.find(':') if ':' in line else min(20, len(line) // 2)
                title = line[:title_end].strip()
                desc = line[title_end:].strip()
                title_words = title.split()
                if title_words:
                    for i, w in enumerate(title_words):
                        tokens.append(w)
                        labels.append(f'B-{ent_type}' if i == 0 else f'I-{ent_type}')
                desc_words = desc.split()
                for w in desc_words:
                    tokens.append(w)
                    labels.append(f'I-{ent_type}')
                print(f"ID: {resume_id} - Labeled {ent_type}: {title}")
                matched = True
                word_idx += len(words)
                continue
            
            if not matched:
                tokens.append(word)
                labels.append('O')
            
            word_idx += 1
        
        if time.time() - start_time > 10:
            print(f"ID: {resume_id} - Timeout")
            break
    
    if len(tokens) != len(labels):
        print(f"ID: {resume_id} - Error: Token-label mismatch (tokens: {len(tokens)}, labels: {len(labels)})")
        return [], []
    
    return tokens, labels

# Initialize examples list before the loop
examples = []

# In the resume processing loop
with Pool(processes=4) as pool:  # Adjust based on Colab CPU
    args = [(row['Resume_str'], str(row['ID'])) for idx, row in df.iterrows()]
    results = pool.map(weak_label_text, args)
    for (tokens, labels), (idx, row) in zip(results, df.iterrows()):
        resume_id = str(row['ID'])
        if not tokens or not labels:
            continue
        if len(tokens) != len(labels):
            print(f"ID: {resume_id} - Skipped: Mismatched lengths")
            continue
        invalid_labels = set(labels) - valid_labels
        if invalid_labels:
            print(f"ID: {resume_id} - Skipped: Invalid labels {invalid_labels}")
            continue
        examples.append({'id': resume_id, 'tokens': tokens, 'labels': labels, 'raw_text': clean_text(' '.join(tokens))})

def load_census_ner():
    try:
        dataset = load_dataset("Josephgflowers/CENSUS-NER-Name-Email-Address-Phone")['train']
        print(f"CENSUS-NER sample: {dataset[0]}")  # Debug dataset structure
        return dataset
    except Exception as e:
        print(f"Error loading CENSUS-NER: {e}")
        return []

def convert_census_to_examples(census_dataset, max_examples=10000):
    new_examples = []
    for idx, ex in enumerate(tqdm(census_dataset[:max_examples], desc="Processing CENSUS-NER")):
        if isinstance(ex, dict):
            text = ex.get('user', '')
            entities_str = ex.get('assistant', '{}')  # Default to empty JSON if missing
            try:
                entities = json.loads(entities_str)
                tokens = text.split()
                labels = ['O'] * len(tokens)
                for ent_type, value in entities.items():
                    if not value or value == 'nan':
                        continue
                    ent_type = ent_type.lower().replace('_', '').replace('number', '')  # Normalize to match label_list
                    if ent_type in ['name', 'email', 'phone']:
                        ent_label = f'B-{ent_type.upper()[:4]}'
                        for i, token in enumerate(tokens):
                            if value.lower() in ' '.join(tokens[i:i+5]).lower():
                                labels[i] = ent_label
                                if i + 1 < len(labels) and tokens[i + 1].lower() in value.lower():
                                    labels[i + 1] = f'I-{ent_type.upper()[:4]}'
                                print(f"CENSUS-NER ID: census_{idx} - Found {ent_type}: {value}")
                                break
            except json.JSONDecodeError:
                print(f"CENSUS-NER ID: census_{idx} - Invalid JSON in assistant: {entities_str}")
                tokens = text.split()
                labels = ['O'] * len(tokens)
        else:
            print(f"CENSUS-NER ID: census_{idx} - Received string, using regex fallback")
            text = ex
            tokens = text.split()
            labels = ['O'] * len(tokens)
            for i in range(len(tokens)):
                candidate = ' '.join(tokens[i:i+5])
                if NAME_RE.match(candidate):
                    name_tokens = candidate.split()
                    if 1 <= len(name_tokens) <= 2:
                        labels[i] = 'B-NAME'
                        for j in range(1, len(name_tokens)):
                            if i+j < len(labels):
                                labels[i+j] = 'I-NAME'
                        print(f"CENSUS-NER ID: census_{idx} - Found name: {candidate}")
                        break
                if EMAIL_RE.search(candidate):
                    email = EMAIL_RE.search(candidate).group()
                    email_tokens = email.split()
                    labels[i] = 'B-EMAIL'
                    for j in range(1, len(email_tokens)):
                        if i+j < len(labels):
                            labels[i+j] = 'I-EMAIL'
                    print(f"CENSUS-NER ID: census_{idx} - Found email: {email}")
                    break
                if PHONE_RE.search(candidate) and not DATE_RE.search(candidate):
                    phone = PHONE_RE.search(candidate).group()
                    phone_tokens = phone.split()
                    labels[i] = 'B-PHONE'
                    for j in range(1, len(phone_tokens)):
                        if i+j < len(labels):
                            labels[i+j] = 'I-PHONE'
                    print(f"CENSUS-NER ID: census_{idx} - Found phone: {phone}")
                    break
        
        if len(tokens) == len(labels) and tokens:
            new_examples.append({
                'id': f"census_{idx}",
                'tokens': tokens,
                'labels': labels,
                'raw_text': clean_text(text)
            })
    
    return new_examples

census_dataset = load_census_ner()
if census_dataset:
    census_examples = convert_census_to_examples(census_dataset)
    examples.extend(census_examples)
    print(f"Added {len(census_examples)} examples from CENSUS-NER")

print("Total number of examples:", len(examples))
for ex in examples[:3]:
    print("ID:", ex['id'])
    for t, l in list(zip(ex['tokens'][:30], ex['labels'][:30])):
        print(f"{t} -> {l}")
    print("-----\n")
unique_labels = set(l for ex in examples for l in ex['labels'])
print("Unique labels found:", unique_labels)
label_counts = Counter(l for ex in examples for l in ex['labels'])
print("Label distribution:", dict(label_counts))

# 6 encode function

from transformers import AutoTokenizer
import torch
from datasets import Dataset
import numpy as np
from tqdm import tqdm

MODEL_NAME = "xlm-roberta-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

entity_types = ['NAME', 'EMAIL', 'PHONE', 'SKILL', 'EXP', 'EDU', 'PROJECT', 'CERT', 'AGE']
label_list = ['O'] + [f'B-{et}' for et in entity_types] + [f'I-{et}' for et in entity_types]
label_list = sorted(set(label_list))
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
print("Label list:", label_list)

def encode_example(examples, batch_size=32):
    tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': [], 'id': []}
    
    for i in tqdm(range(0, len(examples['tokens']), batch_size), desc="Encoding examples"):
        batch_tokens = examples['tokens'][i:i+batch_size]
        batch_labels = examples['labels'][i:i+batch_size]
        batch_ids = examples['id'][i:i+batch_size]
        
        try:
            encodings = tokenizer(
                batch_tokens,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
        except Exception as e:
            print(f"Error in tokenizer for batch {i}-{i+batch_size}: {e}")
            continue
        
        for j, (tokens, labels, ex_id) in enumerate(zip(batch_tokens, batch_labels, batch_ids)):
            if len(tokens) != len(labels):
                print(f"ID: {ex_id} - Skipping: Mismatched token/label lengths ({len(tokens)} vs {len(labels)})")
                continue
            if not tokens:
                print(f"ID: {ex_id} - Skipping: Empty tokens")
                continue
                
            word_ids = encodings.word_ids(batch_index=j)
            label_ids = np.full(512, -100, dtype=np.int64)
            
            previous_word_idx = None
            for idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                try:
                    current_label = labels[word_idx]
                    if current_label not in label2id:
                        print(f"ID: {ex_id} - Invalid label at word {word_idx}: {current_label}")
                        current_label = 'O'
                    if word_idx != previous_word_idx:
                        label_ids[idx] = label2id[current_label]
                    else:
                        if current_label.startswith('B-'):
                            i_label = f'I-{current_label[2:]}'
                            label_ids[idx] = label2id.get(i_label, label2id['O'])
                        else:
                            label_ids[idx] = label2id.get(current_label, label2id['O'])
                except IndexError:
                    print(f"ID: {ex_id} - IndexError at word_idx {word_idx}, len(labels): {len(labels)}")
                    label_ids[idx] = label2id['O']
                previous_word_idx = word_idx
            
            tokenized_inputs['input_ids'].append(encodings['input_ids'][j])
            tokenized_inputs['attention_mask'].append(encodings['attention_mask'][j])
            tokenized_inputs['labels'].append(torch.tensor(label_ids))
            tokenized_inputs['id'].append(ex_id)
            
            if ex_id.startswith('census_') and i < 10:
                print(f"Encoded CENSUS-NER example {ex_id}: {tokens[:20]} -> {labels[:20]}")
    
    try:
        tokenized_inputs['input_ids'] = torch.stack(tokenized_inputs['input_ids'])
        tokenized_inputs['attention_mask'] = torch.stack(tokenized_inputs['attention_mask'])
        tokenized_inputs['labels'] = torch.stack(tokenized_inputs['labels'])
    except Exception as e:
        print(f"Error stacking tensors: {e}")
        return {'input_ids': [], 'attention_mask': [], 'labels': [], 'id': []}
    
    return tokenized_inputs

# Validate examples
print(f"Total examples before encoding: {len(examples)}")
valid_examples = [ex for ex in examples if len(ex['tokens']) == len(ex['labels']) and ex['tokens']]
print(f"Valid examples: {len(valid_examples)}")

# Sample CENSUS-NER if too large
max_census_examples = 10000
if len(valid_examples) > 100000:
    from random import sample
    resume_examples = [ex for ex in valid_examples if not ex['id'].startswith('census_')]
    census_examples = [ex for ex in valid_examples if ex['id'].startswith('census_')]
    census_examples = sample(census_examples, min(max_census_examples, len(census_examples)))
    valid_examples = resume_examples + census_examples
    print(f"Sampled {len(census_examples)} CENSUS-NER examples, total examples: {len(valid_examples)}")

hf_dataset = Dataset.from_list(valid_examples)
try:
    hf_dataset_enc = hf_dataset.map(
        encode_example,
        batched=True,
        batch_size=32,
        remove_columns=hf_dataset.column_names,
        num_proc=4
    )
    hf_dataset_enc = hf_dataset_enc.train_test_split(test_size=0.1, seed=42)
    train_ds = hf_dataset_enc['train']
    eval_ds = hf_dataset_enc['test']
    print("Train dataset size:", len(train_ds), "Eval dataset size:", len(eval_ds))
except Exception as e:
    print(f"Error in dataset mapping or splitting: {e}")
    raise


#7  weightedtrainer 
from sklearn.utils import resample
from collections import Counter
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Oversample rare labels (adjust based on new dataset)
balanced_examples = []
rare_labels = ['B-PHONE', 'B-EMAIL', 'B-NAME']  # Add B-NAME since it was weak
for label in rare_labels:
    label_examples = [ex for ex in examples if label in ex['labels']]
    if len(label_examples) < 2000:  # Increase target to account for CENSUS-NER
        label_examples = resample(label_examples, n_samples=2000, random_state=42)
    balanced_examples.extend(label_examples)
balanced_examples.extend([ex for ex in examples if not any(l in ex['labels'] for l in rare_labels)])
examples = balanced_examples

# Compute weights
all_labels = [l for ex in examples for l in ex['labels']]
label_counts = Counter(all_labels)
total = len(all_labels)
weights = []
for label in label_list:
    count = label_counts.get(label, 1)
    weight = total / (len(label_list) * count)
    weight = min(weight, 5.0)  # Cap for stability
    weights.append(weight)
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






#8 model training
from transformers import AutoModelForTokenClassification, TrainingArguments
import os

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="/content/models/resume-ner-v9",  # New directory for new model
    num_train_epochs=7,  # Increase epochs for larger dataset
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,  # Slightly higher for faster convergence
    weight_decay=0.1,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    push_to_hub=False,
    report_to="none",
    warmup_ratio=0.1,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    fp16=True
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

os.environ["WANDB_DISABLED"] = "true"
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("/content/models/resume-ner-v9")
tokenizer.save_pretrained("/content/models/resume-ner-v9")

#9 translation and extraction functions
#!pip install googletrans==3.1.0a0
from googletrans import Translator
def translate_to_en(text, pdf_path):
    try:
        if not text.strip():
            return text
        lang = detect(text[:500])
        if 'sana' in pdf_path.lower() or 'yasmine' in pdf_path.lower():
            lang = 'fr'
        print(f"Detected/Forced language: {lang}")
        if lang == 'fr':
            translator = Translator()
            words = text.split()
            chunks = [' '.join(words[i:i+500]) for i in range(0, len(words), 500)]
            translated_chunks = [translator.translate(chunk, src='fr', dest='en').text for chunk in chunks]
            translated_text = clean_text(' '.join(translated_chunks))
            print(f"Translated text (first 100 chars): {translated_text[:100]}")
            return translated_text
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def extract_text_from_file(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text(file_path)
        elif file_path.lower().endswith('.docx'):
            text = docx2txt.process(file_path)
        else:
            raise ValueError("Use PDF or DOCX.")
        text = clean_text(text)
        print(f"Extracted text (first 100 chars): {text[:100]}")
        return text
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return ""

#10 inference function
import json
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re
import difflib

def infer_resume(text, model_path="/content/models/resume-ner-v9", chunk_size=300):
    if not text.strip():
        return {}

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading: {e}")
        return {}

    words = re.split(r'\s+', text.strip())
    print(f"Input words count: {len(words)}")
    if len(words) == 0:
        return {}

    all_entities = defaultdict(list)

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        print(f"Chunk {i//chunk_size + 1}: {len(chunk_words)} words")
        if not chunk_words:
            continue

        enc = tokenizer(chunk_words, is_split_into_words=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        word_ids = enc.word_ids(batch_index=0)

        with torch.no_grad():
            logits = model(**enc).logits

        preds = torch.argmax(logits, dim=2)[0].cpu().numpy()
        pred_labels = [id2label.get(p, 'O') for p in preds]
        print(f"Sample preds: {pred_labels[:20]}")

        current_ent = None
        current_tokens = []
        previous_wid = None
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid != previous_wid and current_ent and current_tokens:
                cleaned_tokens = [t.replace('▁', '').replace('##', '') for t in current_tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
                if cleaned_tokens:
                    all_entities[current_ent].append(" ".join(cleaned_tokens))
                current_tokens = []
            token = tokenizer.convert_ids_to_tokens(enc['input_ids'][0][idx].item())
            if not token.startswith("##") and token not in ['[CLS]', '[SEP]', '[PAD]']:
                current_tokens.append(token)
            elif token.startswith("##") and current_tokens:
                current_tokens[-1] += token[2:]

            label = id2label.get(preds[idx], 'O')
            if label == 'O':
                current_ent = None
            elif label.startswith('B-'):
                current_ent = label[2:]
            elif label.startswith('I-') and current_ent == label[2:]:
                pass
            else:
                current_ent = None

            previous_wid = wid

        if current_ent and current_tokens:
            cleaned_tokens = [t.replace('▁', '').replace('##', '') for t in current_tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
            if cleaned_tokens:
                all_entities[current_ent].append(" ".join(cleaned_tokens))

    out_json = {
        'name': [],
        'email': [],
        'phone': [],
        'skills': [],
        'cert': [],
        'education': [],
        'experience': [],
        'projects': [],
        'age': []
    }
    stop_words = set([
        'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'profile', 
        'objective', 'data', 'intern', 'cloud', 'web', 'development', 'engineer', 'keywords', 
        'computing', 'artificial', 'intelligence', 'ai', 'ml', 'big', 'dashboard'
    ])
    print(f"Raw entities before filtering: {dict(all_entities)}")
    for ent, spans in all_entities.items():
        unique_spans = set(spans)
        if ent == 'SKILL':
            skills = set()
            for span in unique_spans:
                span_lower = span.lower()
                if span_lower in stop_words or not span_lower.strip():
                    continue
                best_match = max(GENERAL_SKILLS, key=lambda s: difflib.SequenceMatcher(None, span_lower, s.lower()).ratio() * 100, default=span)
                if difflib.SequenceMatcher(None, span_lower, best_match.lower()).ratio() * 100 > 75:
                    skills.add(best_match)
                else:
                    skills.add(span)
            out_json['skills'] = list(skills)
        elif ent == 'NAME':
            valid_names = [span for span in unique_spans if len(span.split()) >= 2 and not any(w.lower() in stop_words for w in span.split())]
            out_json['name'] = valid_names[:1] or list(unique_spans)[:1]
        elif ent == 'EMAIL':
            out_json['email'] = [span for span in unique_spans if EMAIL_RE.match(span)] or []
        elif ent == 'PHONE':
            out_json['phone'] = [span for span in unique_spans if PHONE_RE.match(span)] or []
        elif ent == 'AGE':
            out_json['age'] = [int(span) for span in unique_spans if span.isdigit() and 16 <= int(span) <= 100]
        else:
            out_json[ent.lower()] = list(unique_spans)
    
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    if not out_json.get('email'):
        out_json['email'] = emails
    if not out_json.get('phone'):
        out_json['phone'] = phones
    
    print(f"Extracted entities: {out_json}")
    return out_json




#11 testing inference
def rule_based_parser(text):
    out = {
        'name': [],
        'email': [],
        'phone': [],
        'skills': [],
        'cert': [],
        'education': [],
        'experience': [],
        'projects': [],
        'age': []
    }
    lines = re.split(r'\n+', text)
    current_section = None
    stop_words = set([
        'summary', 'experience', 'education', 'skills', 'projects', 'certifications', 'profile',
        'objective', 'data', 'intern', 'cloud', 'web', 'development', 'engineer', 'keywords',
        'computing', 'artificial', 'intelligence', 'ai', 'ml', 'big', 'dashboard', 'summa',
        'ry', 'élève', 'ingénieur', 'ingénieure', 'je', 'suis', 'étudiante', 'et'
    ])
    name_found = False
    
    for line_idx, line in enumerate(lines):
        line = clean_text(line)
        if not line:
            continue
        lower_line = line.lower()
        
        # Detect section
        for sec, patterns in SECTION_PATTERNS.items():
            if any(pattern.search(lower_line) for pattern in patterns) and len(line.split()) <= 6:
                current_section = sec
                print(f"Detected section in rule-based parser: {current_section}")
                break
        
        # Name detection (first line only, improved logic)
        if line_idx == 0 and not name_found:
            words = line.split()
            name_span = ' '.join(words[:2])  # Try 1-2 words
            if NAME_RE.match(name_span) and not any(w.lower() in stop_words for w in words[:2]):
                out['name'].append(name_span)
                name_found = True
                print(f"Found name: {name_span}")
            elif len(words) > 2:  # Try 2-3 words if 2 fails
                name_span = ' '.join(words[:3])
                if NAME_RE.match(name_span) and not any(w.lower() in stop_words for w in words[:3]):
                    out['name'].append(name_span)
                    name_found = True
                    print(f"Found name: {name_span}")
        
        # Email and phone
        if EMAIL_RE.search(line):
            emails = EMAIL_RE.findall(line)
            out['email'].extend(emails)
            print(f"Found email: {emails}")
        if PHONE_RE.search(line) and not DATE_RE.search(line):
            phones = PHONE_RE.findall(line)
            out['phone'].extend([re.sub(r'[^\d+]', '', p) for p in phones])  # Normalize to digits only
            print(f"Found phone: {phones}")
        
        # Skills (anywhere, using IT_SKILLS)
        for skill in IT_SKILLS:
            if skill.lower() in lower_line:
                out['skills'].append(skill)
                print(f"Found skill: {skill}")
        
        # Certifications
        if current_section == 'certifications' or CERT_RE.search(lower_line):
            certs = CERT_RE.findall(line)
            out['cert'].extend(certs)
            print(f"Found cert: {certs}")
        
        # Education, Experience, Projects
        if current_section in ['education', 'experience', 'projects']:
            out[current_section].append(line)
            print(f"Added to {current_section}: {line}")
    
    return out

# Test loop
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
    result = infer_resume(translated_text, model_path="/content/models/resume-ner-v9")

    if not result.get('name') or not result.get('skills') or not result.get('phone'):
        print("Falling back to rule-based parser")
        result = rule_based_parser(translated_text)

    print(f"\nExtracted entities from {pdf_path}:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{'='*60}\n")