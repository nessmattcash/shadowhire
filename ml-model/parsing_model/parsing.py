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

