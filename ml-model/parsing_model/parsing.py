#how i create a parsing model
#1 installations
#!pip install  datasets pdfplumber pdfminer.six fuzzywuzzy python-Levenshtein torch accelerate -U -q
#!pip install spacy -q  # For data prep (optional, but helpful)
#!python -m spacy download en_core_web_sm -q  # English model
#!python -m spacy download fr_core_news_sm -q  # French model (since CVs have French)
#import nltk
#nltk.download('punkt', quiet=True)  # For sentence tokenization
#!pip install evaluate seqeval -q


#2 unzip dataset (on google colab)
#!unzip -q /content/aziz.zip -d /content/
#BASE = "/content"
#!ls -R "$BASE"

#3 imports and regex patterns
import json
import os
import re
import logging
from collections import defaultdict
from unicodedata import normalize
from typing import Dict, List, Any, Tuple
from fuzzywuzzy import fuzz
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, TrainingArguments, Trainer
from datasets import Dataset
import torch
import spacy  # For annotation processing
from seqeval.metrics import classification_report  # NEW: For NER evaluation


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# Skills lists (expanded with more terms for accuracy, e.g., AI/ML, devops variants, cloud)
IT_SKILLS = set([
    "python", "java", "c++", "c", "c#", "javascript", "typescript", "php", "ruby", "swift",
    "kotlin", "go", "golang", "rust", "scala", "r", "dart", "perl", "haskell", "elixir",
    "clojure", "lua", "matlab", "objective-c", "bash", "shell", "powershell", "groovy",
    "julia", "cobol", "fortran", "lisp", "scheme", "erlang", "f#", "assembly", "vb.net", "vba",
    "html", "html5", "css", "css3", "sass", "scss", "less", "bootstrap", "tailwind css",
    "react", "angular", "vue", "vue.js", "svelte", "next.js", "nuxt.js", "gatsby", "remix",
    "redux", "context api", "vuex", "rxjs", "webpack", "vite", "parcel", "babel", "npm", "yarn",
    "pnpm", "jquery", "web components", "stencil", "three.js", "d3.js", "chart.js", "jest", "cypress",
    "storybook", "material ui", "ant design", "chakra ui", "figma", "sketch", "xd", "webgl",
    "node.js", "express.js", "nest.js", "koa", "django", "flask", "fastapi", "ruby on rails",
    "spring", "spring boot", "micronaut", "quarkus", "laravel", "symfony", "codeigniter", "asp.net",
    "asp.net core", "blazor", "phoenix", "gin", "echo", "fiber", "actix", "rocket", "android", "ios",
    "react native", "flutter", "xamarin", "ionic", "kotlin multiplatform", "swiftui", "jetpack compose",
    "objective-c", "android sdk", "xcode", "appium", "sql", "mysql", "postgresql", "postgres", "sqlite",
    "oracle", "microsoft sql server", "sql server", "mongodb", "cassandra", "redis", "elasticsearch",
    "dynamodb", "cosmos db", "firebase firestore", "realm", "couchbase", "couchdb", "neo4j", "arangodb",
    "influxdb", "snowflake", "bigquery", "redshift", "table design", "database normalization", "indexing",
    "query optimization", "etl", "elt", "aws", "amazon web services", "azure", "microsoft azure",
    "gcp", "google cloud platform", "oracle cloud", "oci", "ibm cloud", "digitalocean", "linode",
    "akamai", "cloudflare", "heroku", "netlify", "vercel", "firebase", "openshift", "vmware",
    "openstack", "docker", "kubernetes", "k8s", "terraform", "ansible", "puppet", "chef",
    "jenkins", "github actions", "gitlab ci", "circleci", "travis ci", "argo cd", "flux", "helm",
    "prometheus", "grafana", "datadog", "splunk", "new relic", "elk stack", "elastic stack",
    "istio", "linkerd", "envoy", "vagrant", "packer", "consul", "vault", "nomad", "nginx",
    "apache", "iis", "linux", "ubuntu", "bash scripting", "shell scripting", "powershell scripting",
    "ec2", "s3", "lambda", "rds", "dynamodb", "iam", "vpc", "route 53", "cloudfront", "sns",
    "sqs", "eventbridge", "api gateway", "elastic beanstalk", "ecs", "eks", "fargate", "cloudformation",
    "cloudwatch", "codebuild", "codepipeline", "codedeploy", "systems manager", "secrets manager",
    "kms", "cognito", "amplify", "appsync", "glue", "athena", "quicksight", "azure vm",
    "azure app service", "azure functions", "azure sql database", "cosmos db", "azure active directory",
    "aad", "azure devops", "arm templates", "azure resource manager", "azure kubernetes service",
    "aks", "azure container instances", "azure storage", "blob storage", "azure monitor",
    "application insights", "azure pipeline", "key vault", "azure cdn", "azure event grid",
    "service bus", "azure data factory", "synapse analytics", "power bi", "compute engine",
    "app engine", "cloud functions", "cloud run", "bigquery", "bigtable", "cloud spanner",
    "cloud sql", "firestore", "cloud storage", "gcs", "vertex ai", "google kubernetes engine",
    "gke", "cloud build", "cloud deployment manager", "iam", "cloud identity", "vpc",
    "cloud load balancing", "stackdriver", "operations", "pub/sub", "dataflow", "dataproc",
    "looker", "looker studio", "machine learning", "ml", "deep learning", "neural networks",
    "natural language processing", "nlp", "computer vision", "cv", "generative ai", "llm",
    "large language models", "tensorflow", "pytorch", "keras", "scikit-learn", "opencv",
    "hugging face", "transformers", "langchain", "llama index", "apache spark", "pyspark",
    "hadoop", "hdfs", "mapreduce", "hive", "pig", "kafka", "kafka streams", "airflow",
    "prefect", "dagster", "dbt", "data analysis", "data visualization", "tableau", "power bi",
    "qlik", "matplotlib", "seaborn", "plotly", "pandas", "numpy", "jupyter", "rstudio", "mlops",
    "tcp/ip", "dns", "dhcp", "http", "https", "ssl", "tls", "vpn", "ipsec", "ssh", "network security",
    "cybersecurity", "application security", "appsec", "devsecops", "owasp", "penetration testing",
    "pentesting", "vulnerability assessment", "siem", "soc", "firewalls", "waf", "ids", "ips",
    "zero trust", "pki", "cryptography", "encryption", "iso 27001", "soc 2", "pci dss", "gdpr",
    "compliance", "risk management", "incident response", "agile", "scrum", "kanban", "devops",
    "devsecops", "gitops", "ci/cd", "continuous integration", "continuous delivery", "continuous deployment",
    "test driven development", "tdd", "bdd", "pair programming", "code review", "refactoring",
    "clean code", "design patterns", "solid principles", "microservices", "monolith", "serverless",
    "rest", "restful api", "graphql", "grpc", "soap", "api design", "event driven architecture",
    "eda", "domain driven design", "ddd", "twelve factor app", "object oriented programming",
    "oop", "functional programming", "fp", "unit testing", "integration testing", "end-to-end testing",
    "e2e testing", "ui testing", "regression testing", "performance testing", "load testing",
    "stress testing", "security testing", "accessibility testing", "a11y", "manual testing",
    "automated testing", "selenium", "cypress", "playwright", "puppeteer", "jest", "mocha",
    "jasmine", "karma", "phpunit", "junit", "testng", "cucumber", "specflow", "soapui",
    "postman", "jmeter", "gatling", "salesforce", "servicenow", "sap", "oracle ebs", "microsoft dynamics",
    "sharepoint", "sitecore", "adobe experience manager", "aem", "workday", "mainframe", "cobol",
    "soa", "esb", "tibco", "webmethods", "mulesoft", "ibm mq", "active directory", "ldap",
    "windows server", "exchange server", "vmware vsphere", "citrix", "san", "nas", "raid",
    "itil", "ticketing systems", "service desk", "help desk", "technical support", "troubleshooting",
    "hardware", "software installation", "network administration", "system administration",
    "windows", "macos", "active directory", "group policy", "mdm", "intune", "jamf", "sccm",
    "backup", "disaster recovery", "bcdr", "patch management", "remote support", "voip",
    "embedded systems", "arduino", "raspberry pi", "iot", "internet of things", "fpga", "verilog",
    "vhdl", "rtos", "real-time operating system", "device drivers", "kernel development",
    "assembly", "cpp", "unity", "unreal engine", "cryengine", "godot", "directx", "opengl",
    "vulkan", "shaders", "hlsl", "glsl", "game design", "3d modeling", "blender", "maya",
    "3ds max", "physics engine", "vr", "ar", "virtual reality", "augmented reality", "blockchain",
    "smart contracts", "solidity", "web3", "ethereum", "bitcoin", "hyperledger", "fabric",
    "cryptocurrency", "nft", "defi", "distributed ledger", "consensus algorithms", "ai", "artificial intelligence",
    "devsecops", "react native", "llm", "genai", "prompt engineering", "fine-tuning"  # NEW additions
])

GENERAL_SKILLS = set([
    *IT_SKILLS,
    "communication", "written communication", "verbal communication", "presentation", "public speaking",
    "active listening", "storytelling", "negotiation", "influencing", "collaboration", "teamwork",
    "conflict resolution", "mediation", "customer service", "client facing", "stakeholder management",
    "relationship building", "networking", "leadership", "team leadership", "technical leadership",
    "mentoring", "coaching", "people management", "project management", "program management",
    "product management", "agile coaching", "scrum mastery", "decision making", "strategic thinking",
    "vision", "delegation", "change management", "risk management", "resource allocation", "budgeting",
    "forecasting", "kanban", "prioritization", "time management", "problem solving", "critical thinking",
    "analytical skills", "data analysis", "research", "troubleshooting", "root cause analysis", "debugging",
    "log analysis", "business analysis", "requirements gathering", "systems thinking", "innovation",
    "creativity", "design thinking", "curiosity", "attention to detail", "quality assurance",
    "process improvement", "lean", "six sigma", "business acumen", "domain knowledge", "finance",
    "accounting", "marketing", "digital marketing", "seo", "sem", "social media", "content marketing",
    "ecommerce", "retail", "healthcare", "healthtech", "fintech", "edtech", "supply chain", "logistics",
    "manufacturing", "hr", "human resources", "recruiting", "sales", "business development",
    "partner management", "go-to-market", "gtm", "product marketing", "microsoft office", "excel",
    "word", "powerpoint", "outlook", "google workspace", "gsuite", "sheets", "docs", "slides",
    "jira", "confluence", "trello", "asana", "monday.com", "notion", "slack", "microsoft teams",
    "zoom", "sharepoint", "salesforce", "sap", "quickbooks", "xero", "hubspot", "zendesk",
    "adaptability", "flexibility", "resilience", "persistence", "self motivation", "initiative",
    "proactive", "work ethic", "reliability", "accountability", "ownership", "independence",
    "autonomy", "stress management", "patience", "empathy", "cultural fit", "growth mindset",
    "team player", "remote work", "hybrid work"  # NEW for general CVs
])
# Refined Regex patterns (stricter for accuracy)
NAME_PATTERN = r'^[A-Za-zÀ-ÿ\s\'-]{2,}(?:\s[A-Za-zÀ-ÿ\s\'-]{2,})+$'  # Unchanged
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Unchanged
PHONE_PATTERN = r'(?<!\d)(\+?\d{1,4}[\s.-]?)(\(?\d{2,3}\)?[\s.-]?)\d{2,3}[\s.-]?\d{2,4}(?!\d)'  # Changed: Removed optional for separators, require at least one separator or + to avoid dates
LOCATION_PATTERN = r'(?i)(?:[A-Za-zÀ-ÿ\s]{5,},\s*[A-Za-zÀ-ÿ\s]{5,}|tunisia|tunis|ariana|béja|ben arous|bizerte|gabès|gafsa|jendouba|kairouan|kasserine|kebili|kef|mahdia|manouba|medenine|monastir|nabeul|sfax|sidi bouzid|siliana|sousse|tataouine|tozeur|zaghouan)'  # Changed: Min length 5 for better filter
URL_PATTERN = r'(?i)(?:https?://)?(?:www\.)?(?:linkedin|github|netlify|portfolio)\.[a-zA-Z0-9./-]+(?:\b|$)'  # Unchanged
DATE_PATTERN = r'(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|present|current|ongoing)|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|present|current|ongoing)|\d{4})'  # Unchanged
BULLET_PATTERN = r'^[-•*››◦▪\u2022\u25CF\s]{1,3}'  # Unchanged
LINK_PATTERN = r'(?i)(\\s?)?(?:github|link|website|portfolio|\)'  # Unchanged

# Expanded Section patterns (added more variations, e.g., "executive summary", "work history", "certifs")
SECTION_PATTERNS = {
    "Summary": ["summary", "profile", "about me", "profil", "objectif", "résumé", "career objective", "professional summary", "core competencies", "overview", "bio", "introduction", "personal statement", "executive summary", "career profile"],  # Expanded
    "Skills": ["skills", "technical skills", "compétences", "competences", "expertise", "langages et frameworks", "technologies", "core skills", "technical proficiencies", "abilities", "key skills", "proficiencies", "competences techniques", "hard skills", "soft skills"],  # Expanded
    "Education": ["education", "formation", "parcours académique", "études", "academic background", "qualifications", "academic record", "degree", "degrees", "schooling", "academic qualifications", "studies", "diploma", "baccalaureate", "certifications académiques"],  # Expanded
    "Experience": ["experience", "expérience", "work experience", "expérience professionnelle", "professional experience", "stage", "internship", "employment history", "work history", "intern", "professional background", "career history", "achievements", "tasks", "taches realisees", "responsibilities", "professional history", "stages", "job history"],  # Expanded
    "Projects": ["projects", "projets", "portfolio", "projets personnels", "notable projects", "projet de fin d’étude", "pfa", "pidev", "pi", "projets académiques", "personal projects", "academic projects", "key projects", "pfe", "case studies"],  # Expanded
    "Certifications": ["certifications", "certificats", "badges", "certificates and badges", "achievements", "awards", "professional certifications", "certificates", "badges and certifications", "honors", "certificats et badges", "certifs", "accreditations"],  # Expanded
    "Interests": ["interests", "hobbies", "intérêts", "loisirs", "personal interests", "extracurricular", "activites extracurriculaires", "volunteer work"]  # Expanded
}

# Expanded STOP_WORDS (added more noise like "tel", "addr", "www", dates, common fragments)
STOP_WORDS = [
    "contact", "taches realisees", "profil", "summary", "resume", "present", "current",
    "ongoing", "github", "linkedin", "badge", "keywords", "mots", "cles", "via", "tasks",
    "achievements", "conception", "developpement", "gestion", "projet", "technologies",
    "application", "platform", "system", "email", "phone", "url", "tunis", "ariana",
    "french", "english", "arabic", "professional", "native", "fluent", "b2", "beginner",
    "pfa", "pidev", "pi", "stage", "intern", "member", "club", "experience", "education",
    "skills", "competences", "projects", "projets", "certified", "fundamentals", "award",
    "taches", "realisees", "keywords:", "mots-cles:", "technologies:", "link", "workshop",
    "proficiency", "native", "b2", "a1", "c1", "2020", "2021", "2022", "2023", "2024", "2025",
    "tel", "addr", "www", "http", "https", "com", "tn", "fr", "expected", "mention"  # NEW: More noise for better filtering
]




#4 functions of extraction cleaning ect 
def camel_case_split(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', name)
    return name.title()

def clean_line(line: str) -> str:
    if not line or not isinstance(line, str):
        return ""
    line = normalize('NFKD', line).encode('ASCII', 'ignore').decode('ASCII')
    line = re.sub(r'[\uf0b7\uf076\uf09f•●◦▪\t●•▪○∙\u00a0\U0001F000-\U0001FFFF]', ' ', line)  # Unchanged
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'[()[\]{}|]', '', line).strip()
    if re.match(BULLET_PATTERN, line):
        line = re.sub(BULLET_PATTERN, '- ', line)
    if re.match(LINK_PATTERN, line.strip()):
        return ""
    line = camel_case_split(line)
    words = line.split()
    deduped = []
    for word in words:
        if not deduped or fuzz.ratio(word.lower(), deduped[-1].lower()) < 90:  # Changed: Stricter threshold 90 for less dupes
            deduped.append(word)
    return ' '.join(deduped)

def dedup_text(text: str) -> str:
    if not text:
        return ""
    # Improved language detection (more French words)
    is_french = any(word in text.lower() for word in ["et", "de", "le", "la", "un", "une", "pour", "avec", "dans", "du", "des", "sur", "par", "chez", "au"])
    nlp_model = "fr_core_news_sm" if is_french else "en_core_web_sm"
    try:
        nlp = spacy.load(nlp_model, disable=['ner', 'parser', 'lemmatizer'])
        nlp.add_pipe('sentencizer')
        doc = nlp(text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]  # Changed: Min length 20 to avoid fragments
    except:
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    unique = []
    seen = set()
    for s in sentences:
        s_norm = clean_line(s).lower()
        if s_norm not in seen and all(fuzz.ratio(s_norm, prev) < 85 for prev in seen):  # Changed: Stricter threshold 85
            seen.add(s_norm)
            unique.append(s)
    return ' '.join(unique).strip()

def extract_text_from_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2)  # Preserve layout better
                if page_text:
                    text += page_text + "\n"
                # NEW: Extract tables as text for better handling of education/experience
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += ' | '.join([str(cell) for cell in row if cell]) + "\n"
            if not text:
                text = pdfminer_extract(path)
            return text
    except Exception as e:
        logging.error(f"Extraction failed for {path}: {e}")
        return pdfminer_extract(path)

#5 load and preprocessing functions
import json
import os
import re
import logging
from unicodedata import normalize
from datasets import Dataset, concatenate_datasets
import spacy
from fuzzywuzzy import fuzz
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Cleaning function
def clean_text_for_spacy(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = re.sub(r'[\u2022\u25CF\uf0b7\uf076\uf09f•●◦▪\t○∙\u00a0\U0001F000-\U0001FFFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[()[\]{}|]', '', text).strip()
    return text

# Generate synthetic text for master.json entries without "text" field
def generate_synthetic_text(entry: dict) -> str:
    parts = []
    if entry.get("personal_info", {}).get("name") and entry["personal_info"]["name"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Name: {entry['personal_info']['name']}")
    if entry.get("personal_info", {}).get("email") and entry["personal_info"]["email"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Email: {entry['personal_info']['email']}")
    if entry.get("personal_info", {}).get("phone") and entry["personal_info"]["phone"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Phone: {entry['personal_info']['phone']}")
    if entry.get("personal_info", {}).get("location", {}).get("city") and entry["personal_info"]["location"]["city"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Location: {entry['personal_info']['location']['city']}")
    if entry.get("personal_info", {}).get("summary") and entry["personal_info"]["summary"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Summary: {entry['personal_info']['summary']}")
    for exp in entry.get("experience", []):
        if exp.get("title") and exp["title"] != "Unknown":
            parts.append(f"Role: {exp['title']}")
        if exp.get("company") and exp["company"] != "Unknown":
            parts.append(f"Company: {exp['company']}")
        duration = ' '.join([exp.get("dates", {}).get(k, "") for k in ["start", "end", "duration"] if exp.get("dates", {}).get(k, "") != "Unknown"]).strip()
        if duration:
            parts.append(f"Duration: {duration}")
        if exp.get("responsibilities"):
            desc = ' '.join([r for r in exp.get("responsibilities", []) if r != "Unknown"])
            if desc:
                parts.append(f"Responsibilities: {desc}")
    for edu in entry.get("education", []):
        degree = ' '.join([edu.get("degree", {}).get(k, "") for k in ["level", "field", "major"] if edu.get("degree", {}).get(k, "") != "Unknown"]).strip()
        if degree:
            parts.append(f"Degree: {degree}")
        if edu.get("institution", {}).get("name") and edu["institution"]["name"] != "Unknown":
            parts.append(f"Institution: {edu['institution']['name']}")
        duration = ' '.join([edu.get("dates", {}).get(k, "") for k in ["start", "expected_graduation"] if edu.get("dates", {}).get(k, "") != "Unknown"]).strip()
        if duration:
            parts.append(f"Education Duration: {duration}")
    skills = []
    for skill_type in ["programming_languages", "frameworks", "databases", "cloud", "project_management", "automation", "software_tools"]:
        for skill in entry.get("skills", {}).get("technical", {}).get(skill_type, []):
            skill_name = skill.get("name", "") if isinstance(skill, dict) else skill
            if skill_name and skill_name not in ["Unknown", "Not Provided"]:
                skills.append(skill_name)
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")
    for proj in entry.get("projects", []):
        if proj.get("name") and proj["name"] != "Unknown":
            parts.append(f"Project: {proj['name']}")
        if proj.get("description") and proj["description"] != "Unknown":
            parts.append(f"Project Description: {proj['description']}")
    for cert in entry.get("certifications", []):
        cert_name = cert if isinstance(cert, str) else cert.get("name", "")
        if cert_name and cert_name != "Unknown":
            parts.append(f"Certification: {cert_name}")
    return ' '.join(parts).strip()

# Initialize spaCy for tokenization
nlp_en = spacy.load("en_core_web_sm", disable=['ner', 'lemmatizer'])
nlp_fr = spacy.load("fr_core_news_sm", disable=['ner', 'lemmatizer'])

# Collect new data to append
annotated_data = []

# Function to find span in text
def find_span(text: str, target: str, label: str) -> tuple:
    if not target or len(target) < 4:  # Early check for short targets
        logging.warning(f"Skipping target '{target}' (label: {label}) due to length < 4")
        return None
    match_start = text.find(target)
    if match_start != -1:
        return [match_start, match_start + len(target), label]
    words = text.split()
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            candidate = ' '.join(words[i:j])
            if fuzz.ratio(candidate.lower(), target.lower()) > 85 and len(candidate) >= 4:
                start = text.find(candidate)
                return [start, start + len(candidate), label]
    logging.warning(f"Could not find span for '{target}' (label: {label})")
    return None

# Load and process CVS.json
cvs_path = "/content/CVS.json"
if os.path.exists(cvs_path):
    with open(cvs_path, 'r', encoding='utf-8') as f:
        new_data = []
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    new_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing line {line_number} in CVS.json: {line[:100]}... Error: {e}")
                    continue
        for d in new_data:
            text = clean_text_for_spacy(d.get("content", ""))
            if not text:
                logging.warning(f"No valid content in CVS.json entry at line {line_number}, skipping.")
                continue
            annotations = []
            for ann in d.get("annotation", []):
                try:
                    label = ann["label"][0].upper() if isinstance(ann.get("label"), list) and ann["label"] else "UNKNOWN"
                    for p in ann.get("points", []):
                        start = p.get("start", 0)
                        end = p.get("end", len(text))
                        min_span_length = 4
                        if (start < len(text) and end <= len(text) and
                            start < end and (end - start) >= min_span_length):
                            annotations.append([start, end, label])
                        else:
                            logging.warning(f"Skipping invalid CVS.json annotation at line {line_number}: "
                                          f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
                except Exception as e:
                    logging.error(f"Error processing annotation in CVS.json at line {line_number}: {ann}, Error: {e}")
            if text and annotations:
                annotated_data.append({"text": text, "annotations": annotations})
else:
    logging.error(f"CVS.json not found at {cvs_path}")

# Load and process master.json
master_path = "/content/master.json"
if os.path.exists(master_path):
    master_data = []
    with open(master_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    master_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line {line_number} in master.json: {line[:100]}... Error: {e}")
                    continue
    for d in master_data:
        text = clean_text_for_spacy(d.get("text", ""))
        if not text:
            text = generate_synthetic_text(d)
            if not text:
                logging.warning(f"No valid text after synthesis in master.json entry at line {line_number}, skipping.")
                continue
        annotations = []
        name = d.get("personal_info", {}).get("name", "")
        if name and name not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, name, "NAME")
            if span:
                annotations.append(span)
        email = d.get("personal_info", {}).get("email", "")
        if email and email not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, email, "EMAIL")
            if span:
                annotations.append(span)
        phone = d.get("personal_info", {}).get("phone", "")
        if phone and phone not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, phone, "PHONE")
            if span:
                annotations.append(span)
        location = d.get("personal_info", {}).get("location", {}).get("city", "")
        if location and location not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, location, "LOCATION")
            if span:
                annotations.append(span)
        summary = d.get("personal_info", {}).get("summary", "")
        if summary and summary != "Unknown" and len(summary) > 20:
            span = find_span(text, summary, "SUMMARY_TEXT")
            if span:
                annotations.append(span)
        for exp in d.get("experience", []):
            role = exp.get("title", "")
            if role and role != "Unknown":
                span = find_span(text, role, "ROLE")
                if span:
                    annotations.append(span)
            company = exp.get("company", "")
            if company and company != "Unknown":
                span = find_span(text, company, "COMPANY")
                if span:
                    annotations.append(span)
            duration = ' '.join([exp.get("dates", {}).get(k, "") for k in ["start", "end", "duration"] if exp.get("dates", {}).get(k, "") != "Unknown"]).strip()
            if duration:
                span = find_span(text, duration, "DURATION")
                if span:
                    annotations.append(span)
            desc = ' '.join([r for r in exp.get("responsibilities", []) if r != "Unknown"])
            if desc and len(desc) > 20:
                span = find_span(text, desc, "DESCRIPTION")
                if span:
                    annotations.append(span)
        for edu in d.get("education", []):
            degree = ' '.join([edu.get("degree", {}).get(k, "") for k in ["level", "field", "major"] if edu.get("degree", {}).get(k, "") != "Unknown"]).strip()
            if degree:
                span = find_span(text, degree, "DEGREE")
                if span:
                    annotations.append(span)
            institution = edu.get("institution", {}).get("name", "")
            if institution and institution != "Unknown":
                span = find_span(text, institution, "INSTITUTION")
                if span:
                    annotations.append(span)
            duration = ' '.join([edu.get("dates", {}).get(k, "") for k in ["start", "expected_graduation"] if edu.get("dates", {}).get(k, "") != "Unknown"]).strip()
            if duration:
                span = find_span(text, duration, "DURATION")
                if span:
                    annotations.append(span)
        skills = []
        for skill_type in ["programming_languages", "frameworks", "databases", "cloud", "project_management", "automation", "software_tools"]:
            for skill in d.get("skills", {}).get("technical", {}).get(skill_type, []):
                skill_name = skill.get("name", "") if isinstance(skill, dict) else skill
                if skill_name and skill_name not in ["Unknown", "Not Provided"]:
                    skills.append(skill_name)
        for skill_name in skills:
            span = find_span(text, skill_name, "SKILL")
            if span:
                annotations.append(span)
        for proj in d.get("projects", []):
            name = proj.get("name", "")
            if name and name != "Unknown":
                span = find_span(text, name, "PROJECT_NAME")
                if span:
                    annotations.append(span)
            desc = proj.get("description", "")
            if desc and desc != "Unknown" and len(desc) > 20:
                span = find_span(text, desc, "DESCRIPTION")
                if span:
                    annotations.append(span)
        for cert in d.get("certifications", []):
            cert_name = cert if isinstance(cert, str) else cert.get("name", "")
            if cert_name and cert_name != "Unknown":
                span = find_span(text, cert_name, "CERT")
                if span:
                    annotations.append(span)
        if annotations:
            annotated_data.append({"text": text, "annotations": annotations})
        else:
            logging.warning(f"No valid annotations for master.json entry at line {line_number}, skipping.")
else:
    logging.error(f"master.json not found at {master_path}")

# Fix and process test.json
test_path = "/content/test.json"
if os.path.exists(test_path):
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line {line_number} in test.json: {line[:100]}... Error: {e}")
                    continue
    for d in test_data:
        text = clean_text_for_spacy(d.get("text", ""))
        if not text:
            logging.warning(f"No valid text in test.json entry at line {line_number}, skipping.")
            continue
        annotations = []
        for ent in d.get("entities", []):
            label = ent["label"].upper()
            start, end = ent["start"], ent["end"]
            min_span_length = 4
            if (start < len(text) and end <= len(text) and
                start < end and (end - start) >= min_span_length):
                annotations.append([start, end, label])
            else:
                logging.warning(f"Skipping invalid test.json annotation at line {line_number}: "
                              f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
        if annotations:
            annotated_data.append({"text": text, "annotations": annotations})
else:
    logging.error(f"test.json not found at {test_path}")

# Collect unique labels
all_labels = set(["SKILL", "NAME", "DEGREE", "INSTITUTION", "ROLE", "COMPANY", "DURATION", "PROJECT_NAME", "CERT", "LOCATION", "PHONE", "EMAIL", "DESCRIPTION", "SUMMARY_TEXT", "YEARS_OF_EXPERIENCE", "LANGUAGE", "ADDRESS", "PROJECT", "EXPERIENCE", "EDUCATION"])
for item in annotated_data:
    for ann in item["annotations"]:
        all_labels.add(ann[2])
label_list = ["O"] + sorted([f"B-{l}" for l in all_labels] + [f"I-{l}" for l in all_labels])

# Convert to NER format with improved validation
def convert_to_ner_format(data_list):
    ner_data = {"tokens": [], "ner_tags": []}
    for item in data_list:
        text = item["text"]
        is_french = any(word in text.lower() for word in ["et", "de", "le", "la", "un", "une", "pour"])
        try:
            doc = nlp_fr(text) if is_french else nlp_en(text)
            tokens = [token.text for token in doc]
            tags = ["O"] * len(tokens)
            valid_annotations = []
            for start, end, label in item["annotations"]:
                if start < len(text) and end <= len(text) and start < end and (end - start) >= 4:
                    span = doc.char_span(start, end, alignment_mode="expand")
                    if span:
                        token_start = span.start
                        token_end = span.end
                        if token_start < len(tokens) and token_end <= len(tokens):
                            tags[token_start] = f"B-{label}"
                            for i in range(token_start + 1, token_end):
                                tags[i] = f"I-{label}"
                            valid_annotations.append([start, end, label])
                        else:
                            logging.warning(f"Failed to align span [{start}, {end}, {label}] in text: {text[:50]}...")
                    else:
                        logging.warning(f"Failed to align span [{start}, {end}, {label}] in text: {text[:50]}...")
                else:
                    logging.warning(f"Skipping invalid span [{start}, {end}, {label}] in text: {text[:50]}... (text length: {len(text)}, span: {end - start})")
            ner_data["tokens"].append(tokens)
            ner_data["ner_tags"].append(tags)
        except Exception as e:
            logging.error(f"Tokenization failed for text: {text[:100]}... Error: {e}")
            tokens = nltk.word_tokenize(text)
            tags = ["O"] * len(tokens)
            ner_data["tokens"].append(tokens)
            ner_data["ner_tags"].append(tags)
    return Dataset.from_dict(ner_data)

# Try loading existing datasets
try:
    train_dataset_path = "/content/train_dataset.json"
    eval_dataset_path = "/content/eval_dataset.json"
    if os.path.exists(train_dataset_path) and os.path.exists(eval_dataset_path):
        train_dataset = Dataset.from_json(train_dataset_path)
        eval_dataset = Dataset.from_json(eval_dataset_path)
        logging.info(f"Loaded existing datasets: Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    else:
        # Fallback: Recreate from test.json
        logging.warning("Existing dataset files not found. Recreating from test.json...")
        annotated_data_test = []
        if os.path.exists(test_path):
            test_data = []
            with open(test_path, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            test_data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logging.error(f"Failed to parse line {line_number} in test.json: {line[:100]}... Error: {e}")
                            continue
            for d in test_data:
                text = clean_text_for_spacy(d.get("text", ""))
                if not text:
                    logging.warning(f"No valid text in test.json entry at line {line_number}, skipping.")
                    continue
                annotations = []
                for ent in d.get("entities", []):
                    label = ent["label"].upper()
                    start, end = ent["start"], ent["end"]
                    min_span_length = 4
                    if (start < len(text) and end <= len(text) and
                        start < end and (end - start) >= min_span_length):
                        annotations.append([start, end, label])
                    else:
                        logging.warning(f"Skipping invalid test.json annotation at line {line_number}: "
                                      f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
                if annotations:
                    annotated_data_test.append({"text": text, "annotations": annotations})
            total_data = len(annotated_data_test)
            train_size = int(total_data * 0.8)
            train_dataset = convert_to_ner_format(annotated_data_test[:train_size])
            eval_dataset = convert_to_ner_format(annotated_data_test[train_size:])
            logging.info(f"Recreated datasets from test.json: Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
        else:
            logging.error(f"test.json not found at {test_path}. Cannot proceed without existing datasets.")
            exit(1)
except NameError:
    # Fallback: Recreate from test.json if datasets are not in memory
    logging.warning("Existing datasets not in memory. Recreating from test.json...")
    annotated_data_test = []
    if os.path.exists(test_path):
        test_data = []
        with open(test_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if line.strip():
                    try:
                        test_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logging.error(f"Failed to parse line {line_number} in test.json: {line[:100]}... Error: {e}")
                        continue
        for d in test_data:
            text = clean_text_for_spacy(d.get("text", ""))
            if not text:
                logging.warning(f"No valid text in test.json entry at line {line_number}, skipping.")
                continue
            annotations = []
            for ent in d.get("entities", []):
                label = ent["label"].upper()
                start, end = ent["start"], ent["end"]
                min_span_length = 4
                if (start < len(text) and end <= len(text) and
                    start < end and (end - start) >= min_span_length):
                    annotations.append([start, end, label])
                else:
                    logging.warning(f"Skipping invalid test.json annotation at line {line_number}: "
                                  f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
            if annotations:
                annotated_data_test.append({"text": text, "annotations": annotations})
        total_data = len(annotated_data_test)
        train_size = int(total_data * 0.8)
        train_dataset = convert_to_ner_format(annotated_data_test[:train_size])
        eval_dataset = convert_to_ner_format(annotated_data_test[train_size:])
        logging.info(f"Recreated datasets from test.json: Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    else:
        logging.error(f"test.json not found at {test_path}. Cannot proceed without existing datasets.")
        exit(1)

# Convert new data and append to existing datasets
new_dataset = convert_to_ner_format(annotated_data)
train_size = int(len(new_dataset) * 0.8)
new_train_dataset = new_dataset.select(range(train_size))
new_eval_dataset = new_dataset.select(range(train_size, len(new_dataset)))

# Merge datasets using concatenate_datasets
combined_train_dataset = concatenate_datasets([train_dataset, new_train_dataset])
combined_eval_dataset = concatenate_datasets([eval_dataset, new_eval_dataset])

# Save updated datasets
combined_train_dataset.to_json("/content/train_dataset_updated.json")
combined_eval_dataset.to_json("/content/eval_dataset_updated.json")
logging.info(f"Updated datasets saved: Train size: {len(combined_train_dataset)}, Eval size: {len(combined_eval_dataset)}")


import json
import os
import re
import logging
from unicodedata import normalize
from datasets import Dataset
import spacy
from fuzzywuzzy import fuzz
import nltk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Enhanced cleaning function (unchanged)
def clean_text_for_spacy(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = re.sub(r'[\u2022\u25CF\uf0b7\uf076\uf09f•●◦▪\t○∙\u00a0\U0001F000-\U0001FFFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[()[\]{}|]', '', text).strip()
    return text

# Generate synthetic text for master.json entries without "text" field
def generate_synthetic_text(entry: dict) -> str:
    parts = []
    # Personal Info
    if entry.get("personal_info", {}).get("name") and entry["personal_info"]["name"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Name: {entry['personal_info']['name']}")
    if entry.get("personal_info", {}).get("email") and entry["personal_info"]["email"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Email: {entry['personal_info']['email']}")
    if entry.get("personal_info", {}).get("phone") and entry["personal_info"]["phone"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Phone: {entry['personal_info']['phone']}")
    if entry.get("personal_info", {}).get("location", {}).get("city") and entry["personal_info"]["location"]["city"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Location: {entry['personal_info']['location']['city']}")
    if entry.get("personal_info", {}).get("summary") and entry["personal_info"]["summary"] not in ["Unknown", "Not Provided", ""]:
        parts.append(f"Summary: {entry['personal_info']['summary']}")

    # Experience
    for exp in entry.get("experience", []):
        if exp.get("title") and exp["title"] != "Unknown":
            parts.append(f"Role: {exp['title']}")
        if exp.get("company") and exp["company"] != "Unknown":
            parts.append(f"Company: {exp['company']}")
        duration = ' '.join([exp.get("dates", {}).get(k, "") for k in ["start", "end", "duration"] if exp.get("dates", {}).get(k, "") != "Unknown"]).strip()
        if duration:
            parts.append(f"Duration: {duration}")
        if exp.get("responsibilities"):
            desc = ' '.join([r for r in exp.get("responsibilities", []) if r != "Unknown"])
            if desc:
                parts.append(f"Responsibilities: {desc}")

    # Education
    for edu in entry.get("education", []):
        degree = ' '.join([edu.get("degree", {}).get(k, "") for k in ["level", "field", "major"] if edu.get("degree", {}).get(k, "") != "Unknown"]).strip()
        if degree:
            parts.append(f"Degree: {degree}")
        if edu.get("institution", {}).get("name") and edu["institution"]["name"] != "Unknown":
            parts.append(f"Institution: {edu['institution']['name']}")
        duration = ' '.join([edu.get("dates", {}).get(k, "") for k in ["start", "expected_graduation"] if edu.get("dates", {}).get(k, "") != "Unknown"]).strip()
        if duration:
            parts.append(f"Education Duration: {duration}")

    # Skills
    skills = []
    for skill_type in ["programming_languages", "frameworks", "databases", "cloud", "project_management", "automation", "software_tools"]:
        for skill in entry.get("skills", {}).get("technical", {}).get(skill_type, []):
            skill_name = skill.get("name", "") if isinstance(skill, dict) else skill
            if skill_name and skill_name not in ["Unknown", "Not Provided"]:
                skills.append(skill_name)
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")

    # Projects
    for proj in entry.get("projects", []):
        if proj.get("name") and proj["name"] != "Unknown":
            parts.append(f"Project: {proj['name']}")
        if proj.get("description") and proj["description"] != "Unknown":
            parts.append(f"Project Description: {proj['description']}")

    # Certifications
    for cert in entry.get("certifications", []):
        cert_name = cert if isinstance(cert, str) else cert.get("name", "")
        if cert_name and cert_name != "Unknown":
            parts.append(f"Certification: {cert_name}")

    return ' '.join(parts).strip()

# Load annotated folder
annotated_folder = "/content/aziz/ResumesJsonAnnotated/ResumesJsonAnnotated/"
annotated_data = []
for filename in os.listdir(annotated_folder):
    if filename.endswith(".json"):
        try:
            with open(os.path.join(annotated_folder, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                cleaned_text = clean_text_for_spacy(data.get("text", ""))
                if not cleaned_text:
                    logging.warning(f"No valid text in {filename}, skipping.")
                    continue
                normalized_annotations = []
                for ann in data.get("annotations", []):
                    label = ann[2].split(":")[0].strip().upper() if ":" in ann[2] else ann[2].upper()
                    start, end = ann[0], ann[1]
                    min_span_length = 4  # Changed: Stricter min 4 for all labels to reduce noise
                    if (start < len(cleaned_text) and end <= len(cleaned_text) and
                        start < end and (end - start) >= min_span_length):
                        normalized_annotations.append([start, end, label])
                    else:
                        logging.warning(f"Skipping invalid annotation in {filename}: {ann} "
                                      f"(text length: {len(cleaned_text)}, span: {end - start})")
                if normalized_annotations:
                    annotated_data.append({
                        "text": cleaned_text,
                        "annotations": normalized_annotations
                    })
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse {filename}: {e}")
            continue

# Load CVS.json
cvs_path = "/content/CVS.json"
if os.path.exists(cvs_path):
    with open(cvs_path, 'r', encoding='utf-8') as f:
        new_data = []
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    new_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing line {line_number} in CVS.json: {line[:100]}... Error: {e}")
                    continue
        for d in new_data:
            text = clean_text_for_spacy(d.get("content", ""))
            if not text:
                logging.warning(f"No valid content in CVS.json entry at line {line_number}, skipping.")
                continue
            annotations = []
            for ann in d.get("annotation", []):
                try:
                    label = ann["label"][0].upper() if isinstance(ann.get("label"), list) and ann["label"] else "UNKNOWN"
                    for p in ann.get("points", []):
                        start = p.get("start", 0)
                        end = p.get("end", len(text))
                        min_span_length = 4  # Changed: Stricter min 4
                        if (start < len(text) and end <= len(text) and
                            start < end and (end - start) >= min_span_length):
                            annotations.append([start, end, label])
                        else:
                            logging.warning(f"Skipping invalid CVS.json annotation at line {line_number}: "
                                          f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
                except Exception as e:
                    logging.error(f"Error processing annotation in CVS.json at line {line_number}: {ann}, Error: {e}")
            if text and annotations:
                annotated_data.append({"text": text, "annotations": annotations})
else:
    logging.error(f"CVS.json not found at {cvs_path}")

# Load master.json as JSONL
master_path = "/content/master.json"
if os.path.exists(master_path):
    master_data = []
    with open(master_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    master_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line {line_number} in master.json: {line[:100]}... Error: {e}")
                    continue
    for d in master_data:
        text = clean_text_for_spacy(d.get("text", ""))
        if not text:
            text = generate_synthetic_text(d)
            if not text:
                logging.warning(f"No valid text after synthesis in master.json entry at line {line_number}, skipping.")
                continue
        annotations = []

        def find_span(text: str, target: str, label: str) -> tuple:
            if not target:
                return None
            match_start = text.find(target)
            if match_start != -1:
                return [match_start, match_start + len(target), label]
            words = text.split()
            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    candidate = ' '.join(words[i:j])
                    if fuzz.ratio(candidate.lower(), target.lower()) > 85:
                        start = text.find(candidate)
                        return [start, start + len(candidate), label]
            logging.warning(f"Could not find span for '{target}' (label: {label}) in master.json entry")
            return None

        # Personal Info
        name = d.get("personal_info", {}).get("name", "")
        if name and name not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, name, "NAME")
            if span:
                annotations.append(span)
        email = d.get("personal_info", {}).get("email", "")
        if email and email not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, email, "EMAIL")
            if span:
                annotations.append(span)
        phone = d.get("personal_info", {}).get("phone", "")
        if phone and phone not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, phone, "PHONE")
            if span:
                annotations.append(span)
        location = d.get("personal_info", {}).get("location", {}).get("city", "")
        if location and location not in ["Unknown", "Not Provided", ""]:
            span = find_span(text, location, "LOCATION")
            if span:
                annotations.append(span)
        summary = d.get("personal_info", {}).get("summary", "")
        if summary and summary != "Unknown" and len(summary) > 20:
            span = find_span(text, summary, "SUMMARY_TEXT")
            if span:
                annotations.append(span)

        # Experience
        for exp in d.get("experience", []):
            role = exp.get("title", "")
            if role and role != "Unknown":
                span = find_span(text, role, "ROLE")
                if span:
                    annotations.append(span)
            company = exp.get("company", "")
            if company and company != "Unknown":
                span = find_span(text, company, "COMPANY")
                if span:
                    annotations.append(span)
            duration = ' '.join([exp.get("dates", {}).get(k, "") for k in ["start", "end", "duration"] if exp.get("dates", {}).get(k, "") != "Unknown"]).strip()
            if duration:
                span = find_span(text, duration, "DURATION")
                if span:
                    annotations.append(span)
            desc = ' '.join([r for r in exp.get("responsibilities", []) if r != "Unknown"])
            if desc and len(desc) > 20:
                span = find_span(text, desc, "DESCRIPTION")
                if span:
                    annotations.append(span)

        # Education
        for edu in d.get("education", []):
            degree = ' '.join([edu.get("degree", {}).get(k, "") for k in ["level", "field", "major"] if edu.get("degree", {}).get(k, "") != "Unknown"]).strip()
            if degree:
                span = find_span(text, degree, "DEGREE")
                if span:
                    annotations.append(span)
            institution = edu.get("institution", {}).get("name", "")
            if institution and institution != "Unknown":
                span = find_span(text, institution, "INSTITUTION")
                if span:
                    annotations.append(span)
            duration = ' '.join([edu.get("dates", {}).get(k, "") for k in ["start", "expected_graduation"] if edu.get("dates", {}).get(k, "") != "Unknown"]).strip()
            if duration:
                span = find_span(text, duration, "DURATION")
                if span:
                    annotations.append(span)

        # Skills
        skills = []
        for skill_type in ["programming_languages", "frameworks", "databases", "cloud", "project_management", "automation", "software_tools"]:
            for skill in d.get("skills", {}).get("technical", {}).get(skill_type, []):
                skill_name = skill.get("name", "") if isinstance(skill, dict) else skill
                if skill_name and skill_name not in ["Unknown", "Not Provided"]:
                    skills.append(skill_name)
        for skill_name in skills:
            span = find_span(text, skill_name, "SKILL")
            if span:
                annotations.append(span)

        # Projects
        for proj in d.get("projects", []):
            name = proj.get("name", "")
            if name and name != "Unknown":
                span = find_span(text, name, "PROJECT_NAME")
                if span:
                    annotations.append(span)
            desc = proj.get("description", "")
            if desc and desc != "Unknown" and len(desc) > 20:
                span = find_span(text, desc, "DESCRIPTION")
                if span:
                    annotations.append(span)

        # Certifications
        for cert in d.get("certifications", []):
            cert_name = cert if isinstance(cert, str) else cert.get("name", "")
            if cert_name and cert_name != "Unknown":
                span = find_span(text, cert_name, "CERT")
                if span:
                    annotations.append(span)

        if annotations:
            annotated_data.append({"text": text, "annotations": annotations})
        else:
            logging.warning(f"No valid annotations for master.json entry at line {line_number}, skipping.")

# NEW: Load test.json (your +50 new CVs) as additional training data
test_path = "/content/test.json"  # Assume path in Colab
if os.path.exists(test_path):
    test_data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse line {line_number} in test.json: {line[:100]}... Error: {e}")
                    continue
    for d in test_data:
        text = clean_text_for_spacy(d.get("text", ""))
        if not text:
            logging.warning(f"No valid text in test.json entry at line {line_number}, skipping.")
            continue
        annotations = []
        for ent in d.get("entities", []):
            label = ent["label"].upper()
            start, end = ent["start"], ent["end"]
            min_span_length = 4  # Stricter
            if (start < len(text) and end <= len(text) and
                start < end and (end - start) >= min_span_length):
                annotations.append([start, end, label])
            else:
                logging.warning(f"Skipping invalid test.json annotation at line {line_number}: "
                              f"[{start}, {end}, {label}] (text length: {len(text)}, span: {end - start})")
        if annotations:
            annotated_data.append({"text": text, "annotations": annotations})
else:
    logging.error(f"test.json not found at {test_path}")

# Collect unique labels
all_labels = set(["SKILL", "NAME", "DEGREE", "INSTITUTION", "ROLE", "COMPANY", "DURATION", "PROJECT_NAME", "CERT", "LOCATION", "PHONE", "EMAIL", "DESCRIPTION", "SUMMARY_TEXT", "YEARS_OF_EXPERIENCE"])
for item in annotated_data:
    for ann in item["annotations"]:
        all_labels.add(ann[2])
label_list = ["O"] + sorted([f"B-{l}" for l in all_labels] + [f"I-{l}" for l in all_labels])

# Convert to NER format
def convert_to_ner_format(data_list):
    ner_data = {"tokens": [], "ner_tags": []}
    nlp_en = spacy.load("en_core_web_sm", disable=['ner', 'lemmatizer'])
    nlp_fr = spacy.load("fr_core_news_sm", disable=['ner', 'lemmatizer'])
    for item in data_list:
        text = item["text"]
        is_french = any(word in text.lower() for word in ["et", "de", "le", "la", "un", "une", "pour"])
        try:
            doc = nlp_fr(text) if is_french else nlp_en(text)
            tokens = [token.text for token in doc]
            tags = ["O"] * len(tokens)
            for start, end, label in item["annotations"]:
                span = doc.char_span(start, end, alignment_mode="expand")
                if span:
                    token_start = span.start
                    token_end = span.end
                    if token_start < len(tokens) and token_end <= len(tokens):
                        tags[token_start] = f"B-{label}"
                        for i in range(token_start + 1, token_end):
                            tags[i] = f"I-{label}"
                else:
                    logging.warning(f"Failed to align span [{start}, {end}, {label}] in text")
            ner_data["tokens"].append(tokens)
            ner_data["ner_tags"].append(tags)
        except Exception as e:
            logging.error(f"Tokenization failed for text: {text[:100]}... Error: {e}")
            tokens = nltk.word_tokenize(text)
            tags = ["O"] * len(tokens)
            ner_data["tokens"].append(tokens)
            ner_data["ner_tags"].append(tags)
    return Dataset.from_dict(ner_data)

# Process datasets
print("Creating training dataset...")
total_data = len(annotated_data)
train_size = int(total_data * 0.8)
train_dataset = convert_to_ner_format(annotated_data[:train_size])
eval_dataset = convert_to_ner_format(annotated_data[train_size:])
print(f"Datasets created successfully! Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")



#6 train 
import os
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, concatenate_datasets
import torch
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from transformers import get_scheduler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

# Disable Weights & Biases
os.environ["WANDB_MODE"] = "disabled"

# Model and tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load updated datasets
train_dataset_path = "/content/train_dataset_updated.json"
eval_dataset_path = "/content/eval_dataset_updated.json"
try:
    train_dataset_updated = Dataset.from_json(train_dataset_path)
    eval_dataset_updated = Dataset.from_json(eval_dataset_path)
    logging.info(f"Loaded updated datasets: Train size: {len(train_dataset_updated)}, Eval size: {len(eval_dataset_updated)}")
except Exception as e:
    logging.error(f"Failed to load updated datasets: {e}")
    raise

# Collect unique labels from the dataset
all_labels = set()
for dataset in [train_dataset_updated, eval_dataset_updated]:
    for ner_tags in dataset["ner_tags"]:
        all_labels.update(ner_tags)
label_list = ["O"] + sorted([label for label in all_labels if label != "O"])
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}
logging.info(f"Label list: {label_list}")

def tokenize_and_align_labels(examples):
    try:
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding='max_length',
            max_length=512,
            return_tensors="np"
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Default to "O" if label is not in label_to_id
                    label_ids.append(label_to_id.get(label[word_idx], label_to_id["O"]))
                else:
                    # Convert B- to I- for subsequent subwords
                    if label[word_idx].startswith("B-"):
                        i_label = label[word_idx].replace("B-", "I-")
                        label_ids.append(label_to_id.get(i_label, label_to_id["O"]))
                    else:
                        label_ids.append(label_to_id.get(label[word_idx], label_to_id["O"]))
                previous_word_idx = word_idx
            # Pad or truncate labels to match max_length
            label_ids = label_ids[:512] + [-100] * (512 - len(label_ids)) if len(label_ids) < 512 else label_ids[:512]
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    except Exception as e:
        logging.error(f"Error in tokenization: {e}")
        raise

print("Tokenizing datasets...")
try:
    tokenized_train = train_dataset_updated.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing train dataset"
    )
    tokenized_eval = eval_dataset_updated.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=["tokens", "ner_tags"],
        desc="Tokenizing eval dataset"
    )
    print(f"Dataset sizes: Train = {len(tokenized_train)}, Eval = {len(tokenized_eval)}")
except Exception as e:
    logging.error(f"Failed to tokenize datasets: {e}")
    raise

# Initialize model
try:
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id
    )
except Exception as e:
    logging.error(f"Failed to initialize model: {e}")
    raise

def compute_metrics(p):
    try:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
        true_predictions = [[id_to_label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]

        # Remove empty sequences to avoid seqeval errors
        true_labels = [lbl for lbl in true_labels if lbl]
        true_predictions = [pred for pred in true_predictions if pred]

        report = classification_report(true_labels, true_predictions)
        print("Per-label classification report:\n", report)
        metrics = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        return metrics
    except Exception as e:
        logging.error(f"Error in compute_metrics: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=7,
    weight_decay=0.01,
    save_steps=500,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    gradient_accumulation_steps=4,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    resume_from_checkpoint=True,  # Enable resuming from latest checkpoint
)

# Custom Trainer with cosine scheduler
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )

# Initialize trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Start training
print("Starting training...")
try:
    trainer.train(resume_from_checkpoint=True)  # Attempt to resume if checkpoint exists
except Exception as e:
    logging.warning(f"Failed to resume training from checkpoint: {e}. Starting fresh training...")
    trainer.train()

# Save model
try:
    trainer.save_model("./fine_tuned_resume_ner")
    logging.info("Model saved successfully!")
except Exception as e:
    logging.error(f"Failed to save model: {e}")

# Initialize pipeline
try:
    nlp = pipeline(
        "ner",
        model="./fine_tuned_resume_ner",
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("Model saved and pipeline initialized!")
except Exception as e:
    logging.error(f"Failed to initialize pipeline: {e}")

# Evaluate and print metrics
try:
    metrics = trainer.evaluate()
    print(f"Evaluation Metrics: {metrics}")
except Exception as e:
    logging.error(f"Failed to evaluate model: {e}")



#parsing functions and main test script
import json
import os
import re
import logging
from collections import defaultdict
from unicodedata import normalize
from typing import Dict, List, Any, Tuple
from fuzzywuzzy import fuzz
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract
import torch
import spacy
import nltk
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_line(line: str) -> str:
    """Clean and normalize a line of text"""
    if not line or not isinstance(line, str):
        return ""

    # Normalize unicode
    line = normalize('NFKD', line).encode('ASCII', 'ignore').decode('ASCII')

    # Remove special characters and excessive whitespace
    line = re.sub(r'[\uf0b7\uf076\uf09f•●◦▪\t●•▪○∙\u00a0]', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()

    return line

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF with focus on content"""
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"

            if not text.strip():
                text = pdfminer_extract(path)

            return text.strip()
    except Exception as e:
        logging.error(f"PDF extraction failed for {path}: {e}")
        return pdfminer_extract(path)

def extract_contact_info(raw_text: str) -> Dict[str, str]:
    """Extract contact information"""
    # Extract emails
    emails = list(set(re.findall(EMAIL_PATTERN, raw_text)))

    # Extract phones
    phones = []
    phone_matches = re.finditer(PHONE_PATTERN, raw_text)
    for match in phone_matches:
        phone = match.group().strip()
        digits = re.sub(r'\D', '', phone)
        if 8 <= len(digits) <= 15:
            phones.append(phone)

    # Extract locations
    locations = list(set(re.findall(LOCATION_PATTERN, raw_text)))

    # Extract URLs
    urls = list(set(re.findall(URL_PATTERN, raw_text)))

    return {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "location": locations[0] if locations else "",
        "url": urls[0] if urls else ""
    }

def extract_name(lines: List[str], raw_text: str, contact_info: Dict[str, str]) -> str:
    """Extract name using multiple strategies"""
    # Strategy 1: Look for name patterns in first 5 lines
    for i, line in enumerate(lines[:5]):
        clean = clean_line(line)
        if (len(clean) > 5 and len(clean) < 50 and
            re.match(NAME_PATTERN, clean) and
            2 <= len(clean.split()) <= 4 and
            not any(kw in clean.lower() for kw in STOP_WORDS) and
            not re.match(EMAIL_PATTERN, clean) and
            not re.match(PHONE_PATTERN, clean)):
            return clean.title()

    # Strategy 2: Extract from email
    email = contact_info.get("email", "")
    if email and "@" in email:
        username = email.split('@')[0]
        name_parts = re.split(r'[\._\-\d+]', username)
        valid_parts = [part for part in name_parts if len(part) > 2 and not re.search(r'\d', part)]
        if len(valid_parts) >= 2:
            return ' '.join(valid_parts[:2]).title()

    return "Unknown Name"

def detect_section_header(line: str, current_section: str) -> Tuple[str, bool]:
    """Detect section headers"""
    clean = clean_line(line).lower().strip()
    if not clean or len(clean) > 50:
        return current_section, False

    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if pat in clean and len(clean.split()) <= 4:
                return section, True

    return current_section, False

def parse_education(lines: List[str]) -> List[Dict[str, str]]:
    """Parse education information"""
    entries = []

    education_keywords = ["bachelor", "master", "phd", "diploma", "degree", "licence", "engineering", "esprit", "ensit", "university", "college", "institute", "school"]

    i = 0
    while i < len(lines):
        line = lines[i]
        clean = clean_line(line)

        if len(clean) < 8:
            i += 1
            continue

        # Look for education indicators
        has_edu_kw = any(kw in clean.lower() for kw in education_keywords)
        date_match = re.search(DATE_PATTERN, clean)

        if has_edu_kw or date_match:
            entry = {"degree": "", "institution": "", "duration": ""}

            # Extract date if present
            if date_match:
                entry["duration"] = date_match.group()
                clean = re.sub(DATE_PATTERN, '', clean).strip()

            # Try to extract institution
            if " at " in clean.lower():
                parts = clean.split(" at ", 1)
                entry["degree"] = parts[0].strip().title()
                entry["institution"] = parts[1].strip().title()
            elif " in " in clean.lower():
                parts = clean.split(" in ", 1)
                entry["degree"] = parts[0].strip().title()
                entry["institution"] = parts[1].strip().title()
            elif " - " in clean:
                parts = clean.split(" - ", 1)
                entry["degree"] = parts[0].strip().title()
                entry["institution"] = parts[1].strip().title()
            else:
                entry["degree"] = clean.title()

            # Validate and add entry
            if entry["degree"] and len(entry["degree"]) > 3:
                entries.append(entry)

        i += 1

    return entries[:3]

def parse_experience(lines: List[str]) -> List[Dict[str, str]]:
    """PERFECT experience parsing - CLEAN AND SEPARATE from projects"""
    experiences = []

    # Clear experience indicators
    experience_keywords = [
        "engineer", "developer", "analyst", "manager", "specialist", "consultant",
        "architect", "director", "lead", "intern", "stage", "stagiare", "employment",
        "work experience", "professional experience", "expérience professionnelle"
    ]

    current_exp = None
    description_lines = []
    in_experience_section = False

    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue

        # Check if we're in experience section
        section, is_header = detect_section_header(clean, "")
        if is_header:
            if section == "Experience":
                in_experience_section = True
            elif section in ["Projects", "Education", "Skills"]:
                in_experience_section = False

        # Look for experience entries - ONLY in experience section or with clear markers
        date_match = re.search(DATE_PATTERN, clean)
        has_exp_keyword = any(kw in clean.lower() for kw in experience_keywords)
        is_short_line = len(clean) < 120

        # Start new experience when we have date + experience keywords, OR clear role pattern
        if (date_match and (has_exp_keyword or in_experience_section)) or (has_exp_keyword and is_short_line):
            # Save previous experience if valid
            if current_exp and current_exp["role"] and len(current_exp["role"]) > 3:
                current_exp["description"] = ' '.join(description_lines).strip()[:400]
                if len(current_exp["description"]) > 10:
                    experiences.append(current_exp)

            # Start new experience
            duration = date_match.group() if date_match else ""
            remaining = re.sub(DATE_PATTERN, '', clean).strip()

            # Extract role and company using multiple patterns
            role = remaining
            company = ""

            # Pattern 1: "Role at Company"
            if " at " in remaining.lower():
                parts = remaining.split(" at ", 1)
                role = parts[0].strip()
                company = parts[1].strip()
            # Pattern 2: "Role - Company"
            elif " - " in remaining:
                parts = remaining.split(" - ", 1)
                role = parts[0].strip()
                company = parts[1].strip()
            # Pattern 3: "Role, Company"
            elif ", " in remaining and len(remaining.split(", ")) == 2:
                parts = remaining.split(", ", 1)
                role = parts[0].strip()
                company = parts[1].strip()

            current_exp = {
                "role": role.title(),
                "company": company.title(),
                "duration": duration,
                "description": ""
            }
            description_lines = []

            # Look ahead 1-2 lines for initial description
            for j in range(i+1, min(i+3, len(lines))):
                next_line = clean_line(lines[j])
                if (len(next_line) > 15 and
                    not re.search(DATE_PATTERN, next_line) and
                    not detect_section_header(next_line, "")[1] and
                    len(description_lines) < 2):
                    description_lines.append(next_line)

            continue

        # Collect description for current experience
        elif current_exp and len(clean) > 10:
            # Skip if it's clearly a new section or another experience
            is_new_exp = (re.search(DATE_PATTERN, clean) and
                         any(kw in clean.lower() for kw in experience_keywords))

            if not is_new_exp and len(description_lines) < 6:
                description_lines.append(clean)

    # Add final experience
    if current_exp and current_exp["role"] and len(current_exp["role"]) > 3:
        current_exp["description"] = ' '.join(description_lines).strip()[:400]
        if len(current_exp["description"]) > 10:
            experiences.append(current_exp)

    # Final validation - remove experiences that are actually projects
    valid_experiences = []
    for exp in experiences:
        # Filter out project-like entries
        is_actual_experience = (
            not any(proj_kw in exp["role"].lower() for proj_kw in ["project", "projet", "pfa", "pfe"]) and
            not exp["role"].lower().startswith("projet") and
            len(exp["role"]) > 5 and
            exp["duration"]  # Should have dates for real experience
        )

        if is_actual_experience:
            valid_experiences.append(exp)

    return valid_experiences[:4]

def parse_projects(lines: List[str]) -> List[Dict[str, str]]:
    """PERFECT project parsing - CLEARLY SEPARATE from experience"""
    projects = []

    # Clear project indicators
    project_keywords = [
        "project", "projet", "application", "platform", "system", "development",
        "implementation", "built", "created", "developed", "designed", "pfa",
        "pfe", "pidev", "projet de fin", "case study", "prototype"
    ]

    current_project = None
    description_lines = []
    in_project_section = False

    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue

        # Check if we're in projects section
        section, is_header = detect_section_header(clean, "")
        if is_header:
            if section == "Projects":
                in_project_section = True
            elif section in ["Experience", "Education", "Skills"]:
                in_project_section = False

        # Look for project indicators
        is_bullet = re.match(r'^[•\-*]\s', line)
        has_project_kw = any(kw in clean.lower() for kw in project_keywords)
        has_colon = ':' in clean and len(clean.split(':')) > 1
        is_short_name = len(clean) < 80

        # Start new project when we have clear project indicators
        if ((has_project_kw and is_short_name) or
            (is_bullet and is_short_name) or
            (has_colon and is_short_name) or
            (in_project_section and is_short_name)):

            # Save previous project if valid
            if current_project and current_project["name"] and len(current_project["name"]) > 3:
                current_project["description"] = ' '.join(description_lines).strip()[:300]
                projects.append(current_project)

            # Extract project name
            project_name = clean
            if is_bullet:
                project_name = re.sub(r'^[•\-*]\s*', '', project_name)
            if ':' in project_name:
                project_name = project_name.split(':', 1)[0].strip()

            # Clean project name from common prefixes
            project_name = re.sub(
                r'^(?:project|projet|application|platform|system)\s*:\s*',
                '', project_name, flags=re.IGNORECASE
            ).strip()

            current_project = {
                "name": project_name.title(),
                "description": ""
            }
            description_lines = []

            # Extract initial description if available
            if ':' in clean and len(clean.split(':')) > 1:
                desc_part = clean.split(':', 1)[1].strip()
                if len(desc_part) > 8:
                    description_lines.append(desc_part)

            # Look ahead for description (2-4 lines)
            for j in range(i+1, min(i+5, len(lines))):
                next_line = clean_line(lines[j])
                if (len(next_line) > 8 and
                    not re.match(r'^[•\-*]\s', lines[j]) and  # Not another bullet
                    not detect_section_header(next_line, "")[1] and
                    not re.search(DATE_PATTERN, next_line) and  # Not dates (that's experience)
                    len(description_lines) < 3):
                    description_lines.append(next_line)
                else:
                    break

            continue

        # Collect additional description for current project
        elif current_project and len(clean) > 8:
            # Skip if it's clearly a new project or section
            is_new_project = (
                re.match(r'^[•\-*]\s', line) or
                any(kw in clean.lower() for kw in project_keywords) or
                detect_section_header(clean, "")[1]
            )

            if not is_new_project and len(description_lines) < 4:
                description_lines.append(clean)

    # Add final project
    if current_project and current_project["name"] and len(current_project["name"]) > 3:
        current_project["description"] = ' '.join(description_lines).strip()[:300]
        projects.append(current_project)

    # Final validation - remove projects that are actually experiences
    valid_projects = []
    for proj in projects:
        # Filter out experience-like entries
        is_actual_project = (
            not any(exp_kw in proj["name"].lower() for exp_kw in ["engineer", "developer", "intern", "stage"]) and
            not re.search(DATE_PATTERN, proj["name"]) and  # No dates in project names
            len(proj["name"]) > 5 and
            proj["name"].lower() not in ["project", "projet", "projects", "projets"]
        )

        if is_actual_project:
            valid_projects.append(proj)

    return valid_projects[:5]

def parse_skills(raw_text: str) -> List[str]:
    """Parse skills from text"""
    skills_found = set()
    lower_text = raw_text.lower()

    # Exact matching for skills
    for skill in IT_SKILLS.union(GENERAL_SKILLS):
        if re.search(r'\b' + re.escape(skill) + r'\b', lower_text):
            skills_found.add(skill.title())

    # Filter out common false positives
    filtered_skills = []
    for skill in sorted(skills_found):
        skill_lower = skill.lower()
        if (skill_lower not in STOP_WORDS and
            len(skill) > 3 and
            not any(kw in skill_lower for kw in ["years", "year", "level"])):
            filtered_skills.append(skill)

    return filtered_skills[:25]

def parse_certifications(lines: List[str]) -> List[str]:
    """Parse certifications"""
    certs = set()

    cert_keywords = ["certified", "certificate", "certification", "badge", "workshop", "fundamentals"]

    for line in lines:
        clean = clean_line(line)
        clean_lower = clean.lower()

        if any(kw in clean_lower for kw in cert_keywords) and len(clean) > 8:
            # Clean certification text
            cert_text = re.sub(r'(?i)certificat|badge|workshop|certified|certification[:]?\s*', '', clean).strip()
            if len(cert_text) > 5:
                certs.add(cert_text.title())

    return sorted(list(certs))[:5]

def parse_cv(text: str, filename: str = "") -> Dict[str, Any]:
    """PERFECT CV parsing with CLEAR SEPARATION between experience and projects"""
    sections = {
        "Name": "",
        "Contact": {"email": "", "phone": "", "location": "", "url": ""},
        "Summary": "",
        "Skills": [],
        "Education": [],
        "Projects": [],
        "Experience": [],
        "Certifications": [],
        "Interests": []
    }

    if not text:
        return sections

    # Basic text preprocessing
    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    lines = [clean_line(l) for l in text.split('\n') if clean_line(l)]

    # Extract basic information
    sections["Contact"] = extract_contact_info(text)
    sections["Name"] = extract_name(lines, text, sections["Contact"])

    # Section-based parsing
    current_section = None
    section_lines = defaultdict(list)

    # Parse sections
    for line in lines:
        clean = clean_line(line)
        if not clean:
            continue

        # Detect section headers
        new_section, is_header = detect_section_header(clean, current_section)
        if is_header:
            current_section = new_section
            continue

        # Assign lines to sections
        if current_section:
            section_lines[current_section].append(clean)

    # Extract summary from first few non-contact lines
    summary_lines = []
    for i, line in enumerate(lines[:8]):
        clean = clean_line(line)
        if (len(clean) > 20 and
            not re.match(EMAIL_PATTERN, clean) and
            not re.match(PHONE_PATTERN, clean) and
            clean != sections["Name"] and
            not any(section in clean.lower() for section in ["skills", "education", "experience", "projects"])):
            summary_lines.append(clean)

    if summary_lines:
        sections["Summary"] = ' '.join(summary_lines[:3]).strip()[:300]

    # Parse all sections with PERFECT separation
    sections["Skills"] = parse_skills(text)
    sections["Education"] = parse_education(section_lines["Education"])

    # CRITICAL: Parse experience and projects SEPARATELY with clear boundaries
    sections["Experience"] = parse_experience(section_lines["Experience"])
    sections["Projects"] = parse_projects(section_lines["Projects"])

    sections["Certifications"] = parse_certifications(section_lines["Certifications"])

    # Simple interests parsing
    interests = []
    for line in section_lines["Interests"]:
        clean = clean_line(line)
        if len(clean) > 3 and len(clean) < 50 and not re.search(DATE_PATTERN, clean):
            interests.append(clean.title())
    sections["Interests"] = list(set(interests))[:5]

    return sections

def process_pdfs(folder_path: str, output_file: str = "cv_structured_perfect.json", num_to_process: int = 20) -> List[Dict[str, Any]]:
    """Process multiple PDFs and save perfect structured data"""
    results = []
    processed_emails = set()

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")][:num_to_process]

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        logging.info(f"Processing {filename}")

        try:
            text = extract_text_from_pdf(path)
            if not text:
                logging.warning(f"No text extracted from {filename}")
                continue

            cv_data = parse_cv(text, filename)

            # Deduplicate based on email
            email = cv_data["Contact"]["email"]
            if email and email in processed_emails:
                continue

            if email:
                processed_emails.add(email)
            results.append(cv_data)

        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            continue

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logging.info(f"Successfully processed {len(results)} CVs")
    return results

# Main execution
if __name__ == "__main__":
    folder_path = "/content/"
    results = process_pdfs(folder_path)
    print(json.dumps(results, indent=2, ensure_ascii=False))