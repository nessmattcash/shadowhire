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
from transformers import pipeline, AutoTokenizer
import torch
import spacy
import nltk
from seqeval.metrics import classification_report



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
LOCATION_PATTERN = r'(?i)(?:[A-Za-zÀ-ÿ\s]{3,},\s*[A-Za-zÀ-ÿ\s]{3,}|tunisia|tunis|tunisie|ariana|béja|ben arous|bizerte|gabès|gafsa|jendouba|kairouan|kasserine|kebili|kef|mahdia|manouba|medenine|monastir|nabeul|sfax|sidi bouzid|siliana|sousse|tataouine|tozeur|zaghouan)'
URL_PATTERN = r'(?i)(?:https?://)?(?:www\.)?(?:linkedin|github|netlify|portfolio)\.[a-zA-Z0-9./-]+(?:\b|$)'  # Unchanged
DATE_PATTERN = r'(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|present|current|ongoing)|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|present|current|ongoing)|\d{4})'  # Unchanged
BULLET_PATTERN = r'^[-•*››◦▪\u2022\u25CF\s]{1,3}'  # Unchanged
LINK_PATTERN = r'(?i)(\\s?)?(?:github|link|website|portfolio|\)'  # Unchanged

# EXPANDED Section patterns for better coverage
SECTION_PATTERNS = {
    "Summary": ["summary", "profile", "about me", "profil", "objectif", "résumé", "career objective", "professional summary", "core competencies", "overview", "bio", "introduction", "personal statement", "executive summary", "career profile", "profil personnel"],
    "Skills": ["skills", "technical skills", "compétences", "competences", "expertise", "langages et frameworks", "technologies", "core skills", "technical proficiencies", "abilities", "key skills", "proficiencies", "competences techniques", "hard skills", "soft skills", "compétences techniques", "langages de programmation"],
    "Education": ["education", "formation", "parcours académique", "études", "academic background", "qualifications", "academic record", "degree", "degrees", "schooling", "academic qualifications", "studies", "diploma", "baccalaureate", "certifications académiques", "formation académique", "parcours scolaire"],
    "Experience": ["experience", "expérience", "work experience", "expérience professionnelle", "professional experience", "stage", "internship", "employment history", "work history", "intern", "professional background", "career history", "achievements", "tasks", "taches realisees", "responsibilities", "professional history", "stages", "job history", "expériences professionnelles"],
    "Projects": ["projects", "projets", "portfolio", "projets personnels", "notable projects", "projet de fin d'étude", "pfa", "pidev", "pi", "projets académiques", "personal projects", "academic projects", "key projects", "pfe", "case studies", "projet académique"],
    "Certifications": ["certifications", "certificats", "badges", "certificates and badges", "achievements", "awards", "professional certifications", "certificates", "badges and certifications", "honors", "certificats et badges", "certifs", "accreditations", "certificats de réussite"],
    "Interests": ["interests", "hobbies", "intérêts", "loisirs", "personal interests", "extracurricular", "activites extracurriculaires", "volunteer work", "centres d'intérêt", "activités"]
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
    """ULTIMATE PDF extraction with aggressive space insertion for concatenated text"""
    try:
        with pdfplumber.open(path) as pdf:
            text = ""
            for page in pdf.pages:
                # Enhanced text extraction with better layout preservation
                page_text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2)
                if page_text:
                    # AGGRESSIVE space insertion for common concatenation patterns
                    # Fix: wordWORD -> word WORD
                    page_text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', page_text)
                    # Fix: WORDWord -> WORD Word  
                    page_text = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', page_text)
                    # Fix: letterNumber -> letter Number
                    page_text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', page_text)
                    # Fix: numberLetter -> number Letter
                    page_text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', page_text)
                    # Fix: Special character concatenation
                    page_text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', page_text)
                    # Fix: Common CV concatenations
                    page_text = re.sub(r'(Summary|Experience|Education|Skills|Projects)([A-Z])', r'\1 \2', page_text)
                    
                    text += page_text + "\n"
                
                # Enhanced table extraction with better cell handling
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        clean_cells = []
                        for cell in row:
                            if cell and str(cell).strip():
                                cell_text = str(cell).strip()
                                # Apply same space fixes to table cells
                                cell_text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', cell_text)
                                clean_cells.append(cell_text)
                        if clean_cells:
                            text += ' | '.join(clean_cells) + "\n"
            
            # Fallback to pdfminer if pdfplumber fails
            if not text.strip():
                text = pdfminer_extract(path)
                # Apply same fixes to pdfminer text
                text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
            
            # Final cleanup
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
    except Exception as e:
        logging.error(f"PDF extraction failed for {path}: {e}")
        text = pdfminer_extract(path)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

def load_ner_model(model_path: str = "./fine_tuned_resume_ner/fine_tuned_resume_ner"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True
        )
        nlp = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max",  # Changed to "max" to boost multi-token entity scores
            device=0 if torch.cuda.is_available() else -1
        )
        logging.info(f"Successfully loaded NER model from {model_path}")
        return nlp
    except Exception as e:
        logging.error(f"Failed to load NER model from {model_path}: {str(e)}")
        raise

def process_text_in_chunks(nlp_pipeline, text, chunk_size=500, stride=100):
    results = []
    tokens = AutoTokenizer.from_pretrained("./fine_tuned_resume_ner/fine_tuned_resume_ner", local_files_only=True)(text, return_offsets_mapping=True, truncation=False)
    total_length = len(text)
    
    for i in range(0, total_length, chunk_size - stride):
        chunk = text[i:i + chunk_size]
        chunk_results = nlp_pipeline(chunk)
        for res in chunk_results:
            if res['score'] > 0.2:  # Lowered threshold for debugging
                res['start'] += i
                res['end'] += i
                results.append(res)
    
    # Merge overlapping entities
    merged_results = []
    seen_spans = set()
    for res in sorted(results, key=lambda x: x['start']):
        span = (res['start'], res['end'])
        if span not in seen_spans:
            merged_results.append(res)
            seen_spans.add(span)
    
    return merged_results


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
    """ULTIMATE contact info extraction with better filtering"""
    # Extract emails
    emails = list(set(re.findall(EMAIL_PATTERN, raw_text)))
    
    # Enhanced phone extraction - exclude dates and numbers that look like years
    phones = []
    phone_matches = re.finditer(PHONE_PATTERN, raw_text)
    for match in phone_matches:
        phone = match.group().strip()
        digits = re.sub(r'\D', '', phone)
        # Better validation - exclude years and invalid lengths
        if (8 <= len(digits) <= 15 and 
            not re.search(r'(19|20)\d{2}', phone) and  # Exclude years like 2023, 2024
            not re.search(r'\b\d{4}\b', phone)):       # Exclude standalone 4-digit numbers
            phones.append(phone)
    
    # ENHANCED location extraction - look in first 15 lines with better filtering
    locations = []
    lines = raw_text.split('\n')[:15]
    location_blacklist = ["classification", "tensorflow", "flask", "transformers", "engineer", "developer", "summary"]
    
    for line in lines:
        clean = clean_line(line)
        if len(clean) > 5 and len(clean) < 100:  # Reasonable location length
            location_match = re.search(LOCATION_PATTERN, clean, re.IGNORECASE)
            if location_match:
                loc = location_match.group().strip()
                loc_lower = loc.lower()
                # Stronger filtering
                if (len(loc) > 3 and 
                    not any(bad in loc_lower for bad in location_blacklist) and
                    not re.match(EMAIL_PATTERN, loc) and
                    not re.match(PHONE_PATTERN, loc) and
                    not any(word in loc_lower for word in ['stage', 'intern', 'engineer', 'developer'])):
                    locations.append(loc)
    
    # Extract URLs with better filtering
    urls = []
    url_matches = re.finditer(URL_PATTERN, raw_text)
    for match in url_matches:
        url = match.group().strip()
        if len(url) > 10 and not any(keyword in url.lower() for keyword in ['example', 'template']):
            urls.append(url)
    
    return {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "location": locations[0] if locations else "",
        "url": urls[0] if urls else ""
    }
def extract_name(lines: List[str], raw_text: str, contact_info: Dict[str, str], nlp: pipeline = None) -> str:
    """ULTIMATE name extraction with multiple strategies including proper NER"""
    name_blacklist = STOP_WORDS + ["summary", "profile", "contact", "curriculum", "vitae", "resume", "cv"]
    
    # Strategy 1: Look for name patterns in first 3-5 lines (MOST RELIABLE)
    for i, line in enumerate(lines[:5]):
        clean = clean_line(line)
        clean_lower = clean.lower()
        
        # Strong name validation
        is_name_candidate = (
            len(clean) > 5 and len(clean) < 50 and
            re.match(NAME_PATTERN, clean) and 
            2 <= len(clean.split()) <= 4 and
            not any(kw in clean_lower for kw in name_blacklist) and
            not re.match(EMAIL_PATTERN, clean) and 
            not re.match(PHONE_PATTERN, clean) and
            not re.match(LOCATION_PATTERN, clean) and
            not any(word in clean_lower for word in ['linkedin', 'github', 'http', 'www']) and
            not clean[0].islower()  # Names usually start with capital
        )
        
        if is_name_candidate:
            return clean.title()
    
    # Strategy 2: Extract from email (intelligent parsing)
    email = contact_info.get("email", "")
    if email and "@" in email:
        username = email.split('@')[0]
        # Remove numbers and special characters, keep only name parts
        name_parts = re.split(r'[\._\-\d+]', username)
        valid_parts = [part for part in name_parts if len(part) > 2 and part.isalpha()]
        if 2 <= len(valid_parts) <= 3:
            potential_name = ' '.join(valid_parts).title()
            # Validate it looks like a real name
            if re.match(NAME_PATTERN, potential_name):
                return potential_name
    
    # Strategy 3: Use NER with proper integration
    if nlp:
        try:
            # Use first 1000 characters for name detection
            ner_results = nlp(raw_text[:1000])
            name_candidates = []
            
            for ent in ner_results:
                if ent.get('entity_group') == 'NAME' and ent['score'] > 0.7:  # Reasonable threshold
                    name_text = ent['word'].strip()
                    # Clean and validate the NER result
                    name_text = re.sub(r'[^a-zA-ZÀ-ÿ\s\-]', '', name_text).strip()
                    
                    if (re.match(NAME_PATTERN, name_text) and
                        2 <= len(name_text.split()) <= 4 and
                        len(name_text) > 5 and
                        not any(kw in name_text.lower() for kw in name_blacklist)):
                        name_candidates.append((name_text, ent['score']))
            
            # Return the highest confidence valid name
            if name_candidates:
                name_candidates.sort(key=lambda x: x[1], reverse=True)
                return name_candidates[0][0].title()
                
        except Exception as e:
            logging.debug(f"NER name extraction failed: {e}")
    
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
    """Enhanced education parser with better institution detection"""
    entries = []
    
    education_keywords = [
        "bachelor", "master", "phd", "diploma", "degree", "licence", "engineering", 
        "esprit", "ensit", "university", "college", "institute", "school", "faculty",
        "preparatory", "cycle", "computer science", "maths", "physics", "studies",
        "nationale", "supérieure", "école", "faculté", "institut"
    ]
    
    institution_keywords = [
        "esprit", "ensit", "isi", "iset", "issat", "issatm", "fst", "ensi", "enit",
        "université", "university", "college", "institute", "école", "faculty",
        "lycée", "high school", "school"
    ]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        clean = clean_line(line)
        
        if len(clean) < 8:
            i += 1
            continue
        
        clean_lower = clean.lower()
        
        # Look for education indicators
        has_edu_kw = any(kw in clean_lower for kw in education_keywords)
        date_match = re.search(DATE_PATTERN, clean)
        has_edu_structure = any(indicator in clean for indicator in [' - ', ' at ', ' : ', ' | '])
        
        if has_edu_kw or (date_match and has_edu_structure):
            entry = {"degree": "", "institution": "", "duration": ""}
            
            # Extract duration
            if date_match:
                entry["duration"] = date_match.group()
                clean = re.sub(DATE_PATTERN, '', clean).strip()
            
            # ENHANCED institution extraction
            institution_found = ""
            degree_found = clean
            
            # Look for institution patterns
            for inst_kw in institution_keywords:
                if inst_kw in clean_lower:
                    # Find the institution part
                    inst_match = re.search(rf'.*{re.escape(inst_kw)}.*', clean, re.IGNORECASE)
                    if inst_match:
                        institution_found = inst_match.group().strip()
                        # Remove institution from degree
                        degree_found = re.sub(rf'.*{re.escape(inst_kw)}.*', '', clean).strip()
                        break
            
            # If no institution found by keyword, use separators
            if not institution_found:
                if " - " in clean:
                    parts = clean.split(" - ", 1)
                    # Determine which part is institution vs degree
                    if any(edu_kw in parts[0].lower() for edu_kw in education_keywords):
                        degree_found = parts[0].strip()
                        institution_found = parts[1].strip()
                    else:
                        institution_found = parts[0].strip()
                        degree_found = parts[1].strip()
                elif " at " in clean_lower:
                    parts = clean.split(" at ", 1)
                    degree_found = parts[0].strip()
                    institution_found = parts[1].strip()
                elif " : " in clean:
                    parts = clean.split(" : ", 1)
                    degree_found = parts[0].strip()
                    institution_found = parts[1].strip()
                elif " | " in clean:
                    parts = clean.split(" | ", 1)
                    degree_found = parts[0].strip()
                    institution_found = parts[1].strip()
            
            entry["degree"] = degree_found.title() if degree_found else ""
            entry["institution"] = institution_found.title() if institution_found else ""
            
            # Look ahead for additional institution info (next line)
            if i + 1 < len(lines) and not entry["institution"]:
                next_line = clean_line(lines[i + 1])
                if len(next_line) > 5 and any(inst_kw in next_line.lower() for inst_kw in institution_keywords):
                    entry["institution"] = next_line.title()
                    i += 1  # Skip next line since we used it
            
            # Validate entry has meaningful content
            if (entry["degree"] and len(entry["degree"]) > 5 and 
                not any(kw in entry["degree"].lower() for kw in ['current', 'present']) and
                not re.match(PHONE_PATTERN, entry["degree"]) and
                not re.match(EMAIL_PATTERN, entry["degree"])):
                entries.append(entry)
        
        i += 1
    
    return entries[:3]  # Return max 3 most recent education entries
def parse_experience(lines: List[str], nlp: pipeline = None) -> List[Dict[str, str]]:
    """IMPROVED experience parser with NER integration for role/company detection"""
    experiences = []
    
    experience_keywords = [
        "engineer", "developer", "analyst", "manager", "specialist", "consultant", 
        "architect", "director", "lead", "intern", "stage", "stagiare", "employment",
        "work experience", "professional experience", "expérience professionnelle",
        "internship", "volunteering", "hr manager", "member", "network engineering"
    ]
    
    current_exp = None
    description_lines = []
    in_experience_section = False
    
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        
        # Detect experience section
        section, is_header = detect_section_header(clean, "")
        if is_header:
            if section == "Experience":
                in_experience_section = True
                # Save current experience
                if current_exp and current_exp["role"]:
                    current_exp["description"] = ' '.join(description_lines).strip()[:500]
                    if len(current_exp["description"]) > 15:
                        experiences.append(current_exp)
                    current_exp = None
                    description_lines = []
            elif section in ["Projects", "Education", "Skills"]:
                in_experience_section = False
                # Save current experience
                if current_exp and current_exp["role"]:
                    current_exp["description"] = ' '.join(description_lines).strip()[:500]
                    if len(current_exp["description"]) > 15:
                        experiences.append(current_exp)
                    current_exp = None
                    description_lines = []
            continue
        
        clean_lower = clean.lower()
        
        # IMPROVED EXPERIENCE DETECTION WITH NER SUPPORT
        date_match = re.search(DATE_PATTERN, clean)
        has_role_keyword = any(kw in clean_lower for kw in experience_keywords)
        is_short_line = len(clean) < 150
        
        # NER-BASED EXPERIENCE DETECTION
        has_ner_experience = False
        if nlp and is_short_line and (date_match or in_experience_section):
            try:
                ner_results = nlp(clean)
                for ent in ner_results:
                    if ent.get('entity_group') in ['ROLE', 'ORGANIZATION'] and ent['score'] > 0.5:
                        has_ner_experience = True
                        break
            except:
                pass
        
        # Start new experience on clear indicators
        experience_indicator = (
            (date_match and has_role_keyword and is_short_line) or
            (has_ner_experience and is_short_line) or
            (date_match and in_experience_section and is_short_line)
        )
        
        if experience_indicator:
            # Save previous experience
            if current_exp and current_exp["role"] and len(current_exp["role"]) > 3:
                current_exp["description"] = ' '.join(description_lines).strip()[:500]
                if len(current_exp["description"]) > 15:
                    experiences.append(current_exp)
            
            # Extract role, company, duration with NER enhancement
            duration = date_match.group() if date_match else ""
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            
            # Use NER to help with role/company extraction
            role = remaining
            company = ""
            location = ""
            
            if nlp:
                try:
                    ner_results = nlp(remaining)
                    role_parts = []
                    company_parts = []
                    
                    for ent in ner_results:
                        if ent.get('entity_group') == 'ROLE' and ent['score'] > 0.4:
                            role_parts.append(ent['word'])
                        elif ent.get('entity_group') == 'ORGANIZATION' and ent['score'] > 0.4:
                            company_parts.append(ent['word'])
                    
                    if role_parts:
                        role = ' '.join(role_parts)
                        # Remove role from remaining to extract company
                        for role_part in role_parts:
                            remaining = remaining.replace(role_part, '').strip()
                    
                    if company_parts:
                        company = ' '.join(company_parts)
                        
                except Exception as e:
                    logging.debug(f"NER experience parsing failed: {e}")
            
            # Fallback to traditional separation if NER didn't help enough
            if not role or len(role) < 3:
                role = remaining
                company = ""
                
            separators = [" at ", " - ", " : ", ", ", " | ", " – "]
            for sep in separators:
                if sep in role:
                    parts = role.split(sep, 1)
                    role = parts[0].strip()
                    company_location = parts[1].strip()
                    
                    # Extract location from company_location if present
                    location_match = re.search(r'([A-Za-zÀ-ÿ\s]+(?:,\s*[A-Za-zÀ-ÿ\s]+)?)$', company_location)
                    if location_match:
                        location = location_match.group(1).strip()
                        company = re.sub(r'([A-Za-zÀ-ÿ\s]+(?:,\s*[A-Za-zÀ-ÿ\s]+)?)$', '', company_location).strip()
                    else:
                        company = company_location
                    break
            
            # Final validation and cleaning
            role = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s\-&]', '', role).strip()
            company = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s\-&]', '', company).strip()
            
            # Validate it's actually a role (not a description)
            is_valid_role = (
                len(role) > 5 and
                (any(role_indicator in role.lower() for role_indicator in experience_keywords) or
                 any(word in role.lower() for word in ['intern', 'stage', 'volunteer'])) and
                not role.lower().startswith(('built', 'created', 'developed', 'managed', 'led'))
            )
            
            if is_valid_role:
                current_exp = {
                    "role": role.title(),
                    "company": company.title(),
                    "duration": duration,
                    "location": location,
                    "description": ""
                }
                description_lines = []
                
                # Look ahead for description (2-3 lines only)
                look_ahead = []
                for j in range(i+1, min(i+4, len(lines))):
                    next_line = clean_line(lines[j])
                    if (len(next_line) > 10 and 
                        not re.search(DATE_PATTERN, next_line) and
                        not detect_section_header(next_line, "")[1] and
                        len(look_ahead) < 2):
                        look_ahead.append(next_line)
                    else:
                        break
                
                description_lines.extend(look_ahead)
                continue
        
        # Collect description ONLY if we have current experience
        elif current_exp and len(clean) > 10:
            # Check for new experience
            is_new_exp = (
                re.search(DATE_PATTERN, clean) and 
                any(kw in clean_lower for kw in experience_keywords)
            )
            
            if not is_new_exp and len(description_lines) < 5:
                description_lines.append(clean)
    
    # Add final experience
    if current_exp and current_exp["role"] and len(current_exp["role"]) > 3:
        current_exp["description"] = ' '.join(description_lines).strip()[:500]
        if len(current_exp["description"]) > 15:
            experiences.append(current_exp)
    
    return experiences[:4]

def parse_projects(lines: List[str], nlp: pipeline = None) -> List[Dict[str, str]]:
    """ULTIMATE project parser with NER integration for better project name detection"""
    projects = []
    
    project_keywords = [
        "project", "projet", "application", "platform", "system", "development",
        "implementation", "built", "created", "developed", "designed", "pfa", 
        "pfe", "pidev", "case study", "prototype", "simulation", "tracking",
        "detection", "prediction", "gestion", "plateforme", "agent", "recruitment",
        "donation", "management", "skillswap", "blooder", "biometric", "shadowhire",
        "cloud-naturelink", "picocloud", "splunk", "soc", "anemia", "healthcare",
        "security", "web", "mobile", "desktop", "automation", "hackathon", "eventmatch",
        "parky", "psychwell", "zedney", "9antra", "lambda", "super store"
    ]
    
    # EXPANDED BAD PROJECT NAMES to exclude
    bad_names = {
        "artificielle", "mots-cles", "keywords", "features", "include", "description",
        "technologies", "tools", "built", "created", "developed", "implementation",
        "achievements", "tasks", "responsibilities", "duties", "english", "french",
        "arabic", "professional", "proficiency", "contact", "summary", "github", "link"
    }
    
    current_project = None
    description_lines = []
    in_projects_section = False
    
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean or len(clean) < 5:
            continue
        
        clean_lower = clean.lower()
        
        # Detect project section
        section, is_header = detect_section_header(clean, "")
        if is_header:
            if section == "Projects":
                in_projects_section = True
                # Save current project when entering section
                if current_project and current_project["name"]:
                    current_project["description"] = ' '.join(description_lines).strip()[:400]
                    if len(current_project["description"]) > 20:
                        projects.append(current_project)
                    current_project = None
                    description_lines = []
            elif section in ["Experience", "Education", "Skills"]:
                in_projects_section = False
                # Save current project when leaving section
                if current_project and current_project["name"]:
                    current_project["description"] = ' '.join(description_lines).strip()[:400]
                    if len(current_project["description"]) > 20:
                        projects.append(current_project)
                    current_project = None
                    description_lines = []
            continue
        
        # ULTIMATE PROJECT DETECTION WITH NER
        is_bullet = re.match(r'^[•\-*››◦▪\u2022]', line.strip())
        has_project_kw = any(keyword in clean_lower for keyword in project_keywords)
        has_year = re.search(r'\(\d{4}\)', clean) or re.search(r'\d{4}\s*[-–—]', clean)
        is_short_line = len(clean) < 150
        
        # NER-BASED PROJECT DETECTION
        has_ner_project = False
        project_name_ner = ""
        if nlp and is_short_line and (has_project_kw or in_projects_section or is_bullet):
            try:
                ner_results = nlp(clean)
                for ent in ner_results:
                    if ent.get('entity_group') == 'PROJECT' and ent['score'] > 0.4:
                        has_ner_project = True
                        project_name_ner = ent['word'].strip()
                        break
            except Exception as e:
                logging.debug(f"NER project detection failed: {e}")
        
        # STRONG PROJECT INDICATORS
        project_indicator = (
            (has_project_kw and is_short_line) or
            (is_bullet and is_short_line and in_projects_section) or
            (has_year and is_short_line and in_projects_section) or
            (re.search(r'^[•\-*]?\s*[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s*[\(\:\-]', clean)) or
            (clean_lower.startswith(tuple(project_keywords))) or
            (re.search(r'^(pfa|pfe|pidev|skillswap|blooder|biometric|shadowhire|splunk|anemia|healthcare|lambda|super store|eventmatch|parky|psychwell|zedney|9antra)', clean_lower)) or
            has_ner_project
        )
        
        if project_indicator:
            # Save previous project
            if current_project and current_project["name"] and len(current_project["name"]) > 3:
                current_project["description"] = ' '.join(description_lines).strip()[:400]
                if len(current_project["description"]) > 20:
                    projects.append(current_project)
            
            # ULTIMATE PROJECT NAME EXTRACTION WITH NER PRIORITY
            if has_ner_project and project_name_ner:
                project_name = project_name_ner
            else:
                project_name = clean
            
            # Remove bullet points and numbering
            project_name = re.sub(r'^[•\-*››◦▪\u2022]\s*', '', project_name)
            project_name = re.sub(r'^\d+[\.\)]\s*', '', project_name)
            
            # Extract name before colon/dash/parenthesis if present
            separators = [':', ' - ', ' – ', ' — ', ' (', ' | ']
            for sep in separators:
                if sep in project_name:
                    project_name = project_name.split(sep, 1)[0].strip()
                    break
            
            # Remove year parentheses but keep the text
            project_name = re.sub(r'\s*\(\d{4}\)\s*', ' ', project_name)
            project_name = re.sub(r'\s*\d{4}\s*[-–—]\s*\d{4}\s*', ' ', project_name)
            project_name = re.sub(r'\s*\d{4}\s*[-–—]\s*(?:present|current)\s*', ' ', project_name)
            
            # Remove common prefixes but preserve actual project names
            project_name = re.sub(r'^(?:project|projet|application|platform|system|development|stage|internship)[\s:\-]*', '', 
                                project_name, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            project_name = re.sub(r'\s+', ' ', project_name).strip()
            
            # ULTIMATE VALIDATION
            is_valid_name = (
                len(project_name) > 5 and
                len(project_name) < 100 and
                not any(bad_name in project_name.lower() for bad_name in bad_names) and
                not project_name.lower().startswith(('built', 'created', 'developed', 'features', 'technologies', 'tools')) and
                not re.search(r'^(?:mots-cles|keywords|technologies|tools|achievements|responsibilities)', project_name.lower()) and
                len(project_name.split()) <= 12 and
                not re.match(PHONE_PATTERN, project_name) and
                not re.match(EMAIL_PATTERN, project_name) and
                not project_name.isdigit() and
                not all(len(word) == 1 for word in project_name.split())  # Not all single letters
            )
            
            if is_valid_name:
                current_project = {
                    "name": project_name.title(),
                    "description": ""
                }
                description_lines = []
                
                # Extract initial description from current line
                for sep in separators:
                    if sep in clean:
                        desc_part = clean.split(sep, 1)[1].strip()
                        if len(desc_part) > 10:
                            # Clean description part
                            desc_part = re.sub(r'^(mots-cles|keywords|technologies)[\s:\-]*', '', desc_part, flags=re.IGNORECASE)
                            description_lines.append(desc_part)
                        break
                
                # Look ahead for description (3-6 lines)
                look_ahead = []
                for j in range(i+1, min(i+7, len(lines))):
                    next_clean = clean_line(lines[j])
                    if (len(next_clean) > 8 and 
                        not re.match(r'^[•\-*››◦▪\u2022]', lines[j].strip()) and
                        not detect_section_header(next_clean, "")[1] and
                        len(look_ahead) < 5):
                        look_ahead.append(next_clean)
                    else:
                        break
                
                description_lines.extend(look_ahead)
                continue
        
        # Collect description for current project
        elif current_project and len(clean) > 8:
            # Skip if it's clearly a new project or section
            is_new_project = (
                re.match(r'^[•\-*››◦▪\u2022]', line.strip()) or
                detect_section_header(clean, "")[1] or
                (any(keyword in clean_lower for keyword in project_keywords) and len(clean) < 120)
            )
            
            if not is_new_project and len(description_lines) < 8:
                description_lines.append(clean)
    
    # Add final project
    if current_project and current_project["name"] and len(current_project["name"]) > 3:
        current_project["description"] = ' '.join(description_lines).strip()[:400]
        if len(current_project["description"]) > 20:
            projects.append(current_project)
    
    # ULTIMATE DEDUPLICATION AND VALIDATION
    unique_projects = []
    seen_names = set()
    
    for project in projects:
        name_lower = project["name"].lower()
        # Final validation - ensure it's a real project name
        is_real_project = (
            name_lower not in seen_names and
            not any(bad_name in name_lower for bad_name in bad_names) and
            len(project["name"]) > 5 and
            len(project["description"]) > 25 and
            not re.match(r'^[0-9\s\-–—]+$', project["name"]) and
            not project["name"].lower().startswith(('http', 'www', 'github'))
        )
        
        if is_real_project:
            seen_names.add(name_lower)
            unique_projects.append(project)

    return unique_projects[:8]

def parse_skills(raw_text: str, nlp: pipeline = None) -> List[str]:
    """Perfect skills parser with ZERO garbage using dual NER + logic validation"""
    skills_found = set()
    lower_text = raw_text.lower()
    
    # METHOD 1: NER-based extraction (Primary)
    if nlp:
        try:
            ner_results = nlp(raw_text[:2500])
            for ent in ner_results:
                if ent.get('entity_group') in ['SKILL', 'SKILLS']:
                    skill_text = ent['word'].strip()
                    # Aggressive cleaning
                    skill_text = re.sub(r'^##|[^a-zA-Z0-9+#\.]', '', skill_text)
                    skill_lower = skill_text.lower()
                    
                    # STRICT VALIDATION: Must be in known skills list
                    is_valid_skill = (
                        2 <= len(skill_text) <= 35 and
                        not skill_text.isdigit() and
                        skill_lower not in STOP_WORDS and
                        not any(kw in skill_lower for kw in ['years', 'year', 'level', 'proficiency', 'native']) and
                        not re.search(r'\d{4}', skill_text) and
                        any(known_skill in skill_lower for known_skill in IT_SKILLS)  # MUST match known skills
                    )
                    
                    if is_valid_skill:
                        # Find the best matching known skill
                        for known_skill in IT_SKILLS:
                            if known_skill in skill_lower:
                                skills_found.add(known_skill.title())
                                break
        except Exception as e:
            logging.debug(f"NER skill extraction: {str(e)}")
    
    # METHOD 2: Direct keyword matching (Fallback)
    for skill in IT_SKILLS.union(GENERAL_SKILLS):
        pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
        if re.search(pattern, lower_text):
            skills_found.add(skill.title())
    
    # METHOD 3: Extract from skills section with validation
    skills_sections = re.findall(r'(?i)(?:skills?|compétences|technologies?)[\s:\-]*(.*?)(?=(?:education|experience|projects|certifications|$))', raw_text, re.DOTALL)
    
    for section in skills_sections:
        lines = section.split('\n')
        for line in lines:
            clean_line = re.sub(r'[•\-*▪›◦\u2022]', ' ', line).strip()
            if 3 < len(clean_line) < 100:
                # Split and validate each part
                parts = re.split(r'[:,;/|]', clean_line)
                for part in parts:
                    part = part.strip()
                    if 2 < len(part) < 50:
                        part_lower = part.lower()
                        # Only add if it matches known skills
                        for known_skill in IT_SKILLS:
                            if known_skill in part_lower:
                                skills_found.add(known_skill.title())
                                break
    
    # FINAL AGGRESSIVE FILTERING
    filtered_skills = []
    garbage_terms = {'stack', 'end', 'form', 'isco', 'kroc', 'ops', 'cer', 'i / cd', 'languages', 
                    'tools', 'platforms', 'areas', 'programming', 'frameworks', 'cloud', 'devops',
                    'bern', 'ible', 'jang', 'net', 'fiber', 'isco', 'kroc'}
    
    for skill in sorted(skills_found):
        skill_lower = skill.lower()
        
        # ULTRA-STRICT FILTERING
        is_valid = (
            len(skill) >= 3 and 
            skill_lower not in STOP_WORDS and
            skill_lower not in garbage_terms and
            not skill.isdigit() and
            not re.match(r'^[^a-zA-Z]', skill) and
            not re.search(r'\d{4}', skill) and
            not any(len(word) == 1 for word in skill.split()) and
            len(skill) <= 30 and
            any(known_skill in skill_lower for known_skill in IT_SKILLS)  # Final validation
        )
        
        if is_valid:
            filtered_skills.append(skill)
    
    return filtered_skills[:25]
def parse_certifications(lines: List[str], nlp: pipeline = None, raw_text: str = "") -> List[str]:
    """FINAL BALANCED certification parser - finds ALL real certs without being too strict"""
    certs = set()
    
    # Strategy 1: Section-based extraction (MOST RELIABLE)
    cert_section_patterns = [
        r'(?i)(?:certifications?|certificats?|certificates?|certifs?)[\s:\-]*(.*?)(?=(?:\n[A-Z][a-z]|\n\s*\n|\n\w{3,}|$))',
        r'(?i)(?:badges?|credentials?|accreditations?)[\s:\-]*(.*?)(?=(?:\n[A-Z][a-z]|\n\s*\n|\n\w{3,}|$))'
    ]
    
    for pattern in cert_section_patterns:
        sections = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        for section in sections:
            section_certs = extract_certs_from_section_balanced(section)
            certs.update(section_certs)
    
    # Strategy 2: Line-by-line analysis with BALANCED approach
    for line in lines:
        clean = clean_line(line)
        if not clean or len(clean) < 6 or len(clean) > 200:
            continue
            
        clean_lower = clean.lower()
        
        # Check if line contains multiple certifications
        if contains_multiple_certifications(clean):
            separated_certs = separate_merged_certifications(clean)
            for cert in separated_certs:
                cleaned_cert = clean_certification_balanced(cert)
                if cleaned_cert and is_plausible_certification(cleaned_cert):
                    certs.add(cleaned_cert)
        else:
            # BROAD certification detection - be more inclusive
            is_cert = (
                re.search(r'(?i)\b(certified|certification|certificate|certificat|cert)\b', clean_lower) or
                re.search(r'(?i)\b(associate|professional|specialist|expert|master|foundation|fundamental)\b', clean_lower) or
                any(provider in clean_lower for provider in CERTIFICATION_PROVIDERS) or
                re.search(r'(AZ-\d+|AWS-|Google Cloud|CCNA|CCNP|CISSP|PMP|CEH|OSCP)', clean, re.IGNORECASE) or
                # Include training courses that are commonly listed as certs
                (re.search(r'(?i)(training course|challenges|fundamentals|deep learning)', clean_lower) and 
                 any(provider in clean_lower for provider in ['kodekloud', 'nvidia', 'coursera']))
            )
            
            # MINIMAL garbage filtering - only obvious non-certs
            is_garbage = (
                re.search(r'@|\+216|\d{8,}', clean) or  # Contact info
                clean_lower.startswith(('http', 'www', 'linkedin', 'github')) or
                # Only filter obvious language proficiency
                (any(lang in clean_lower for lang in ['english', 'french', 'arabic']) and 
                 'proficiency' in clean_lower and 
                 not any(cert_word in clean_lower for cert_word in ['certif', 'associate', 'professional']))
            )
            
            if is_cert and not is_garbage:
                cleaned_cert = clean_certification_balanced(clean)
                if cleaned_cert and is_plausible_certification(cleaned_cert):
                    certs.add(cleaned_cert)
    
    # Strategy 3: Bullet point extraction
    bullet_certs = extract_certs_from_bullets_balanced(lines)
    certs.update(bullet_certs)
    
    # Strategy 4: NER with lower threshold
    if nlp and raw_text:
        try:
            ner_results = nlp(raw_text[:4000])
            for ent in ner_results:
                if ent.get('entity_group') == 'CERTIFICATION' and ent['score'] > 0.5:  # Lower threshold
                    cert_text = ent['word'].strip()
                    cleaned_cert = clean_certification_balanced(cert_text)
                    if cleaned_cert and is_plausible_certification(cleaned_cert):
                        certs.add(cleaned_cert)
        except Exception as e:
            logging.debug(f"NER certification failed: {e}")
    
    # FINAL BALANCED CLEANING
    final_certs = []
    seen_certs = set()
    
    for cert in sorted(certs):
        cert_lower = cert.lower()
        
        # BALANCED VALIDATION - much less strict
        is_valid = (
            len(cert) >= 6 and len(cert) <= 100 and
            cert_lower not in seen_certs and
            not re.search(r'@|\+216|\d{8,}', cert) and  # Basic contact filter only
            len(cert.split()) >= 1 and len(cert.split()) <= 8 and  # More flexible
            cert[0].isupper() and
            # Must contain SOME certification indicators (not too strict)
            has_certification_indicators(cert_lower) and
            # Must NOT contain obvious garbage
            not contains_obvious_garbage(cert_lower)
        )
        
        if is_valid:
            # Gentle deduplication
            is_duplicate = False
            for seen_cert in seen_certs:
                if fuzz.ratio(cert_lower, seen_cert) > 85:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                final_certs.append(cert)
                seen_certs.add(cert_lower)
    
    return final_certs[:8]

def contains_multiple_certifications(text: str) -> bool:
    """Check if text contains multiple merged certifications"""
    text_lower = text.lower()
    
    # Patterns indicating multiple certs
    multiple_indicators = [
        r'kodekloud.*kodekloud',  # Repeated provider
        r'docker.*kubernetes',     # Multiple technologies
        r'azure.*aws.*google',     # Multiple cloud providers
        r'cisco.*cisco',           # Repeated vendor
        r'\b\w+.*\b\w+.*certificat', # Multiple words before certificate
    ]
    
    return any(re.search(pattern, text_lower) for pattern in multiple_indicators)

def separate_merged_certifications(text: str) -> List[str]:
    """Separate merged certifications into individual ones"""
    separators = [
        r'\s+•\s+', r'\s+\s+', r'\s+-\s+', r'\s+–\s+', r'\s+—\s+',
        r'\s*\|\s*', r'\s*,\s*', r'\s+and\s+', r'\s+&\s+'
    ]
    
    # Try different separators
    for sep in separators:
        if re.search(sep, text):
            parts = re.split(sep, text)
            if len(parts) > 1:
                return [part.strip() for part in parts if len(part.strip()) > 5]
    
    # If no separators found, try to split by repeated providers
    providers = ['kodekloud', 'cisco', 'microsoft', 'aws', 'azure', 'google', 'nvidia']
    for provider in providers:
        pattern = r'(.+?' + re.escape(provider) + r'.+?)(?=' + re.escape(provider) + r'|$)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        if len(matches) > 1:
            return matches
    
    return [text]  # Return original if can't separate

def extract_certs_from_section_balanced(section_text: str) -> List[str]:
    """Extract certifications from section with balanced cleaning"""
    certs = set()
    
    lines = section_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 6 or len(line) > 120:
            continue
        
        # Check for multiple certifications first
        if contains_multiple_certifications(line):
            separated_certs = separate_merged_certifications(line)
            for cert in separated_certs:
                cleaned_cert = clean_certification_balanced(cert)
                if cleaned_cert and is_plausible_certification(cleaned_cert):
                    certs.add(cleaned_cert)
        else:
            # Single certification
            cleaned_cert = clean_certification_balanced(line)
            if cleaned_cert and is_plausible_certification(cleaned_cert):
                certs.add(cleaned_cert)
    
    return list(certs)

def extract_certs_from_bullets_balanced(lines: List[str]) -> List[str]:
    """Extract certifications from bullet points with balanced approach"""
    certs = set()
    
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean or len(clean) < 6:
            continue
        
        # Check if it's a bullet point
        is_bullet = re.match(r'^[•\-*››◦▪\u2022]\s*', line.strip())
        
        if is_bullet and len(clean) < 120:
            if contains_multiple_certifications(clean):
                separated_certs = separate_merged_certifications(clean)
                for cert in separated_certs:
                    cleaned_cert = clean_certification_balanced(cert)
                    if cleaned_cert and is_plausible_certification(cleaned_cert):
                        certs.add(cleaned_cert)
            else:
                cleaned_cert = clean_certification_balanced(clean)
                if cleaned_cert and is_plausible_certification(cleaned_cert):
                    certs.add(cleaned_cert)
    
    return list(certs)

def clean_certification_balanced(cert_text: str) -> str:
    """Balanced cleaning - preserve certification content"""
    if not cert_text:
        return ""
    
    # Remove bullet points and numbering
    cert_text = re.sub(r'^[•\-*››◦▪\u2022\s]*', '', cert_text)
    cert_text = re.sub(r'^\d+[\.\)]\s*', '', cert_text)
    
    # Remove dates but be careful
    cert_text = re.sub(r'\s*\d{4}\s*[-–—]\s*\d{4}\s*', ' ', cert_text)  # Date ranges
    cert_text = re.sub(r'\s*\(\s*\d{4}\s*\)\s*', ' ', cert_text)  # (2023)
    
    # Remove only obvious garbage prefixes
    cert_text = re.sub(r'^(?:Link|Github|Badge|Blood|Donation|Platform)[\s:\-]*', '', cert_text, flags=re.IGNORECASE)
    
    # Remove proficiency levels but keep "Professional" if it's part of cert name
    cert_text = re.sub(r'\s*(?:professional proficiency|native proficiency|full professional|beginner|intermediate|advanced|b2|c1|a1|maternelle|courant|fluent)\b', '', cert_text, flags=re.IGNORECASE)
    
    # Remove specific garbage terms
    garbage_terms = ['mlflow', 'prometheus', 'grafana', 'ansible', 'blood donation', 'september', 'january', 'obtained at', 'aquired from']
    for term in garbage_terms:
        cert_text = re.sub(r'\b' + re.escape(term) + r'\b', '', cert_text, flags=re.IGNORECASE)
    
    # Clean separators and spaces
    cert_text = re.sub(r'[^\w\s\-&+#/\.]', ' ', cert_text)
    cert_text = re.sub(r'\s+', ' ', cert_text).strip()
    
    # Remove trailing dashes and hyphens
    cert_text = re.sub(r'\s*[-–—]\s*$', '', cert_text)
    
    # Smart title casing
    words = cert_text.split()
    cleaned_words = []
    
    for i, word in enumerate(words):
        if word.upper() in CERTIFICATION_ACRONYMS:
            cleaned_words.append(word.upper())
        elif re.match(r'^\d+\.\d+$', word):  # Version numbers
            cleaned_words.append(word)
        elif len(word) > 1:
            if i == 0 or word[0].isupper():
                cleaned_words.append(word)
            else:
                cleaned_words.append(word.title())
        else:
            cleaned_words.append(word)
    
    result = ' '.join(cleaned_words)
    
    # Final length validation - more permissive
    if len(result) < 6 or len(result) > 100:
        return ""
    
    return result

def has_certification_indicators(text_lower: str) -> bool:
    """Check if text has certification indicators - BROAD approach"""
    return (
        re.search(r'(?i)\b(certified|certification|certificate|certificat|cert)\b', text_lower) or
        re.search(r'(?i)\b(associate|professional|specialist|expert|master|foundation|fundamental)\b', text_lower) or
        any(provider in text_lower for provider in CERTIFICATION_PROVIDERS) or
        re.search(r'(AZ-\d+|AWS-|Google Cloud|CCNA|CCNP|CISSP|PMP)', text_lower) or
        # Include common certification-like patterns
        re.search(r'(?i)(training course|challenges|fundamentals|deep learning|docker|kubernetes|terraform)', text_lower)
    )

def contains_obvious_garbage(text_lower: str) -> bool:
    """Check if text contains obvious garbage - MINIMAL list"""
    return any(garbage in text_lower for garbage in [
        'full professional proficiency', 'native proficiency', 
        'blood donation platform', 'september', 'january',
        'obtained at', 'aquired from', 'member', 'club'
    ])

def is_plausible_certification(cert_text: str) -> bool:
    """Check if text is plausibly a certification - BROAD validation"""
    cert_lower = cert_text.lower()
    
    return (
        has_certification_indicators(cert_lower) and
        not contains_obvious_garbage(cert_lower) and
        len(cert_text.split()) >= 1 and  # Even single words can be certs
        len(cert_text.split()) <= 10 and  # More flexible
        not cert_text.isdigit() and
        cert_text[0].isupper()
    )

# Constants - EXPANDED providers list
CERTIFICATION_PROVIDERS = [
    'microsoft', 'azure', 'aws', 'amazon', 'google cloud', 'google', 'oracle', 'cisco', 
    'comptia', 'pmi', 'scrum', 'itil', 'isc2', 'sans', 'ec council', 'red hat', 'linux',
    'kubernetes', 'docker', 'terraform', 'ansible', 'salesforce', 'servicenow', 'sap',
    'ibm', 'nvidia', 'vmware', 'atlassian', 'datacamp', 'aws academy', 'kodekloud', 
    'coursera', 'istqb', 'ibm cloud', 'redhat'
]

CERTIFICATION_ACRONYMS = {
    'AWS', 'AZ', 'AI', 'ML', 'IT', 'CCNA', 'CCNP', 'CISSP', 'PMP', 'CEH', 
    'OSCP', 'CISA', 'CRISC', 'ITIL', 'ISO', 'GDPR', 'SOC', 'NIST', 'DLI',
    'AZ-900', 'AWS-', 'Google Cloud'
}
def parse_languages(raw_text: str) -> List[str]:
    """Enhanced language extraction with multiple strategies"""
    languages_found = set()
    
    # Strategy 1: Look for language sections with common patterns
    language_section_patterns = [
        r'(?i)(?:languages?|langues?)[\s:\-]*(.*?)(?=(?:\n[A-Z]|\n\s*\n|$))',
        r'(?i)(?:linguistic skills|compétences linguistiques)[\s:\-]*(.*?)(?=(?:\n[A-Z]|\n\s*\n|$))'
    ]
    
    for pattern in language_section_patterns:
        section_matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        for section in section_matches:
            # Extract languages from the section
            languages_found.update(extract_languages_from_section(section))
    
    # Strategy 2: Look for language lists with proficiency levels
    language_patterns = [
        # Pattern for: English (Professional), French (Native), etc.
        r'(?i)\b(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand|italian|italien|chinese|chinois|japanese|japonais|russian|russe|portuguese|portugais)\s*[:\-\(]?\s*(?:professional|proficiency|fluent|intermediate|beginner|native|bilingual|b2|c1|a1|a2|b1|c2|maternelle|courant|current|moyen)',
        # Pattern for: Arabic (Native), French (B2), etc.
        r'(?i)\b(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand)\s*[:\-\(]?\s*(?:native|b2|c1|a1|a2|b1|c2|maternelle|courant)',
        # Simple language mentions in context
        r'(?i)(?:\b(?:language|langue)s?[\s:\-]+)(.*?)(?=\n|$)',
    ]
    
    for pattern in language_patterns:
        matches = re.finditer(pattern, raw_text)
        for match in matches:
            lang_text = match.group(1) if match.groups() else match.group()
            normalized_lang = normalize_language_name(lang_text)
            if normalized_lang:
                languages_found.add(normalized_lang)
    
    # Strategy 3: Look for common language phrases in the text
    common_language_phrases = [
        r'(?i)\b(?:arabic|arabe)\s*(?:native|maternelle)',
        r'(?i)\b(?:french|français|francais)\s*(?:courant|fluent|professional)',
        r'(?i)\b(?:english|anglais)\s*(?:professional|fluent|courant|b2|c1)',
        r'(?i)\b(?:spanish|espagnol|german|allemand)\s*(?:intermediate|beginner|moyen)'
    ]
    
    for phrase in common_language_phrases:
        if re.search(phrase, raw_text, re.IGNORECASE):
            lang_match = re.search(r'(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand)', phrase, re.IGNORECASE)
            if lang_match:
                normalized_lang = normalize_language_name(lang_match.group(1))
                if normalized_lang:
                    languages_found.add(normalized_lang)
    
    # Strategy 4: Extract from structured language lists (bullet points, etc.)
    lines = raw_text.split('\n')
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if len(clean_line) < 50:  # Language lines are usually short
            # Look for language patterns in short lines
            lang_match = re.search(r'(?i)^\s*[•\-*]?\s*(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand)', clean_line)
            if lang_match:
                normalized_lang = normalize_language_name(lang_match.group(1))
                if normalized_lang:
                    languages_found.add(normalized_lang)
    
    return sorted(list(languages_found))

def extract_languages_from_section(section_text: str) -> List[str]:
    """Extract languages from a dedicated language section"""
    languages = set()
    
    # Split by common separators
    parts = re.split(r'[,\|\-\n•]', section_text)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Look for language-proficiency patterns
        lang_match = re.search(r'(?i)(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand)', part)
        if lang_match:
            normalized_lang = normalize_language_name(lang_match.group(1))
            if normalized_lang:
                languages.add(normalized_lang)
    
    return list(languages)

def normalize_language_name(lang_text: str) -> str:
    """Normalize language names to standard English"""
    lang_text = lang_text.strip().lower()
    
    language_map = {
        'english': 'English',
        'anglais': 'English',
        'french': 'French',
        'français': 'French',
        'francais': 'French',
        'arabic': 'Arabic',
        'arabe': 'Arabic',
        'spanish': 'Spanish',
        'espagnol': 'Spanish',
        'german': 'German',
        'allemand': 'German',
        'italian': 'Italian',
        'italien': 'Italian',
        'chinese': 'Chinese',
        'chinois': 'Chinese',
        'japanese': 'Japanese',
        'japonais': 'Japanese',
        'russian': 'Russian',
        'russe': 'Russian',
        'portuguese': 'Portuguese',
        'portugais': 'Portuguese'
    }
    
    return language_map.get(lang_text, '')

# Also update the parse_cv function to better handle language extraction
def parse_cv(text: str, filename: str = "", nlp: pipeline = None) -> Dict[str, Any]:
    """Enhanced CV parsing with NLP support across all functions"""
    sections = {
        "Name": "",
        "Contact": {"email": "", "phone": "", "location": "", "url": ""},
        "Summary": "",
        "Skills": [],
        "Education": [],
        "Projects": [],
        "Experience": [],
        "Certifications": [],
        "languages": [],
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
    
    # Extract summary
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
    
    # Parse all sections WITH NLP SUPPORT
    sections["Skills"] = parse_skills(text, nlp=nlp)
    sections["Education"] = parse_education(section_lines["Education"])
    sections["Experience"] = parse_experience(section_lines["Experience"], nlp=nlp)
    sections["Projects"] = parse_projects(section_lines["Projects"], nlp=nlp)
    sections["Certifications"] = parse_certifications(section_lines["Certifications"], nlp=nlp, raw_text=text)    
    # Enhanced language extraction - try multiple sources
    languages_from_section = parse_languages_from_section_lines(section_lines.get("Languages", []))
    languages_from_text = parse_languages(text)
    
    # Combine both sources, prioritizing section-based extraction
    if languages_from_section:
        sections["languages"] = languages_from_section
    else:
        sections["languages"] = languages_from_text
    
    # Simple interests parsing
    interests = []
    for line in section_lines["Interests"]:
        clean = clean_line(line)
        if len(clean) > 3 and len(clean) < 50 and not re.search(DATE_PATTERN, clean):
            interests.append(clean.title())
    sections["Interests"] = list(set(interests))[:5]
    
    return sections

def parse_languages_from_section_lines(section_lines: List[str]) -> List[str]:
    """Parse languages specifically from language section lines"""
    languages_found = set()
    
    for line in section_lines:
        clean_line_text = clean_line(line)
        if not clean_line_text:
            continue
            
        # Look for language patterns in section lines
        lang_match = re.search(r'(?i)(english|anglais|french|français|francais|arabic|arabe|spanish|espagnol|german|allemand)', clean_line_text)
        if lang_match:
            normalized_lang = normalize_language_name(lang_match.group(1))
            if normalized_lang:
                languages_found.add(normalized_lang)
    
    return sorted(list(languages_found))
    
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

