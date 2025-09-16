#1 parse functions
import fitz  # PyMuPDF
import re
import os
import json
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define patterns
name_pattern = r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$'
contact_pattern = r'[\+]\d{6,}|\w+@\w+\.\w+|www\.\w+\.\w+|\uf0a7||||LinkedIn|\+|@'

def extract_text(pdf_path):
    """Extract text with block-level awareness and raw output."""
    try:
        text = ""
        raw_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks", sort=True)
                for block in blocks:
                    if len(block) >= 5 and block[4].strip():
                        raw_line = block[4].strip()
                        raw_text += raw_line + "\n"
                        text += f"{raw_line} [BLKSEP_{block[1]:.2f}_{block[0]:.2f}]\n"
        return text.strip(), raw_text
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return "", ""

def clean_line(line):
    """Clean text, removing coordinates and contact info."""
    line = re.sub(contact_pattern, '', line)  # Remove contact info first
    line = re.sub(r'[\uf0b7\uf076\uf09f•●◦▪\t]', '', line)
    line = re.sub(r'\s+', ' ', line).strip()
    return line

def parse_cv(text, raw_text):
    """Parse CV with refined section logic and raw data."""
    sections = defaultdict(list, {"Name": "", "Summary": "", "Skills": [], "Education": [], "Projects": [], "Experience": [], "Certifications": [], "Languages": []})

    lines = [l.split('[BLKSEP_')[0] for l in text.split('[BLKSEP_') if l.strip()]  # Strip coordinates here
    if not lines:
        logging.warning("No text extracted from CV")
        return {"raw": raw_text, **dict(sections)}

    # Name detection with raw data cross-check
    name_found = False
    for i, line in enumerate(lines):
        line_text = clean_line(line)
        if not name_found and re.match(name_pattern, line_text):
            sections["Name"] = line_text
            name_found = True
            if i + 1 < len(lines):
                next_line = clean_line(lines[i + 1])
                if not re.match(name_pattern, next_line) and not re.search(r'\d{4}\s*-\s*\d{4}', next_line):
                    sections["Summary"] = next_line
                    lines = lines[i + 2:]
                else:
                    lines = lines[i + 1:]
            else:
                lines = lines[i + 1:]
        elif not name_found:
            for raw_line in raw_text.split('\n'):
                raw_clean = clean_line(raw_line)
                if re.match(name_pattern, raw_clean):
                    sections["Name"] = raw_clean
                    name_found = True
                    break
            if name_found:
                continue
    if not sections["Name"]:
        for line in lines:
            line_text = clean_line(line)
            if re.match(name_pattern, line_text):
                sections["Name"] = line_text
                lines.remove(line)
                break
        if not sections["Name"]:
            sections["Name"] = "Unknown"

    current_section = None
    buffer_lines = []

    headers = {
        "Summary": [r"(?i)\b(summary|profile|about\s*me|profil|je suis)\b"],
        "Skills": [r"(?i)\b(skills|technologies|compétences|technical\s*skills)\b"],
        "Education": [r"(?i)\b(education|formation|éducation|academic)\b"],
        "Projects": [r"(?i)\b(projects|projets|portfolio|pfa)\b"],
        "Experience": [r"(?i)\b(experience|work\s*history|professional\s*experience|expérience|employment|stage|intern)\b"],
        "Certifications": [r"(?i)\b(certifications|certificats|certificates|achievements)\b"],
        "Languages": [r"(?i)\b(languages|langues)\b"]
    }

    header_patterns = {key: [re.compile(pat) for pat in patterns] for key, patterns in headers.items()}

    for line in lines:
        line_text = clean_line(line)
        header_found = False

        for section, patterns in header_patterns.items():
            if any(pat.search(line_text.lower()) for pat in patterns):
                if buffer_lines and current_section:
                    process_buffer(buffer_lines, current_section, sections)
                buffer_lines = []
                current_section = section
                header_found = True
                break

        if header_found:
            continue

        if not current_section:
            if re.search(r'\b(20\d{2}\s*-\s*(20\d{2}|present|current)|university|degree|diploma)\b', line_text, re.I):
                current_section = "Education"
            elif re.search(r'\b(project|projet|pfa|github|developed|built)\b', line_text, re.I):
                current_section = "Projects"
            elif re.search(r'\b(intern|engineer|developer|manager|stage)\b', line_text, re.I):
                current_section = "Experience"
            elif re.search(r'\b(certificate|certified|badge|workshop)\b', line_text, re.I):
                current_section = "Certifications"
            elif re.search(r'(french|english|arabic|native|professional|fluent|beginner)', line_text, re.I):
                current_section = "Languages"
            elif re.search(r'\b(python|java|sql|skills|technologies)\b', line_text, re.I):
                current_section = "Skills"
            else:
                if not re.search(r'\d{4}\s*-\s*\d{4}', line_text) and not sections["Summary"]:
                    sections["Summary"] = line_text
                continue

        buffer_lines.append(line_text)

    if buffer_lines and current_section:
        process_buffer(buffer_lines, current_section, sections)

    # Reassign misclassified data using raw context
    if sections["Skills"] and any(re.search(r'(french|english|arabic|native|professional|fluent|beginner)', s) for s in sections["Skills"]):
        sections["Languages"].extend([s for s in sections["Skills"] if re.search(r'(french|english|arabic|native|professional|fluent|beginner)', s)])
        sections["Skills"] = [s for s in sections["Skills"] if not re.search(r'(french|english|arabic|native|professional|fluent|beginner)', s)]
    if sections["Certifications"] and any(re.search(r'\b(pfa|project|developed|built)\b', c.lower()) for c in sections["Certifications"]):
        projects = []
        current_project = {"title": "", "description": ""}
        for cert in sections["Certifications"]:
            if re.search(r'\b(20\d{2}\s*-\s*(20\d{2}|present|current)|pfa)\b', cert.lower()):
                if current_project["title"] or current_project["description"]:
                    projects.append({k: clean_line(v) for k, v in current_project.items()})
                current_project = {"title": cert, "description": ""}
            else:
                current_project["description"] += " " + cert
        if current_project["title"] or current_project["description"]:
            projects.append({k: clean_line(v) for k, v in current_project.items()})
        sections["Projects"].extend(projects)
        sections["Certifications"] = [c for c in sections["Certifications"] if not re.search(r'\b(pfa|project|developed|built)\b', c.lower())]

    sections = post_process_sections(sections)
    return {"raw": raw_text, **dict(sections)}

def process_buffer(buffer_lines, section, sections):
    """Process buffered lines with multi-line support."""
    if not buffer_lines:
        return
    buffer_text = ' '.join(buffer_lines)

    if section == "Summary":
        sections["Summary"] = clean_line(buffer_text)
    elif section in ["Skills", "Certifications", "Languages"]:
        items = [clean_line(item.strip()) for item in re.split(r'[,\n]', buffer_text) if item.strip()]
        sections[section].extend([item for item in items if item and item not in sections[section]])
    elif section == "Education":
        current_entry = ""
        for line in buffer_lines:
            current_entry += line + " "
            if re.search(r'(?i)(university|degree|diploma|\d{4}\s*-\s*(present|current|\d{4}))', line):
                clean_entry = clean_line(current_entry)
                if clean_entry and clean_entry not in sections[section]:
                    sections[section].append(clean_entry)
                current_entry = ""
        if current_entry.strip():
            clean_entry = clean_line(current_entry)
            if clean_entry and clean_entry not in sections[section]:
                sections[section].append(clean_entry)
    elif section in ["Projects", "Experience"]:
        entries = []
        current_entry = {"title": "", "description": ""} if section == "Projects" else {"role": "", "description": ""}
        for line in buffer_lines:
            if re.search(r'(20\d{2}\s*-\s*(20\d{2}|present|current)|:)', line):
                if current_entry["title" if section == "Projects" else "role"] or current_entry["description"]:
                    clean_entry = {k: clean_line(v) for k, v in current_entry.items()}
                    if clean_entry not in entries:
                        entries.append(clean_entry)
                if ':' in line:
                    title, desc = line.split(':', 1)
                    current_entry = {"title": title.strip(), "description": desc.strip()} if section == "Projects" else {"role": title.strip(), "description": desc.strip()}
                elif re.search(r'\b(20\d{2}\s*-\s*(20\d{2}|present|current))\b', line):
                    current_entry["title" if section == "Projects" else "role"] = line
                    current_entry["description"] = ""
            else:
                current_entry["description"] += " " + line
        if current_entry["title" if section == "Projects" else "role"] or current_entry["description"]:
            clean_entry = {k: clean_line(v) for k, v in current_entry.items()}
            if clean_entry not in entries:
                entries.append(clean_entry)
        sections[section].extend(entries)

def post_process_sections(sections):
    """Clean and validate sections."""
    for key in ["Skills", "Certifications", "Languages", "Education"]:
        sections[key] = list(dict.fromkeys([clean_line(item) for item in sections[key] if item]))
    for key in ["Projects", "Experience"]:
        sections[key] = [entry for entry in sections[key] if entry.get("title", entry.get("role", "")) or entry.get("description")]
    if not sections["Name"] or sections["Name"] == "Unknown":
        logging.warning("Name not detected or misclassified, setting to derived value")
        for line in sections["Education"] + sections["Skills"]:
            match = re.match(name_pattern, clean_line(line))
            if match:
                sections["Name"] = match.group()
                break
        if not sections["Name"]:
            sections["Name"] = "Unknown"
    sections["Summary"] = clean_line(sections["Summary"])
    return sections

def process_pdfs(folder_path, output_file="cv_structured.json"):
    """Process all PDFs and save results with raw data."""
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            logging.info(f"Processing {filename}")
            text, raw_text = extract_text(path)
            if not text:
                logging.error(f"No text extracted from {filename}")
                continue
            cv_data = parse_cv(text, raw_text)
            results.append(cv_data)
            logging.info(f"Processed {filename} → {cv_data['Name']}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")

    print("\n\nFull parsed JSON with raw data:\n")
    print(json.dumps(results, ensure_ascii=False, indent=4))
    print("\nExtraction completed!")

    return results

if __name__ == "__main__":
    process_pdfs("/content/")



#2nd approch 
import fitz  # Keep for compatibility if needed, but primary is pdfminer
from pdfminer.high_level import extract_text
import re
import os
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from unicodedata import normalize
import difflib  # For deduping

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Expanded regex patterns
NAME_PATTERN = r'^[A-Za-zÀ-ÿ\s\'-]{2,}(?:\s[A-Za-zÀ-ÿ\s\'-]{2,})+$'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\+?\d{1,4}[\s.-]?\d{2,3}[\s.-]?\d{2,3}[\s.-]?\d{2,4}\b'
LOCATION_PATTERN = r'(?i)(?:[A-Za-zÀ-ÿ\s]+,\s*[A-Za-zÀ-ÿ\s]+|Tunisia|Tunis|Ariana|Bizerte|Kairouan)'
URL_PATTERN = r'(?i)(?:https?://)?(?:www\.)?(?:linkedin|github|netlify|portfolio)\.[a-zA-Z0-9./-]+'
DATE_PATTERN = r'(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|present|current|ongoing)|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|present|current|ongoing)|\d{4})'
BULLET_PATTERN = r'^[-•* ]'  # New for bullets

# Expanded section patterns with stricter matching (e.g., short lines, caps)
SECTION_PATTERNS = {
    "Summary": [r"(?i)^(?:summary|profile|about\s*me|profil|objectif|résumé|career\s*objective|professional\s*summary|core\s*competencies)$"],
    "Skills": [r"(?i)^(?:skills|technical\s*skills|compétences|competences|expertise|langages\s*et\s*frameworks|technologies|core\s*skills|technical\s*proficiencies)$"],
    "Education": [r"(?i)^(?:education|formation|parcours\s*académique|études|academic\s*background|qualifications|academic\s*record|degree)$"],
    "Experience": [r"(?i)^(?:experience|expérience|work\s*experience|expérience\s*professionnelle|professional\s*experience|stage|internship|employment\s*history|work\s*history|intern|professional\s*experience)$"],
    "Projects": [r"(?i)^(?:projects|projets|portfolio|projets\s*personnels|notable\s*projects|projet\s*de\s*fin\s*d’étude|pfa|pidev|pi|projets\s*académiques)$"],
    "Certifications": [r"(?i)^(?:certifications|certificats|badges|certificates\s*and\s*badges|achievements|awards|professional\s*certifications)$"],
    "Languages": [r"(?i)^(?:languages|langues|compétences\s*linguistiques|language\s*skills|linguistic\s*proficiency)$"],
    "Volunteering": [r"(?i)^(?:volunteering|activités\s*extracurriculaires|clubs|vie\s*associative|extracurricular|community\s*service|associations)$"],
    "Interests": [r"(?i)^(?:interests|hobbies|intérêts|loisirs|personal\s*interests)$"]
}

# Expanded stop words + common noise
STOP_WORDS = [
    "contact", "taches realisees", "profil", "summary", "resume", "present", "current",
    "ongoing", "github", "linkedin", "badge", "keywords", "mots", "cles", "via", "tasks",
    "achievements", "conception", "developpement", "gestion", "projet", "technologies",
    "application", "platform", "system", "email", "phone", "url", "tunis", "ariana",
    "french", "english", "arabic", "professional", "native", "fluent", "b2", "beginner",
    "pfa", "pidev", "pi", "stage", "intern", "member", "club", "experience", "education",
    "skills", "competences", "projects", "projets", "certified", "fundamentals", "award",
    "taches", "realisees", "keywords:", "mots-cles:", "technologies:", "link", "workshop",
    "|"  # For special designs with bars
]

# Expanded technical terms for skills
TECHNICAL_TERMS = [
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "sql", "r", "matlab",
    "html", "css", "react", "angular", "vue", "node.js", "express.js", "django", "flask",
    "spring boot", "fastapi", "mongodb", "mysql", "postgresql", "oracle", "sqlite",
    "tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas", "matplotlib",
    "seaborn", "docker", "kubernetes", "jenkins", "ansible", "terraform", "aws", "azure",
    "gcp", "openstack", "git", "github", "gitlab", "bitbucket", "sonarqube", "phpstan",
    "phpunit", "karma", "jest", "kivy", "kivymd", "java3d", "three.js", "firebase",
    "prometheus", "grafana", "mlflow", "next.js", "bootstrap", "materialui", "figma",
    "linux", "ubuntu", "windows", "bash", "spark", "airflow", "snowflake", "dbt",
    "power bi", "superset", "cassandra", "kafka", "tensorrt", "llama", "minilm",
    "symfony", "asp.net", "mern", "web scraping", "computer vision", "nlp", "generative ai",
    "deep learning", "machine learning", "regression", "decision trees", "random forest",
    "svm", "gradient boosting", "time series", "image processing", "iac", "ci/cd",
    "microservices", "rest apis", "uml", "mikroc", "proteus", "cisco packet tracer",
    "nmap", "openvas", "nessus", "gobuster", "metasploit", "burp suite", "hydra",
    "sqlmap", "john the ripper", "wireshark", "windump", "snort", "autopsy", "ftk imager",
    "volatility", "dumpit", "azure devops", "devsecops", "penetration testing",
    "ethical hacking", "cloud security", "splunk", "javafx", "codename one", "php",
    "ionic", "pug.js", "flink", "superset", "power bi", "t5-small", "faiss", "helsinki mt",
    "bert", "tf-idf", "sentence-bert", "yolo", "maven", "junit", "mockito", "kvm", "iaas",
    "solid", "material", "adsl", "sdh", "pic16f877", "gtk", "net"
]

def clean_line(line: str) -> str:
    if not line or not isinstance(line, str):
        return ""
    line = normalize('NFKD', line).encode('ASCII', 'ignore').decode('ASCII')
    line = re.sub(r'[\uf0b7\uf076\uf09f•●◦▪\t●•▪○∙\u00a0\U0001F000-\U0001FFFF]', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'[()[\]{}|]', '', line).strip()  # Remove bars for special designs
    return line

def extract_contact_info(raw_text: str) -> Dict[str, str]:
    emails = list(set(re.findall(EMAIL_PATTERN, raw_text)))
    phones = list(set(re.findall(PHONE_PATTERN, raw_text)))
    locations = list(set(re.findall(LOCATION_PATTERN, raw_text)))
    urls = list(set(re.findall(URL_PATTERN, raw_text)))
    
    valid_location = next((loc.strip() for loc in locations if loc.strip() and len(loc) > 4 and len(loc.split(',')) >= 1 and not any(kw in loc.lower() for kw in STOP_WORDS + TECHNICAL_TERMS)), "")
    
    valid_phone = next((phone for phone in phones if len(phone.replace(" ", "").replace("-", "").replace(".", "")) >= 8 and not re.match(r'^\d{4}$', phone)), "")
    
    return {
        "email": emails[0] if emails else "",
        "phone": valid_phone,
        "location": valid_location,
        "url": urls[0] if urls else ""
    }

def extract_name(lines: List[str], raw_text: str, contact_info: Dict[str, str]) -> str:
    email = contact_info.get("email", "")
    url = contact_info.get("url", "")
    phone = contact_info.get("phone", "")
    email_hint = email.split("@")[0].replace(".", " ").replace("-", " ").lower() if email else ""
    url_hint = url.split("/")[-1].replace("-", " ").replace(".", " ").lower() if url else ""
    phone_hint = phone.replace("+216", "").strip() if phone else ""

    # Priority: First 10 lines, match pattern + hint, skip if technical
    for line in lines[:10]:
        clean = clean_line(line)
        if re.match(NAME_PATTERN, clean) and len(clean.split()) >= 2 and not any(kw in clean.lower() for kw in STOP_WORDS + ["tunisia", "tunis", "esprit", "ensit"]):
            if any(hint in clean.lower() for hint in [email_hint, url_hint, phone_hint] if hint):
                return clean.title()

    # Fallback: Near contact info in raw
    raw_lines = raw_text.split('\n')
    contact_indices = [i for i, l in enumerate(raw_lines) if email in l or url in l or phone in l]
    if contact_indices:
        start = max(0, min(contact_indices) - 5)
        for line in raw_lines[start : min(contact_indices) + 1]:
            clean = clean_line(line)
            if re.match(NAME_PATTERN, clean) and len(clean.split()) >= 2 and not any(t in clean.lower() for t in TECHNICAL_TERMS):
                return clean.title()

    # Ultimate fallback: Capitalize email_hint properly (split on numbers)
    if email_hint:
        hint_parts = re.split(r'\d+', email_hint)
        return ' '.join(word.capitalize() for word in hint_parts[0].split() if word)
    return (email_hint or url_hint).title() or ""

def detect_section_header(line: str, current_section: str) -> Tuple[str, bool]:
    clean = clean_line(line).lower().strip()
    if not clean or len(clean) > 30:  # Avoid long lines as headers
        return current_section, False
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.match(pat, clean):
                logging.debug(f"Detected section: {section} from line: {line}")
                return section, True
    return current_section, False

def parse_education(lines: List[str]) -> List[Dict[str, str]]:
    entries = []
    current = {"degree": "", "institution": "", "duration": ""}
    collecting = False
    for line in lines:
        clean = clean_line(line)
        if not clean:
            continue
        date_match = re.search(DATE_PATTERN, clean)
        degree_keywords = ["diplome", "degree", "bachelor", "master", "licence", "ingenieur", "bac", "cycle", "preparatoire"]
        if any(kw in clean.lower() for kw in degree_keywords):
            if collecting and any(current.values()):
                entries.append(current)
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            current = {"degree": remaining, "institution": "", "duration": date_match.group(0) if date_match else ""}
            collecting = True
            continue
        if date_match and collecting:
            current["duration"] = date_match.group(0)
        elif collecting and not current["institution"] and not any(kw in clean.lower() for kw in STOP_WORDS):
            current["institution"] = clean
        elif collecting and current["institution"] and len(clean.split()) < 10 and not date_match:  # Limit aggregation
            current["institution"] += " " + clean
    if any(current.values()):
        entries.append(current)
    if len(entries) > 5:
        entries = entries[:5]  # Limit max
    return [{"degree": e["degree"].strip().title(), "institution": e["institution"].strip().title(), "duration": e["duration"].strip()} for e in entries if e["degree"] or e["institution"]]

def parse_experience(lines: List[str]) -> List[Dict[str, str]]:
    experiences = []
    current = {"role": "", "company": "", "duration": "", "description": "", "type": "work", "location": ""}
    collecting = False
    role_keywords = ["intern", "stage", "stag", "engineer", "developer", "analyst", "manager", "internship", "pfe"]
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        date_match = re.search(DATE_PATTERN, clean)
        if date_match or any(kw in clean.lower() for kw in role_keywords) or re.match(BULLET_PATTERN, clean):
            if collecting and any(current.values()):
                experiences.append(current)
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            current = {"role": remaining if not re.match(BULLET_PATTERN, remaining) else clean, "company": "", "duration": date_match.group(0) if date_match else "", "description": "", "type": "work", "location": ""}
            if any(kw in clean.lower() for kw in ["intern", "stage", "stag", "internship", "pfe"]):
                current["type"] = "intern"
            collecting = True
            # Aggregate next lines
            for j in range(i + 1, min(i + 20, len(lines))):
                next_clean = clean_line(lines[j])
                if not next_clean or detect_section_header(next_clean, "")[1] or re.search(DATE_PATTERN, next_clean) or any(trig in next_clean.lower() for trig in role_keywords):
                    break
                if re.match(BULLET_PATTERN, next_clean):
                    current["description"] += "\n" + next_clean
                elif re.match(LOCATION_PATTERN, next_clean):
                    current["location"] = next_clean
                elif not current["company"] and len(next_clean.split()) > 1 and not any(kw in next_clean.lower() for kw in STOP_WORDS):
                    current["company"] = next_clean
                else:
                    current["description"] += " " + next_clean
    if any(current.values()):
        experiences.append(current)
    # Dedupe stricter
    unique = []
    for e in experiences:
        if not any(difflib.SequenceMatcher(None, e['role'].lower(), u['role'].lower()).ratio() > 0.6 for u in unique):
            unique.append(e)
    return [{"role": e["role"].strip().title(), "company": e["company"].strip().title(), 
             "duration": e["duration"].strip(), "description": e["description"].strip(), 
             "type": e["type"], "location": e["location"].strip().title()} 
            for e in unique if e["role"] or e["company"]]

def parse_skills(lines: List[str], project_keywords: List[str], summary: str, raw_text: str) -> List[str]:
    skills = set()
    all_text = ' '.join(lines + project_keywords + [summary, raw_text]).lower()
    for term in TECHNICAL_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', all_text, re.IGNORECASE):
            skills.add(term.title())
    return sorted(list(skills))

def parse_projects(lines: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    projects = []
    keywords = []
    current = {"name": "", "description": "", "type": "personal"}
    collecting = False
    project_triggers = ["pfa", "pidev", "pi", "project", "projet", "application", "platform", "system"]
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        keyword_match = re.search(r'(?i)(?:mots-clés|keywords|technologies)\s*[:]', clean)
        if keyword_match:
            kw_str = re.sub(r'(?i)(?:mots-clés|keywords|technologies)\s*[:]', '', clean).strip()
            if kw_str:
                keywords.append(kw_str)
            continue
        date_match = re.search(DATE_PATTERN, clean)
        is_bullet = re.match(BULLET_PATTERN, clean)
        if date_match or any(trig in clean.lower() for trig in project_triggers) or is_bullet:
            if collecting and any(current.values()) and current["name"]:
                projects.append(current)
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            if is_bullet:
                remaining = re.sub(BULLET_PATTERN, '', remaining).strip()
            current = {"name": remaining, "description": "", "type": "personal"}
            if any(kw in clean.lower() for kw in ["pfa", "pidev", "academic", "academique", "etude"]):
                current["type"] = "academic"
            collecting = True
            # Aggregate next lines
            for j in range(i + 1, min(i + 20, len(lines))):
                next_clean = clean_line(lines[j])
                if not next_clean or detect_section_header(next_clean, "")[1] or re.search(DATE_PATTERN, next_clean) or any(trig in next_clean.lower() for trig in project_triggers):
                    break
                if keyword_match:
                    kw_str = re.sub(r'(?i)(?:mots-clés|keywords|technologies)\s*[:]', '', next_clean).strip()
                    if kw_str:
                        keywords.append(kw_str)
                    continue
                current["description"] += " " + next_clean
    if any(current.values()) and current["name"]:
        projects.append(current)
    # Dedupe stricter
    unique_projects = []
    for p in projects:
        if p["name"] and not any(difflib.SequenceMatcher(None, p['name'].lower(), u['name'].lower()).ratio() > 0.6 for u in unique_projects):
            unique_projects.append(p)
    if len(unique_projects) > 10:
        unique_projects = unique_projects[:10]  # Limit max
    return (unique_projects, keywords)

def parse_generic_list(lines: List[str], section_name: str) -> List[str]:
    items = []
    for line in lines:
        clean = clean_line(line)
        if clean and not any(pat.match(clean) for pat in [re.compile(EMAIL_PATTERN), re.compile(PHONE_PATTERN), re.compile(URL_PATTERN), re.compile(DATE_PATTERN)]) and not any(kw in clean.lower() for kw in STOP_WORDS):
            items.append(clean.title())
    return sorted(set(items))

def extract_text_with_layout(path: str) -> Tuple[str, str]:
    try:
        text = extract_text(path)
        return text, text
    except Exception as e:
        logging.error(f"Error extracting text from {path}: {e}")
        return "", ""

def parse_cv(text: str, raw_text: str, filename: str = "") -> Dict[str, Any]:
    sections = {
        "Name": "", 
        "Contact": {"email": "", "phone": "", "location": "", "url": ""},
        "Summary": "", 
        "Skills": [], 
        "Education": [], 
        "Projects": [], 
        "Experience": [], 
        "Certifications": [], 
        "Languages": [], 
        "Volunteering": [], 
        "Interests": []
    }
    
    if not text or not raw_text:
        return {"raw": raw_text, **sections}

    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    raw_text = normalize('NFKD', raw_text).encode('ASCII', 'ignore').decode('ASCII')
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sections["Contact"] = extract_contact_info(raw_text)
    sections["Name"] = extract_name(lines, raw_text, sections["Contact"])

    current_section = None
    section_lines = defaultdict(list)
    summary_lines = []
    for line in lines:
        clean = clean_line(line)
        if not clean:
            continue
        new_section, is_header = detect_section_header(clean, current_section)
        if is_header:
            current_section = new_section
            continue  # Skip header line itself
        if current_section:
            section_lines[current_section].append(clean)
        else:
            if len(clean) > 20 and not any(kw in clean.lower() for kw in STOP_WORDS + [sections["Name"].lower()]) and not re.match(EMAIL_PATTERN, clean) and not re.match(PHONE_PATTERN, clean) and not re.match(URL_PATTERN, clean):  # Longer for summary
                summary_lines.append(clean)

    # Parse focused sections with fallbacks from entire lines if section empty
    edu_lines = section_lines["Education"] or [l for l in lines if any(kw in l.lower() for kw in ["diplome", "degree", "bac", "licence", "ingenieur", "cycle", "preparatoire", "esprit", "ensit", "university"])]
    sections["Education"] = parse_education(edu_lines)
    
    exp_lines = section_lines["Experience"] or [l for l in lines if any(kw in l.lower() for kw in ["stage", "intern", "pfe", "emploi", "job", "professional", "experience", "internship"])]
    sections["Experience"] = parse_experience(exp_lines)
    
    proj_lines = section_lines["Projects"] or [l for l in lines if any(kw in l.lower() for kw in ["pfa", "pidev", "pi", "project", "projet", "application", "platform"])]
    projects, project_keywords = parse_projects(proj_lines)
    sections["Projects"] = projects
    
    skill_lines = section_lines["Skills"] or []
    sections["Skills"] = parse_skills(skill_lines, project_keywords, '\n'.join(summary_lines), raw_text)  # Join with \n for multi-line summary
    
    sections["Summary"] = '\n'.join(summary_lines).strip()
    sections["Certifications"] = parse_generic_list(section_lines["Certifications"] or [l for l in lines if any(kw in l.lower() for kw in ["certificat", "badge", "award", "certified", "workshop"])], "Certifications")
    sections["Languages"] = [lang.strip().title() for lang in re.split(r'[,|]', ' '.join(section_lines["Languages"]).replace('(', '').replace(')', '')) if lang.strip() and not any(kw in lang.lower() for kw in ["native", "b2", "fluent", "professional", "beginner", "proficiency"])]
    sections["Volunteering"] = parse_generic_list(section_lines["Volunteering"] or [l for l in lines if any(kw in l.lower() for kw in ["club", "association", "volunteering", "vie associative", "extracurricular", "member"])], "Volunteering")
    sections["Interests"] = parse_generic_list(section_lines["Interests"], "Interests")

    return {"raw": raw_text, **sections}

def process_pdfs(folder_path: str, output_file: str = "cv_structured_unstructured3.json") -> List[Dict[str, Any]]:
    results = []
    processed_keys = set()
    if not os.path.exists(folder_path):
        logging.error(f"Folder path {folder_path} does not exist")
        return results

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(f"No PDF files found in {folder_path}")
        return results

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        logging.info(f"Processing {filename}")
        text, raw_text = extract_text_with_layout(path)
        if not text or not raw_text:
            logging.error(f"No text extracted from {filename}")
            continue
        cv_data = parse_cv(text, raw_text, filename)
        email = cv_data["Contact"]["email"]
        name = cv_data["Name"]
        key = (email, name)
        if key in processed_keys:
            logging.warning(f"Skipping duplicate CV for email: {email}, name: {name}")
            continue
        processed_keys.add(key)
        results.append(cv_data)
        logging.info(f"Processed {filename} → Name: {cv_data['Name']}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")

    return results

if __name__ == "__main__":
    folder_path = "/content/"
    results = process_pdfs(folder_path)
    print("\nExtraction completed!")    


 #3rd approch
 # import fitz
from pdfminer.high_level import extract_text
import PyPDF2
import re
import os
import json
import logging
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from unicodedata import normalize
import difflib

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Regex patterns
NAME_PATTERN = r'^[A-Za-zÀ-ÿ\s\'-]{2,}(?:\s[A-Za-zÀ-ÿ\s\'-]{2,})+$'
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_PATTERN = r'\+?\d{1,4}[\s.-]?\d{2,3}[\s.-]?\d{2,3}[\s.-]?\d{2,4}\b'
LOCATION_PATTERN = r'(?i)(?:[A-Za-zÀ-ÿ\s]+,\s*[A-Za-zÀ-ÿ\s]+|Tunisia|Tunis|Ariana|Bizerte|Kairouan)'
URL_PATTERN = r'(?i)(?:https?://)?(?:www\.)?(?:linkedin|github|netlify|portfolio)\.[a-zA-Z0-9./-]+'
DATE_PATTERN = r'(?i)(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\d{4}\s*[-–—]\s*(?:\d{4}|present|current|ongoing)|\d{2}/\d{4}\s*[-–—]\s*(?:\d{2}/\d{4}|present|current|ongoing)|\d{4})'
BULLET_PATTERN = r'^[-•*››◦▪\u2022\u25CF\s]{1,3}'

# Section patterns - flexible
SECTION_PATTERNS = {
    "Summary": [r"(?i)(?:summary|profile|about\s*me|profil|objectif|résumé|career\s*objective|professional\s*summary|core\s*competencies|overview)"],
    "Skills": [r"(?i)(?:skills|technical\s*skills|compétences|competences|expertise|langages\s*et\s*frameworks|technologies|core\s*skills|technical\s*proficiencies|abilities|key skills)"],
    "Education": [r"(?i)(?:education|formation|parcours\s*académique|études|academic\s*background|qualifications|academic\s*record|degree|degrees|schooling)"],
    "Experience": [r"(?i)(?:experience|expérience|work\s*experience|expérience\s*professionnelle|professional\s*experience|stage|internship|employment\s*history|work\s*history|intern|professional\s*experience|professional background|career history|achievements|tasks|taches\s*realisees|responsibilities)"],
    "Projects": [r"(?i)(?:projects|projets|portfolio|projets\s*personnels|notable\s*projects|projet\s*de\s*fin\s*d’étude|pfa|pidev|pi|projets\s*académiques|personal projects|academic projects)"],
    "Certifications": [r"(?i)(?:certifications|certificats|badges|certificates\s*and\s*badges|achievements|awards|professional\s*certifications|certificates|badges and certifications)"],
    "Languages": [r"(?i)(?:languages|langues|compétences\s*linguistiques|language\s*skills|linguistic\s*proficiency)"],
    "Volunteering": [r"(?i)(?:volunteering|activités\s*extracurriculaires|clubs|vie\s*associative|extracurricular|community\s*service|associations|volunteer experience)"],
    "Interests": [r"(?i)(?:interests|hobbies|intérêts|loisirs|personal\s*interests)"]
}

STOP_WORDS = [
    "contact", "taches realisees", "profil", "summary", "resume", "present", "current",
    "ongoing", "github", "linkedin", "badge", "keywords", "mots", "cles", "via", "tasks",
    "achievements", "conception", "developpement", "gestion", "projet", "technologies",
    "application", "platform", "system", "email", "phone", "url", "tunis", "ariana",
    "french", "english", "arabic", "professional", "native", "fluent", "b2", "beginner",
    "pfa", "pidev", "pi", "stage", "intern", "member", "club", "experience", "education",
    "skills", "competences", "projects", "projets", "certified", "fundamentals", "award",
    "taches", "realisees", "keywords:", "mots-cles:", "technologies:", "link", "workshop"
]

TECHNICAL_TERMS = set([
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "sql", "r", "matlab",
    "html", "css", "react", "angular", "vue", "node.js", "express.js", "django", "flask",
    "spring boot", "fastapi", "mongodb", "mysql", "postgresql", "oracle", "sqlite",
    "tensorflow", "pytorch", "keras", "scikit-learn", "numpy", "pandas", "matplotlib",
    "seaborn", "docker", "kubernetes", "jenkins", "ansible", "terraform", "aws", "azure",
    "gcp", "openstack", "git", "github", "gitlab", "bitbucket", "sonarqube", "phpstan",
    "phpunit", "karma", "jest", "kivy", "kivymd", "java3d", "three.js", "firebase",
    "prometheus", "grafana", "mlflow", "next.js", "bootstrap", "materialui", "figma",
    "linux", "ubuntu", "windows", "bash", "spark", "airflow", "snowflake", "dbt",
    "power bi", "superset", "cassandra", "kafka", "tensorrt", "llama", "minilm",
    "symfony", "asp.net", "mern", "web scraping", "computer vision", "nlp", "generative ai",
    "deep learning", "machine learning", "regression", "decision trees", "random forest",
    "svm", "gradient boosting", "time series", "image processing", "iac", "ci/cd",
    "microservices", "rest apis", "uml", "mikroc", "proteus", "cisco packet tracer",
    "nmap", "openvas", "nessus", "gobuster", "metasploit", "burp suite", "hydra",
    "sqlmap", "john the ripper", "wireshark", "windump", "snort", "autopsy", "ftk imager",
    "volatility", "dumpit", "azure devops", "devsecops", "penetration testing",
    "ethical hacking", "cloud security", "splunk", "javafx", "codename one", "php",
    "ionic", "pug.js", "flink", "t5-small", "faiss", "helsinki mt", "bert", "tf-idf",
    "sentence-bert", "yolo", "maven", "junit", "mockito", "kvm", "iaas", "solid", "material",
    "adsl", "sdh", "pic16f877", "gtk", "net"
])

def dedup_text(text: str) -> str:
    if not text:
        return ""
    sentences = re.split(r'(?<=[\.!\?])\s+', text)
    unique = []
    seen = set()
    for s in sentences:
        s_lower = s.lower()
        if s_lower not in seen:
            seen.add(s_lower)
            unique.append(s)
    return ' '.join(unique)

def clean_line(line: str) -> str:
    if not line or not isinstance(line, str):
        return ""
    line = normalize('NFKD', line).encode('ASCII', 'ignore').decode('ASCII')
    line = re.sub(r'[\uf0b7\uf076\uf09f•●◦▪\t●•▪○∙\u00a0\U0001F000-\U0001FFFF]', ' ', line)
    line = re.sub(r'\s+', ' ', line).strip()
    line = re.sub(r'[()[\]{}|]', '', line).strip()
    if re.match(BULLET_PATTERN, line):
        line = re.sub(BULLET_PATTERN, '- ', line)
    words = line.split()
    deduped = []
    for word in words:
        if not deduped or word != deduped[-1]:
            deduped.append(word)
    return ' '.join(deduped)

def extract_text_pypdf2(path: str) -> str:
    try:
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        logging.error(f"PyPDF2 extraction failed for {path}: {e}")
        return ""

def extract_text_with_layout(path: str) -> Tuple[str, str]:
    try:
        pdfminer_text = extract_text(path)
        pypdf2_text = extract_text_pypdf2(path)
        return pdfminer_text, pypdf2_text
    except Exception as e:
        logging.error(f"Error extracting text from {path}: {e}")
        return "", ""

def extract_contact_info(raw_text: str) -> Dict[str, str]:
    emails = list(set(re.findall(EMAIL_PATTERN, raw_text)))
    phones = list(set(re.findall(PHONE_PATTERN, raw_text)))
    locations = list(set(re.findall(LOCATION_PATTERN, raw_text)))
    urls = list(set(re.findall(URL_PATTERN, raw_text)))
    
    valid_location = next((loc.strip() for loc in locations if loc.strip() and len(loc) > 4 and not any(kw in loc.lower() for kw in STOP_WORDS + list(TECHNICAL_TERMS))), "")
    valid_phone = next((phone for phone in phones if len(phone.replace(" ", "").replace("-", "").replace(".", "")) >= 8), "")
    
    return {
        "email": emails[0] if emails else "",
        "phone": valid_phone,
        "location": valid_location,
        "url": urls[0] if urls else ""
    }

def extract_name(lines: List[str], raw_text: str, contact_info: Dict[str, str]) -> str:
    email = contact_info.get("email", "")
    url = contact_info.get("url", "")
    phone = contact_info.get("phone", "")
    email_hint = email.split("@")[0].replace(".", " ").replace("-", " ").lower() if email else ""
    url_hint = url.split("/")[-1].replace("-", " ").replace(".", " ").lower() if url else ""
    phone_hint = phone.replace("+216", "").strip() if phone else ""

    candidates = []
    for i, line in enumerate(lines[:15]):
        clean = clean_line(line)
        tokens = clean.split()
        if re.match(NAME_PATTERN, clean) and len(tokens) >= 2 and not any(kw in clean.lower() for kw in STOP_WORDS + ["tunisia", "tunis", "esprit", "ensit"]):
            score = sum(hint in ' '.join(tokens).lower() for hint in [email_hint, url_hint, phone_hint] if hint)
            score += 2 if i < 3 else 0
            score += 1 if clean.isupper() else 0
            candidates.append((clean, score))
    
    if candidates:
        best_name = max(candidates, key=lambda x: x[1])[0]
        return ' '.join(best_name.split()).title()

    raw_lines = raw_text.split('\n')
    contact_indices = [i for i, l in enumerate(raw_lines) if email in l or url in l or phone in l]
    if contact_indices:
        start = max(0, min(contact_indices) - 5)
        for line in raw_lines[start:min(contact_indices) + 1]:
            clean = clean_line(line)
            tokens = clean.split()
            if re.match(NAME_PATTERN, clean) and len(tokens) >= 2 and not any(t in clean.lower() for t in TECHNICAL_TERMS):
                return ' '.join(tokens).title()

    if email_hint:
        hint_parts = re.split(r'\d+', email_hint)
        return ' '.join(word.capitalize() for word in hint_parts[0].split() if word)
    return (email_hint or url_hint).title() or ""

def detect_section_header(line: str, current_section: str) -> Tuple[str, bool]:
    clean = clean_line(line).lower().strip()
    if not clean or len(clean) > 50:
        return current_section, False
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if len(clean.split()) < 6 and re.search(pat, clean):
                logging.debug(f"Detected section: {section} from line: {line}")
                return section, True
    if clean.isupper() or len(clean.split()) < 4:
        for section in SECTION_PATTERNS:
            if section.lower() in clean:
                return section, True
    return current_section, False

def parse_education(lines: List[str]) -> List[Dict[str, str]]:
    entries = []
    current = {"degree": "", "institution": "", "duration": ""}
    collecting = False
    degree_keywords = ["diplome", "degree", "bachelor", "master", "licence", "ingenieur", "bac", "cycle", "preparatoire", "expected", "diploma", "genie", "d'ingenieur", "d’ingénieur", "baccalaureat"]
    institution_keywords = ["university", "school", "institute", "faculte", "ecole", "lycee", "esprit", "ensit", "ipei", "fst", "nationale", "superieure", "pilote"]
    for line in lines:
        clean = clean_line(line)
        if not clean:
            continue
        date_match = re.search(DATE_PATTERN, clean)
        remaining = re.sub(DATE_PATTERN, '', clean).strip()
        if any(kw in remaining.lower() for kw in degree_keywords) and not any(tech in remaining.lower() for tech in TECHNICAL_TERMS):
            if collecting and any(current.values()):
                entries.append(current)
            parts = remaining.split(':') if ':' in remaining else remaining.split(',')
            degree = parts[0].strip()
            institution = ' '.join(parts[1:]).strip() if len(parts) > 1 else ""
            current = {"degree": degree, "institution": institution, "duration": date_match.group(0) if date_match else ""}
            collecting = True
            continue
        if any(kw in remaining.lower() for kw in institution_keywords) and date_match:
            if collecting and any(current.values()):
                entries.append(current)
            current = {"degree": "", "institution": remaining, "duration": date_match.group(0)}
            collecting = True
            continue
        if date_match and collecting:
            current["duration"] = date_match.group(0)
        elif collecting and not current["institution"] and any(kw in clean.lower() for kw in institution_keywords):
            current["institution"] = clean
        elif collecting and current["institution"] and len(clean.split()) < 10 and not date_match and not any(tech in clean.lower() for tech in TECHNICAL_TERMS):
            current["institution"] += " " + clean
        elif collecting and not current["degree"] and len(clean.split()) < 10:
            current["degree"] = clean
    if any(current.values()):
        entries.append(current)
    unique = []
    for e in entries:
        if not any(difflib.SequenceMatcher(None, e['degree'].lower(), u['degree'].lower()).ratio() > 0.8 for u in unique):
            unique.append(e)
    return [{"degree": e["degree"].strip().title(), "institution": e["institution"].strip().title(), "duration": e["duration"].strip()} for e in unique if e["degree"] or e["institution"]][:5]

def parse_experience(lines: List[str]) -> List[Dict[str, str]]:
    experiences = []
    current = {"role": "", "company": "", "duration": "", "description": "", "type": "work", "location": ""}
    collecting = False
    role_keywords = ["intern", "stage", "stag", "engineer", "developer", "analyst", "manager", "internship", "pfe", "interning"]
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        date_match = re.search(DATE_PATTERN, clean)
        is_bullet = re.match(BULLET_PATTERN, clean)
        if (clean.isupper() or date_match or any(kw in clean.lower() for kw in role_keywords) or is_bullet) and not re.match(r'(?i)(?:keywords|technologies|outils)', clean):
            if collecting and any(current.values()):
                current["description"] = dedup_text(current["description"])
                experiences.append(current)
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            if is_bullet:
                remaining = re.sub(BULLET_PATTERN, '', remaining).strip()
            parts = remaining.split(',', 1) if ',' in remaining else remaining.split('-', 1) if '-' in remaining else [remaining, '']
            role = parts[0].strip()
            company = parts[1].strip() if len(parts) > 1 else ""
            current = {"role": role, "company": company, "duration": date_match.group(0) if date_match else "", "description": "", "type": "work", "location": ""}
            if any(kw in clean.lower() for kw in ["intern", "stage", "stag", "internship", "pfe"]):
                current["type"] = "intern"
            collecting = True
        if collecting:
            desc_lines = []
            for j in range(i + 1, min(i + 50, len(lines))):
                next_clean = clean_line(lines[j])
                if not next_clean or detect_section_header(next_clean, "")[1] or re.search(DATE_PATTERN, next_clean) or any(trig in next_clean.lower() for trig in role_keywords):
                    break
                if re.match(LOCATION_PATTERN, next_clean) and not current["location"]:
                    current["location"] = next_clean
                elif len(next_clean) > 5 and not re.match(r'(?i)(?:mots-clés|keywords|technologies|outils)\s*[:]', next_clean):
                    desc_lines.append(next_clean)
            current["description"] += ' '.join(desc_lines).strip()
    if any(current.values()):
        current["description"] = dedup_text(current["description"])
        experiences.append(current)
    unique = []
    for e in experiences:
        if not any(difflib.SequenceMatcher(None, e['role'].lower(), u['role'].lower()).ratio() > 0.8 for u in unique):
            unique.append(e)
    return [{"role": e["role"].title(), "company": e["company"].title(), 
             "duration": e["duration"].strip(), "description": e["description"].strip(), 
             "type": e["type"], "location": e["location"].strip().title()} 
            for e in unique if e["role"] or e["company"]][:5]

def parse_skills(lines: List[str], project_keywords: List[str], summary: str, raw_text: str) -> List[str]:
    skills = set()
    all_text = ' '.join(lines + project_keywords + summary.split('.') + [raw_text]).lower()
    for term in TECHNICAL_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', all_text):
            skills.add(term.title())
    for line in lines:
        if ':' in line or ',' in line:
            terms = line.split(':')[-1] if ':' in line else line
            term_list = re.split(r'[,\s/|;]+', terms)
            skills.update([t.strip().title() for t in term_list if t.strip() and t.lower() in TECHNICAL_TERMS])
    return sorted(list(skills))

def parse_projects(lines: List[str]) -> Tuple[List[Dict[str, str]], List[str]]:
    projects = []
    keywords = []
    current = {"name": "", "description": "", "type": "personal", "keywords": ""}
    collecting = False
    project_triggers = ["pfa", "pidev", "pi", "project", "projet", "application", "platform", "system", "platforme", "simulation", "detection", "prediction", "developpement", "mise en place"]
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        keyword_match = re.search(r'(?i)(?:mots-clés|keywords|technologies|outils)\s*[:]', clean)
        if keyword_match:
            kw_str = re.sub(r'(?i)(?:mots-clés|keywords|technologies|outils)\s*[:]', '', clean).strip()
            if kw_str:
                keywords.append(kw_str)
            if collecting:
                current["keywords"] += kw_str + ' '
            continue
        date_match = re.search(DATE_PATTERN, clean)
        is_bullet = re.match(BULLET_PATTERN, clean)
        if date_match or any(trig in clean.lower() for trig in project_triggers) or is_bullet:
            if collecting and current["name"]:
                current["description"] = dedup_text(current["description"])
                projects.append(current)
            remaining = re.sub(DATE_PATTERN, '', clean).strip()
            if is_bullet:
                remaining = re.sub(BULLET_PATTERN, '', remaining).strip()
            parts = remaining.split(':', 1) if ':' in remaining else remaining.split('-', 1) if '-' in remaining else [remaining, '']
            current = {"name": parts[0].strip(), "description": parts[1].strip() if len(parts) > 1 else "", "type": "personal", "keywords": ""}
            if any(kw in clean.lower() for kw in ["pfa", "pidev", "academic", "academique", "etude"]):
                current["type"] = "academic"
            collecting = True
        if collecting:
            desc_lines = []
            for j in range(i + 1, min(i + 50, len(lines))):
                next_clean = clean_line(lines[j])
                if not next_clean or detect_section_header(next_clean, "")[1] or any(trig in next_clean.lower() for trig in project_triggers):
                    break
                if keyword_match:
                    continue
                if len(next_clean) > 5:
                    desc_lines.append(next_clean)
            current["description"] += ' ' + ' '.join(desc_lines).strip()
    if current["name"]:
        current["description"] = dedup_text(current["description"])
        projects.append(current)
    unique_projects = []
    for p in projects:
        if p["name"] and not any(difflib.SequenceMatcher(None, p['name'].lower(), u['name'].lower()).ratio() > 0.8 for u in unique_projects):
            unique_projects.append(p)
    return ([{"name": p["name"].title(), "description": p["description"].strip(), "type": p["type"], "keywords": p["keywords"].strip()} for p in unique_projects][:10], keywords)

def parse_certifications(lines: List[str]) -> List[str]:
    items = []
    current_item = ""
    cert_keywords = ["certified", "certificate", "badge", "workshop", "fundamentals", "associate", "professional", "cisco", "microsoft", "nvidia", "aws", "azure", "kodekloud", "ccna", "cyberops", "devnet", "deep learning"]
    for line in lines:
        clean = clean_line(line)
        if clean and len(clean) > 5 and not any(pat.match(clean) for pat in [re.compile(EMAIL_PATTERN), re.compile(PHONE_PATTERN), re.compile(URL_PATTERN), re.compile(DATE_PATTERN)]) and any(kw in clean.lower() for kw in cert_keywords):
            if current_item and re.match(r'\d{4}', clean):
                current_item += ' ' + clean
                items.append(current_item.title())
                current_item = ""
            else:
                current_item = clean if not current_item else current_item + ' ' + clean
    if current_item:
        items.append(current_item.title())
    return sorted(set(items[:10]))

def parse_generic_list(lines: List[str], section_name: str) -> List[str]:
    items = []
    current_item = ""
    cert_keywords = ["certified", "certificate", "badge", "workshop", "fundamentals", "associate", "professional"]
    vol_kw = ["member", "club", "association", "vice", "president", "formateur", "hr", "ieee"]
    for line in lines:
        clean = clean_line(line)
        if clean and len(clean) > 5 and not any(pat.match(clean) for pat in [re.compile(EMAIL_PATTERN), re.compile(PHONE_PATTERN), re.compile(URL_PATTERN), re.compile(DATE_PATTERN)]) and not any(kw in clean.lower() for kw in STOP_WORDS):
            if section_name == "Certifications" and not any(kw in clean.lower() for kw in cert_keywords):
                continue
            if section_name == "Volunteering" and not any(vkw in clean.lower() for vkw in vol_kw):
                continue
            if current_item and re.match(r'\d{4}', clean):
                current_item += ' ' + clean
                items.append(current_item.title())
                current_item = ""
            else:
                current_item = clean if not current_item else current_item + ' ' + clean
    if current_item:
        items.append(current_item.title())
    return sorted(set(items[:10]))

def merge_extractions(pdfminer_data: Dict[str, Any], pypdf2_data: Dict[str, Any]) -> Dict[str, Any]:
    merged = pdfminer_data.copy()
    
    if not merged["Name"] and pypdf2_data["Name"]:
        merged["Name"] = pypdf2_data["Name"]
    
    for key in merged["Contact"]:
        if not merged["Contact"][key] and pypdf2_data["Contact"][key]:
            merged["Contact"][key] = pypdf2_data["Contact"][key]
    
    if len(pypdf2_data["Summary"]) > len(merged["Summary"]):
        merged["Summary"] = pypdf2_data["Summary"]
    
    merged["Skills"] = sorted(set(merged["Skills"] + pypdf2_data["Skills"]))
    
    all_edu = merged["Education"] + pypdf2_data["Education"]
    unique_edu = []
    for e in all_edu:
        if not any(difflib.SequenceMatcher(None, e['degree'].lower(), u['degree'].lower()).ratio() > 0.8 for u in unique_edu):
            unique_edu.append(e)
    merged["Education"] = sorted(unique_edu, key=lambda x: len(x["degree"] + x["institution"]), reverse=True)[:5]
    
    all_exp = merged["Experience"] + pypdf2_data["Experience"]
    unique_exp = []
    for e in all_exp:
        if not any(difflib.SequenceMatcher(None, e['role'].lower(), u['role'].lower()).ratio() > 0.8 for u in unique_exp):
            unique_exp.append(e)
    merged["Experience"] = sorted(unique_exp, key=lambda x: len(x["description"]), reverse=True)[:5]
    
    all_proj = merged["Projects"] + pypdf2_data["Projects"]
    unique_proj = []
    for p in all_proj:
        if not any(difflib.SequenceMatcher(None, p['name'].lower(), u['name'].lower()).ratio() > 0.8 for u in unique_proj):
            unique_proj.append(p)
    merged["Projects"] = sorted(unique_proj, key=lambda x: len(x["description"] + x["keywords"]), reverse=True)[:10]
    
    for field in ["Certifications", "Languages", "Volunteering", "Interests"]:
        merged[field] = sorted(set(merged[field] + pypdf2_data[field]))[:10]
    
    return merged

def parse_cv(pdfminer_text: str, pypdf2_text: str, raw_text: str, filename: str = "") -> Dict[str, Any]:
    sections = {
        "Name": "", 
        "Contact": {"email": "", "phone": "", "location": "", "url": ""},
        "Summary": "", 
        "Skills": [], 
        "Education": [], 
        "Projects": [], 
        "Experience": [], 
        "Certifications": [], 
        "Languages": [], 
        "Volunteering": [], 
        "Interests": []
    }
    
    if not pdfminer_text or not pypdf2_text or not raw_text:
        return {"raw": raw_text, **sections}

    pdfminer_data = parse_cv_single(pdfminer_text, raw_text, filename)
    pypdf2_data = parse_cv_single(pypdf2_text, raw_text, filename)
    merged_data = merge_extractions(pdfminer_data, pypdf2_data)
    merged_data["raw"] = raw_text
    return merged_data

def parse_cv_single(text: str, raw_text: str, filename: str = "") -> Dict[str, Any]:
    sections = {
        "Name": "", 
        "Contact": {"email": "", "phone": "", "location": "", "url": ""},
        "Summary": "", 
        "Skills": [], 
        "Education": [], 
        "Projects": [], 
        "Experience": [], 
        "Certifications": [], 
        "Languages": [], 
        "Volunteering": [], 
        "Interests": []
    }
    
    if not text:
        return sections

    text = normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    sections["Contact"] = extract_contact_info(raw_text)
    sections["Name"] = extract_name(lines, raw_text, sections["Contact"])

    current_section = None
    section_lines = defaultdict(list)
    summary_lines = []
    for i, line in enumerate(lines):
        clean = clean_line(line)
        if not clean:
            continue
        new_section, is_header = detect_section_header(clean, current_section)
        if is_header:
            current_section = new_section
            continue
        if current_section:
            section_lines[current_section].append(clean)
        else:
            if len(clean) > 20 and not any(kw in clean.lower() for kw in STOP_WORDS + [sections["Name"].lower()]) and not re.match(EMAIL_PATTERN, clean) and not re.match(PHONE_PATTERN, clean) and not re.match(URL_PATTERN, clean) and i < 20:
                summary_lines.append(clean)

    edu_lines = section_lines["Education"] or [l for l in lines if any(kw in l.lower() for kw in ["diplome", "degree", "bac", "licence", "ingenieur", "cycle", "preparatoire", "esprit", "ensit", "university", "school"])]
    sections["Education"] = parse_education(edu_lines)
    
    exp_lines = section_lines["Experience"] or [l for l in lines if any(kw in l.lower() for kw in ["stage", "intern", "pfe", "emploi", "job", "professional", "experience", "internship", "achievements", "tasks", "taches", "responsibilities"])]
    sections["Experience"] = parse_experience(exp_lines)
    
    proj_lines = section_lines["Projects"] or [l for l in lines if any(kw in l.lower() for kw in ["pfa", "pidev", "pi", "project", "projet", "application", "platform", "system"])]
    projects, project_keywords = parse_projects(proj_lines)
    sections["Projects"] = projects
    
    skill_lines = section_lines["Skills"] or [l for l in lines if any(kw in l.lower() for kw in ["skills", "competences", "expertise", "technologies"])]
    sections["Skills"] = parse_skills(skill_lines, project_keywords, '\n'.join(summary_lines), raw_text)
    
    sections["Summary"] = dedup_text(' '.join(summary_lines).strip())
    cert_lines = section_lines["Certifications"] or [l for l in lines if any(kw in l.lower() for kw in ["certificat", "badge", "award", "certified", "workshop", "certificates"])]
    sections["Certifications"] = parse_certifications(cert_lines)
    lang_kw = ["arabic", "arabe", "french", "francais", "english", "anglais", "german", "allemand", "spanish", "espagnol", "italian", "italien", "chinese", "mandarin"]
    sections["Languages"] = [lang.strip().title() for lang in re.split(r'[,|;]', ' '.join(section_lines["Languages"]).replace('(', '').replace(')', '').replace('proficiency', '')) if lang.strip() and len(lang) > 3 and not re.match(DATE_PATTERN, lang) and any(lkw in lang.lower() for lkw in lang_kw)]
    sections["Volunteering"] = parse_generic_list(section_lines["Volunteering"] or [l for l in lines if any(kw in l.lower() for kw in ["club", "association", "volunteering", "vie associative", "extracurricular", "member"])], "Volunteering")
    sections["Interests"] = parse_generic_list(section_lines["Interests"], "Interests")

    return sections

def process_pdfs(folder_path: str, output_file: str = "cv_structured_2.json") -> List[Dict[str, Any]]:
    results = []
    processed_keys = set()
    if not os.path.exists(folder_path):
        logging.error(f"Folder path {folder_path} does not exist")
        return results

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logging.warning(f"No PDF files found in {folder_path}")
        return results

    for filename in pdf_files:
        path = os.path.join(folder_path, filename)
        logging.info(f"Processing {filename}")
        pdfminer_text, pypdf2_text = extract_text_with_layout(path)
        raw_text = pdfminer_text or pypdf2_text
        if not raw_text:
            logging.error(f"No text extracted from {filename}")
            continue
        cv_data = parse_cv(pdfminer_text, pypdf2_text, raw_text, filename)
        email = cv_data["Contact"]["email"]
        name = cv_data["Name"]
        key = (email, name)
        if key in processed_keys:
            logging.warning(f"Skipping duplicate CV for email: {email}, name: {name}")
            continue
        processed_keys.add(key)
        results.append(cv_data)
        logging.info(f"Processed {filename} → Name: {cv_data['Name']}")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        logging.error(f"Error saving JSON: {e}")

    return results

if __name__ == "__main__":
    folder_path = "/content/"
    results = process_pdfs(folder_path)
    print("\nExtraction completed!")   