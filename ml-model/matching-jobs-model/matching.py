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