"""
Resume parser that segments resumes into 5 canonical sections.
Uses rule-based patterns + spaCy for named entity recognition.
This runs without any trained model — works immediately.
"""

import re
import json
import spacy
from pathlib import Path
from typing import Optional
import fitz  # PyMuPDF

nlp = spacy.load("en_core_web_sm")

SECTION_PATTERNS = {
    "summary": [
        r"(?i)(objective|summary|profile|about me|career objective|professional summary)"
    ],
    "education": [
        r"(?i)(education|academic|qualification|degree|university|college)"
    ],
    "experience": [
        r"(?i)(experience|employment|work history|professional experience|internship)"
    ],
    "skills": [
        r"(?i)(skills|technical skills|competencies|expertise|technologies|tools)"
    ],
    "projects": [
        r"(?i)(projects|research|portfolio|personal projects|academic projects)"
    ],
    "achievements": [
        r"(?i)(achievement|award|certification|honor|extracurricular|volunteer)"
    ],
}

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_path: str) -> str:
    from docx import Document
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def detect_section_boundaries(lines: list[str]) -> dict[str, int]:
    """Returns {section_name: line_index} for each detected section header."""
    boundaries = {}
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if len(line_stripped) < 3 or len(line_stripped) > 60:
            continue
        for section, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line_stripped):
                    if section not in boundaries:
                        boundaries[section] = i
                    break
    return boundaries

def segment_resume(text: str) -> dict[str, str]:
    """
    Splits resume text into 5 canonical sections.
    Falls back to heuristics if headers not found.
    """
    lines = text.split("\n")
    boundaries = detect_section_boundaries(lines)
    
    sections = {s: "" for s in ["summary", "education", "experience", "skills", "projects", "achievements"]}
    
    if not boundaries:
        # Fallback: treat entire text as experience
        sections["experience"] = text
        return sections
    
    sorted_boundaries = sorted(boundaries.items(), key=lambda x: x[1])
    
    for idx, (section_name, start_line) in enumerate(sorted_boundaries):
        if idx + 1 < len(sorted_boundaries):
            end_line = sorted_boundaries[idx + 1][1]
        else:
            end_line = len(lines)
        
        section_text = "\n".join(lines[start_line:end_line]).strip()
        sections[section_name] = section_text
    
    # If no summary found, use first 3 lines
    if not sections["summary"] and lines:
        sections["summary"] = "\n".join(lines[:3])
    
    return sections

def extract_skills_from_text(text: str) -> list[str]:
    """Extract skill mentions using pattern matching + NLP."""
    # Common tech skills pattern
    tech_pattern = r'\b(Python|Java|C\+\+|JavaScript|TypeScript|React|Vue|Angular|Node\.?js|' \
                   r'Django|Flask|FastAPI|SQL|MySQL|PostgreSQL|MongoDB|Redis|Docker|Kubernetes|' \
                   r'AWS|Azure|GCP|Git|Linux|TensorFlow|PyTorch|scikit-learn|pandas|numpy|' \
                   r'HTML|CSS|REST|GraphQL|Figma|Jira|Agile|Scrum|R|MATLAB|Tableau|Power BI|' \
                   r'Machine Learning|Deep Learning|NLP|Computer Vision|Data Analysis)\b'
    
    found = re.findall(tech_pattern, text, re.IGNORECASE)
    
    # Also extract from skills section specifically
    doc = nlp(text[:5000])  # limit for speed
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks 
                   if 2 <= len(chunk.text.split()) <= 4]
    
    all_skills = list(set([s.lower() for s in found] + noun_chunks[:20]))
    return all_skills[:50]  # cap at 50

def parse_resume_file(file_path: str) -> dict:
    """Main entry point. Returns structured resume dict."""
    path = Path(file_path)
    
    if path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif path.suffix.lower() == ".docx":
        text = extract_text_from_docx(file_path)
    elif path.suffix.lower() in [".txt", ".text"]:
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    
    sections = segment_resume(text)
    skills = extract_skills_from_text(sections.get("skills", "") + " " + sections.get("experience", ""))
    
    # Build combined text for each section (used by encoder)
    section_texts = {
        "summary":      sections.get("summary", "")[:512],
        "skills":       sections.get("skills", "")[:512],
        "experience":   sections.get("experience", "")[:1024],
        "education":    sections.get("education", "")[:512],
        "projects":     sections.get("projects", "")[:512],
        "achievements": sections.get("achievements", "")[:256],
    }
    
    return {
        "file_path":     str(file_path),
        "full_text":     text[:3000],
        "sections":      section_texts,
        "skills_list":   skills,
        "section_count": sum(1 for v in section_texts.values() if len(v) > 20),
    }


def run_phase1(resumes_dir: str = "data/raw/resumes",
               output_path: str = "data/processed/parsed_resumes.jsonl"):
    """Parse all resumes and save to JSONL."""
    import os
    resume_dir = Path(resumes_dir)
    # Support more extensions if needed
    files = list(resume_dir.glob("*.pdf")) + list(resume_dir.glob("*.txt")) + list(resume_dir.glob("*.docx"))
    
    print(f"Found {len(files)} resume files")
    
    parsed = []
    failed = []
    
    for f in files:
        try:
            result = parse_resume_file(str(f))
            if result["section_count"] >= 2:  # at least 2 sections
                parsed.append(result)
            else:
                failed.append(str(f))
        except Exception as e:
            failed.append(f"{f}: {e}")
    
    with open(output_path, "w") as out:
        for item in parsed:
            out.write(json.dumps(item) + "\n")
    
    print(f"Successfully parsed: {len(parsed)}")
    print(f"Failed/skipped:      {len(failed)}")
    print(f"Output: {output_path}")
    
    # VALIDATION CHECK
    assert len(parsed) >= 500, f"Too few usable resumes: {len(parsed)}. Check your input files."
    return parsed


if __name__ == "__main__":
    run_phase1()
