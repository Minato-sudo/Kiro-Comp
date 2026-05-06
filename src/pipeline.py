import os
import fitz  # PyMuPDF
from docx import Document
import re
from typing import List, Dict, Any
from src.model import DualEncoderMatcher, ScoreCalibration
from src.ontology import skill_graph
import torch

class ResumeParser:
    """Parses text from PDF and DOCX files and segments into sections."""
    @staticmethod
    def extract_text(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        elif ext == ".docx":
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        return ""

    @staticmethod
    def segment_sections(text: str) -> Dict[str, str]:
        """Simple rule-based segmentation into canonical sections."""
        sections = {
            "summary": "",
            "skills": "",
            "experience": "",
            "education": "",
            "projects": ""
        }
        
        patterns = {
            "skills": r"(?i)skills|technical skills|competencies",
            "experience": r"(?i)experience|work history|employment",
            "education": r"(?i)education|academic background",
            "projects": r"(?i)projects|academic projects"
        }
        
        lines = text.split('\n')
        current_section = "summary"
        
        for line in lines:
            found = False
            for sec, pattern in patterns.items():
                if re.search(pattern, line):
                    current_section = sec
                    found = True
                    break
            if not found:
                sections[current_section] += line + "\n"
        
        return sections

class MatcherPipeline:
    def __init__(self, model_path="models/fine_tuned_matcher.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualEncoderMatcher().to(self.device)
        
        # Load fine-tuned weights if they exist
        if os.path.exists(model_path):
            print(f"Loading fine-tuned weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print("Using base pre-trained model weights.")
            
        self.model.eval()
        self.calibrator = ScoreCalibration()
        
    def match(self, resume_path: str, jd_text: str) -> Dict[str, Any]:
        # 1. Parse and Segment
        resume_text = ResumeParser.extract_text(resume_path)
        resume_sections_dict = ResumeParser.segment_sections(resume_text)
        
        resume_sections_list = [
            resume_sections_dict["skills"],
            resume_sections_dict["experience"],
            resume_sections_dict["education"],
            resume_sections_dict["projects"],
            resume_sections_dict["summary"]
        ]
        
        # 2. Process JD Sentences
        jd_sentences = [s.strip() for s in jd_text.split('.') if len(s.strip()) > 10]
        if not jd_sentences:
            jd_sentences = [jd_text]

        # 3. Model Inference
        with torch.no_grad():
            res_proj, jd_proj, attn_weights = self.model([resume_sections_list], [jd_sentences])
            
        # Raw cosine similarity
        raw_score = torch.sum(res_proj * jd_proj).item()
        
        # 4. Calibration
        calibrated_prob = self.calibrator.predict_proba(raw_score)
        
        # 5. Skill Analysis (Ontology based)
        # Mock skill extraction (would use SpaCy/NLP in production)
        resume_skills = set(re.findall(r"\w+", resume_sections_dict["skills"].lower()))
        jd_skills = set(re.findall(r"\w+", jd_text.lower()))
        
        found_skills = resume_skills.intersection(jd_skills)
        missing_skills = jd_skills - resume_skills
        
        # Deep semantic overlap check via ontology
        semantic_matches = []
        for m_skill in missing_skills:
            best_sim = 0
            best_ref = ""
            for r_skill in resume_skills:
                sim = skill_graph.skill_similarity(m_skill, r_skill)
                if sim > best_sim:
                    best_sim = sim
                    best_ref = r_skill
            if best_sim > 0.5:
                semantic_matches.append({"jd_skill": m_skill, "resume_ref": best_ref, "sim": best_sim})

        return {
            "total_score": round(calibrated_prob * 100, 1),
            "raw_similarity": round(raw_score, 3),
            "sections": resume_sections_dict,
            "jd_sentences": jd_sentences,
            "skills": {
                "direct_overlap": list(found_skills)[:10],
                "missing": list(missing_skills)[:10],
                "semantic_near_misses": semantic_matches
            },
            "attention_map": attn_weights[0].cpu().numpy()
        }
