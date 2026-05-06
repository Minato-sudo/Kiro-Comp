"""
Heuristic-based training pair generator (Fallback for N3).
Generates (resume, JD, label) pairs without requiring an external LLM.
This ensures the pipeline continues even without API keys.
"""

import json
import random
import os
from pathlib import Path

def generate_heuristic_jds(resume: dict) -> list[dict]:
    skills = resume.get("skills_list", [])
    sections = resume.get("sections", {})
    
    # 1. Positive JD: Built from actual resume skills
    pos_skills = skills[:5] if len(skills) >= 5 else skills
    pos_jd = {
        "title": f"Senior {skills[0].title() if skills else 'Software'} Engineer",
        "company": "GrowthTech Inc.",
        "description": f"We are looking for a specialist with expertise in {', '.join(pos_skills)}. " + 
                       f"The ideal candidate has experience in {sections.get('experience', '')[:100]}...",
        "required_skills": pos_skills,
        "match_score": random.randint(80, 95)
    }
    
    # 2. Hard Negative: Adjacent skills
    adjacent_skills = ["Management", "Leadership", "Budgeting", "Sales"]
    hard_neg_jd = {
        "title": "Technical Project Manager",
        "company": "BizCorp",
        "description": f"Focus on {', '.join(adjacent_skills)}. Background in {pos_skills[0] if pos_skills else 'Tech'} preferred.",
        "required_skills": adjacent_skills + (pos_skills[:1] if pos_skills else []),
        "match_score": random.randint(40, 55)
    }
    
    # 3. Easy Negative: Completely unrelated
    easy_neg_jd = {
        "title": "Head Chef",
        "company": "The Grand Hotel",
        "description": "Leading kitchen operations, menu planning, and staff management in a high-pressure environment.",
        "required_skills": ["Culinary Arts", "Menu Design", "Food Safety"],
        "match_score": random.randint(5, 15)
    }
    
    pairs = []
    for pair_type, jd_data in [
        ("positive", pos_jd),
        ("hard_negative", hard_neg_jd),
        ("easy_negative", easy_neg_jd),
    ]:
        pairs.append({
            "resume_file":   resume["file_path"],
            "resume_text":   resume["full_text"],
            "resume_sections": resume["sections"],
            "jd_title":      jd_data["title"],
            "jd_text":       jd_data["description"],
            "jd_skills":     jd_data["required_skills"],
            "pair_type":     pair_type,
            "llm_score":     jd_data["match_score"],
            "label":         1 if pair_type == "positive" else 0,
        })
    return pairs

def run_phase2_fallback(parsed_path: str = "data/processed/parsed_resumes.jsonl",
                        output_path: str = "data/processed/training_pairs.jsonl"):
    if not os.path.exists(parsed_path):
        print(f"Error: {parsed_path} not found.")
        return
        
    resumes = []
    with open(parsed_path) as f:
        for line in f:
            resumes.append(json.loads(line))
            
    print(f"Generating heuristic pairs for {len(resumes)} resumes...")
    
    all_pairs = []
    for resume in resumes:
        all_pairs.extend(generate_heuristic_jds(resume))
        
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Successfully generated {len(all_pairs)} pairs.")
    return all_pairs

if __name__ == "__main__":
    run_phase2_fallback()
