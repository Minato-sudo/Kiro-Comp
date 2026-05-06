"""
Generates (resume, JD, label) training pairs using an LLM.
This is N3 — LLM as Weak Labeller.
Run this ONCE offline. 
Uses Groq for speed and cost efficiency.
"""

import json
import time
import os
from pathlib import Path
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

GENERATION_PROMPT = """You are a recruitment expert. Given a resume, generate 3 job descriptions:

1. POSITIVE JD: A job this candidate would be 75-90% fit for based on their actual skills
2. HARD NEGATIVE JD: A job they would be 40-55% fit for (adjacent field, some skill overlap)  
3. EASY NEGATIVE JD: A job they would be 0-20% fit for (completely different field)

Resume:
{resume_text}

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{{
  "positive_jd": {{
    "title": "Job title",
    "company": "Company name",
    "description": "Full job description with requirements (150-200 words)",
    "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "match_score": 82
  }},
  "hard_negative_jd": {{
    "title": "Job title",
    "company": "Company name", 
    "description": "Full job description (150-200 words)",
    "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "match_score": 47
  }},
  "easy_negative_jd": {{
    "title": "Job title",
    "company": "Company name",
    "description": "Full job description (150-200 words)",
    "required_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
    "match_score": 12
  }}
}}"""

def generate_pairs_for_resume(resume: dict, max_retries: int = 3) -> list[dict]:
    """Generate 3 JDs for one resume. Returns list of pair dicts."""
    
    resume_text = f"""
Name/Profile: {resume['sections'].get('summary', '')}
Skills: {resume['sections'].get('skills', '')}
Experience: {resume['sections'].get('experience', '')[:600]}
Education: {resume['sections'].get('education', '')}
Projects: {resume['sections'].get('projects', '')}
""".strip()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{
                    "role": "user",
                    "content": GENERATION_PROMPT.format(resume_text=resume_text)
                }],
                response_format={"type": "json_object"}
            )
            
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)
            
            pairs = []
            for pair_type, jd_data in [
                ("positive", data["positive_jd"]),
                ("hard_negative", data["hard_negative_jd"]),
                ("easy_negative", data["easy_negative_jd"]),
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
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    
    return []  # failed after retries


def run_phase2(parsed_path: str = "data/processed/parsed_resumes.jsonl",
               output_path: str = "data/processed/training_pairs.jsonl",
               batch_size: int = 10,
               limit: int = 50): # Added limit for demo/speed
    """Generate pairs for all resumes."""
    
    resumes = []
    if not os.path.exists(parsed_path):
        print(f"Error: {parsed_path} not found. Run Phase 1 first.")
        return []
        
    with open(parsed_path) as f:
        for line in f:
            resumes.append(json.loads(line))
    
    if limit:
        resumes = resumes[:limit]
        
    print(f"Generating JD pairs for {len(resumes)} resumes...")
    
    all_pairs = []
    failed_count = 0
    
    for i, resume in enumerate(resumes):
        pairs = generate_pairs_for_resume(resume)
        if pairs:
            all_pairs.extend(pairs)
        else:
            failed_count += 1
        
        if (i + 1) % batch_size == 0:
            print(f"  Processed {i+1}/{len(resumes)} resumes, {len(all_pairs)} pairs so far")
            # Save checkpoint
            with open(output_path, "w") as f:
                for p in all_pairs:
                    f.write(json.dumps(p) + "\n")
        
        time.sleep(0.5)  # rate limit buffer
    
    # Final save
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    
    print(f"\nTotal pairs generated: {len(all_pairs)}")
    print(f"Failed resumes: {failed_count}")
    
    # VALIDATION CHECK
    positive_count = sum(1 for p in all_pairs if p["label"] == 1)
    # Adjusted check for limited run
    if len(resumes) >= 500:
        assert positive_count >= 400, f"Too few positive pairs: {positive_count}"
    print(f"Positive pairs: {positive_count}, Negative pairs: {len(all_pairs) - positive_count}")
    return all_pairs


if __name__ == "__main__":
    run_phase2()
