"""
N5: Counterfactual rewriting with delta-score measurement.
Uses Groq for re-scoring.
"""

import os
import json
from groq import Groq
from dataclasses import dataclass
from src.matching.scorer import HierarchicalScorer

client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

@dataclass
class CounterfactualResult:
    original_bullet:     str
    rewritten_bullet:    str
    original_score:      float
    new_score:           float
    delta:               float    
    jd_requirement:      str      
    is_truthful_rewrite: bool


REWRITE_PROMPT = """You are a professional resume editor helping a job seeker improve their resume.

Job requirement to target: "{jd_requirement}"

Current bullet point: "{original_bullet}"

The candidate's match score is currently {current_score:.0f}%. 
This bullet only matched at {match_strength:.0f}% strength.

Rewrite ONLY this bullet to better demonstrate alignment with the job requirement.
Rules:
1. Keep it truthful — only rephrase/reframe what was already there
2. Use action verbs and quantify where reasonable
3. Match the terminology used in the job requirement
4. Keep it to 1-2 sentences maximum

Return JSON only:
{{
  "rewritten_bullet": "...",
  "reasoning": "what I changed and why",
  "is_truthful": true
}}"""


class CounterfactualRewriter:
    
    def __init__(self, scorer: HierarchicalScorer, tokenizer, model):
        self.scorer = scorer
        self.tokenizer = tokenizer
        self.model = model  

    def find_weakest_bullet(self, experience_section: str, jd_text: str,
                             resume_sections: dict) -> tuple[str, str, float]:
        bullets = [b.strip() for b in experience_section.split("\n")
                   if b.strip() and len(b.strip()) > 20]
        
        if not bullets:
            return experience_section[:200], "relevant experience", 0.0
        
        worst_score = 1.0
        worst_bullet = bullets[0]
        
        for bullet in bullets[:5]:  
            temp_sections = dict(resume_sections)
            temp_sections["experience"] = bullet
            score = self.scorer.compute_score(temp_sections, jd_text)
            
            if score.raw_similarity < worst_score:
                worst_score = score.raw_similarity
                worst_bullet = bullet
        
        return worst_bullet, "job requirements", worst_score
    
    def rewrite_with_delta(self, resume_sections: dict, jd_text: str,
                            jd_skills: list[str] = None) -> CounterfactualResult:
        baseline = self.scorer.compute_score(resume_sections, jd_text)
        
        experience = resume_sections.get("experience", "")
        worst_bullet, jd_req, match_strength = self.find_weakest_bullet(
            experience, jd_text, resume_sections
        )
        
        target_req = jd_skills[0] if jd_skills else "the job requirements"
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{
                "role": "user",
                "content": REWRITE_PROMPT.format(
                    jd_requirement=target_req,
                    original_bullet=worst_bullet,
                    current_score=baseline.overall_score,
                    match_strength=match_strength * 100,
                )
            }],
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content.strip()
        result_data = json.loads(raw)
        rewritten_bullet = result_data["rewritten_bullet"]
        
        new_sections = dict(resume_sections)
        new_sections["experience"] = experience.replace(worst_bullet, rewritten_bullet, 1)
        
        new_score_obj = self.scorer.compute_score(new_sections, jd_text)
        delta = new_score_obj.overall_score - baseline.overall_score
        
        return CounterfactualResult(
            original_bullet=worst_bullet,
            rewritten_bullet=rewritten_bullet,
            original_score=baseline.overall_score,
            new_score=new_score_obj.overall_score,
            delta=delta,
            jd_requirement=target_req,
            is_truthful_rewrite=result_data.get("is_truthful", True),
        )
