import os
import json
from groq import Groq
import copy
from typing import Dict, Any, List

# Initialize Groq client (requires GROQ_API_KEY environment variable)
# Using Groq with Llama 3 for fast LLM inference
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))
except:
    client = None

def get_llm_labels(resume_text: str, jd_text: str) -> dict:
    """
    N3: LLM as weak labeller.
    Prompts LLM to score match 0-100 with reasoning.
    Used for knowledge distillation to train the dual encoder.
    """
    if not client: return {"score": 0, "reasoning": "No API key"}
    
    prompt = f"""
    Evaluate the compatibility between this Resume and Job Description.
    Score the match from 0 to 100.
    
    Resume:
    {resume_text}
    
    Job Description:
    {jd_text}
    
    Return a JSON object with:
    - score: integer 0-100
    - reasoning: short paragraph explaining why
    - missing_skills: list of strings
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"score": 50, "reasoning": "Failed to parse", "missing_skills": []}

def counterfactual_rewrite(
    resume_sections: List[str], 
    jd_sentences: List[str], 
    weak_bullet: str, 
    unmatched_requirement: str,
    model_scorer_func
) -> Dict[str, Any]:
    """
    N5: Counterfactual rewriting with delta-score measurement.
    Rewrites a weak bullet, immediately re-scores the whole resume, 
    and returns the delta score.
    
    model_scorer_func: A function that takes (resume_sections, jd_sentences) and returns a score [0-100]
    """
    # 1. Base Score
    base_score = model_scorer_func(resume_sections, jd_sentences)
    
    # 2. Rewrite Generation
    prompt = f"""
    Task: Rewrite the resume bullet below to better align with the Job Description requirement.
    Keep it truthful — only enhance phrasing and framing. Include relevant skills if implicit.
    
    Requirement: "{unmatched_requirement}"
    Current bullet: "{weak_bullet}"
    
    Return ONLY the rewritten bullet point text, nothing else.
    """
    
    try:
        if client:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
            )
            rewritten_bullet = response.choices[0].message.content.strip()
        else:
            rewritten_bullet = f"Enhanced: {weak_bullet} (Aligned with {unmatched_requirement})"
    except:
        rewritten_bullet = weak_bullet
        
    # 3. Re-evaluate Score
    # We substitute the weak_bullet with rewritten_bullet in the sections
    new_resume_sections = copy.deepcopy(resume_sections)
    for i in range(len(new_resume_sections)):
        new_resume_sections[i] = new_resume_sections[i].replace(weak_bullet, rewritten_bullet)
        
    new_score = model_scorer_func(new_resume_sections, jd_sentences)
    delta = new_score - base_score
    
    return {
        "original_bullet": weak_bullet,
        "rewritten_bullet": rewritten_bullet,
        "base_score": round(base_score, 1),
        "new_score": round(new_score, 1),
        "delta": round(delta, 1),
        "explanation": f"If you use this rewrite, your score goes from {round(base_score, 1)} → {round(new_score, 1)} (+{round(delta, 1)})"
    }

def generate_suggestions(resume_data: dict, jd_data: dict, score_output: dict) -> str:
    """Standard grounded suggestions generation."""
    if not client: return "Suggestions generation requires Groq API key."
    prompt = f"""
    Resume candidate: {resume_data.get('name', 'Candidate')}
    Job: {jd_data.get('title', 'Job')} at {jd_data.get('company', 'Company')}
    Match score: {score_output.get('total', 50)}/100

    Missing required skills: {score_output.get('missing_skills', [])}
    Weakly matched experience: {score_output.get('weak_matches', [])}

    Task: Provide 3 concise, actionable suggestions for improving this resume for this specific job.
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"
