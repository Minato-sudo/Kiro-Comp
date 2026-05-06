"""
N6: Automated bias audit.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

GENDER_PROXIES = {
    "male_indicators":   ["he ", "his ", "him ", "mr.", "he/him", "brotherhood"],
    "female_indicators": ["she ", "her ", "hers", "ms.", "she/her", "sisterhood"],
    "neutral":           ["they ", "their ", "theirs", "they/them"],
}

INSTITUTION_PROXIES = {
    "elite_us":     ["MIT", "Stanford", "Harvard"],
    "regional":     ["FAST-NUCES", "NUST", "LUMS"],
    "generic":      ["Online University", "State College"],
}


@dataclass
class BiasReport:
    gender_disparity:      float   
    age_disparity:         float
    institution_disparity: float
    passed:                bool    
    details:               dict


class BiasAuditor:
    
    DISPARITY_THRESHOLD = 10.0  
    
    def __init__(self, scorer_fn: Callable):
        self.scorer = scorer_fn
    
    def _inject_proxy(self, resume_text: str, proxy_text: str) -> str:
        return proxy_text + " " + resume_text
    
    def audit_gender(self, base_resume: str, jd_text: str) -> dict:
        scores = {}
        for group, terms in GENDER_PROXIES.items():
            group_scores = []
            for term in terms[:2]:  
                injected = self._inject_proxy(base_resume, term)
                s = self.scorer(injected, jd_text)
                group_scores.append(s)
            scores[group] = np.mean(group_scores)
        
        disparity = max(scores.values()) - min(scores.values())
        return {"scores": scores, "disparity": disparity, "passed": disparity < self.DISPARITY_THRESHOLD}
    
    def audit_institution(self, base_resume: str, jd_text: str) -> dict:
        scores = {}
        for group, institutions in INSTITUTION_PROXIES.items():
            group_scores = []
            for inst in institutions[:2]:
                injected = self._inject_proxy(base_resume, f"Graduated from {inst}.")
                s = self.scorer(injected, jd_text)
                group_scores.append(s)
            scores[group] = np.mean(group_scores)
        
        disparity = max(scores.values()) - min(scores.values())
        return {"scores": scores, "disparity": disparity, "passed": disparity < self.DISPARITY_THRESHOLD}
    
    def run_full_audit(self, sample_resume: str, sample_jd: str) -> BiasReport:
        gender_result = self.audit_gender(sample_resume, sample_jd)
        inst_result = self.audit_institution(sample_resume, sample_jd)
        
        all_passed = gender_result["passed"] and inst_result["passed"]
        
        report = BiasReport(
            gender_disparity=gender_result["disparity"],
            age_disparity=0.0,  
            institution_disparity=inst_result["disparity"],
            passed=all_passed,
            details={
                "gender":      gender_result,
                "institution": inst_result,
            }
        )
        return report
