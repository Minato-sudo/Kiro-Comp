import numpy as np
import pandas as pd
from typing import List, Dict, Callable

class BiasAudit:
    """
    N6: Bias audit system.
    Measures whether the model assigns systematically lower scores to specific demographic indicators.
    """
    def __init__(self):
        # Sensitive indicators to check for bias
        self.sensitive_terms = {
            "gender": {
                "female": ["she", "her", "hers", "women's", "female", "sister", "mother", "mrs", "ms"],
                "male": ["he", "him", "his", "men's", "male", "brother", "father", "mr"]
            },
            "age": {
                "older": ["senior", "experienced", "leader", "years of experience", "10+ years", "director"],
                "younger": ["junior", "entry-level", "intern", "recent graduate", "fresher"]
            }
        }

    def detect_attributes(self, text: str) -> Dict[str, str]:
        """Detect potential demographic attributes based on keyword presence."""
        text = text.lower()
        results = {}
        for category, groups in self.sensitive_terms.items():
            for group_name, terms in groups.items():
                if any(term in text for term in terms):
                    results[category] = group_name
                    break
        return results

    def audit_model(self, model_scorer: Callable[[str, str], float], resumes: List[str], jds: List[str]):
        """
        Runs an audit over a set of resumes and JDs.
        Returns a report on score parity across detected attributes.
        """
        results = []
        for resume in resumes:
            attributes = self.detect_attributes(resume)
            if not attributes:
                continue
                
            for jd in jds:
                score = model_scorer(resume, jd)
                results.append({
                    **attributes,
                    "score": score
                })
        
        if not results:
            return "No sensitive attributes detected in audit set."
            
        df = pd.DataFrame(results)
        report = {}
        
        for category in self.sensitive_terms.keys():
            if category in df.columns:
                group_means = df.groupby(category)["score"].mean().to_dict()
                report[category] = group_means
                
        return report

    def mitigate_bias(self, text: str) -> str:
        """Simple mitigation: strip sensitive terms before encoding."""
        # In a real system, this would involve adversarial training or re-weighting
        modified_text = text.lower()
        for groups in self.sensitive_terms.values():
            for terms in groups.values():
                for term in terms:
                    modified_text = modified_text.replace(f" {term} ", " ")
        return modified_text
