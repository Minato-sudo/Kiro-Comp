"""
Skill gap analysis: identifies missing and weakly matched skills.
"""

from dataclasses import dataclass
from src.ontology.dynamic_graph import DynamicSkillGraph

STRONG_MATCH_THRESHOLD = 0.7
WEAK_MATCH_THRESHOLD   = 0.3


@dataclass
class SkillGapReport:
    matched_skills:        list[tuple[str, str, float]]  
    weak_matches:          list[tuple[str, str, float]]
    missing_skills:        list[str]                     
    transferable_skills:   list[tuple[str, str, float]]  


class SkillGapAnalyzer:
    
    def __init__(self, skill_graph: DynamicSkillGraph):
        self.graph = skill_graph
    
    def analyze(self, resume_skills: list[str], jd_skills: list[str]) -> SkillGapReport:
        
        matched, weak, transferable = [], [], []
        jd_covered = set()
        
        for jd_skill in jd_skills:
            best_score = 0.0
            best_resume_skill = None
            
            for res_skill in resume_skills:
                sim = self.graph.skill_similarity(res_skill, jd_skill)
                if sim > best_score:
                    best_score = sim
                    best_resume_skill = res_skill
            
            if best_score >= STRONG_MATCH_THRESHOLD:
                matched.append((best_resume_skill, jd_skill, best_score))
                jd_covered.add(jd_skill)
            elif best_score >= WEAK_MATCH_THRESHOLD:
                weak.append((best_resume_skill, jd_skill, best_score))
                jd_covered.add(jd_skill)
            else:
                related = self.graph.get_related_skills(jd_skill, top_k=3)
                transfer_found = False
                for rel_skill, rel_sim in related:
                    if any(self.graph.skill_similarity(r, rel_skill) > 0.5 
                           for r in resume_skills):
                        transferable.append((rel_skill, jd_skill, rel_sim * 0.5))
                        transfer_found = True
                        break
        
        missing = [s for s in jd_skills if s not in jd_covered and
                   s not in [t[1] for t in transferable]]
        
        return SkillGapReport(
            matched_skills=sorted(matched, key=lambda x: -x[2]),
            weak_matches=sorted(weak, key=lambda x: x[2]),
            missing_skills=missing,
            transferable_skills=transferable,
        )
