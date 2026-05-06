"""
Main inference pipeline. Assembles all components.
"""

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.getcwd())

from src.model.dual_encoder import DualEncoderModel
from src.model.cross_attention import AsymmetricCrossAttention
from src.model.calibration import PlattCalibration
from src.matching.scorer import HierarchicalScorer, MatchScore
from src.matching.gap_analyzer import SkillGapAnalyzer, SkillGapReport
from src.ontology.esco_client import ESCOClient
from src.ontology.dynamic_graph import DynamicSkillGraph
from src.explainability.counterfactual import CounterfactualRewriter
from src.data.resume_parser import parse_resume_file, extract_skills_from_text
from transformers import AutoTokenizer

BERT_MODEL = "bert-base-uncased"


@dataclass
class FullAnalysis:
    match_score:      MatchScore
    skill_gaps:       SkillGapReport
    counterfactual:   Optional[object]   
    attention_map:    Optional[dict]
    resume_sections:  dict[str, str]


class ResumeMatchingPipeline:
    
    def __init__(self, model_path: str = "models/best_model.pt",
                 calibration_path: str = "models/calibration.pkl",
                 skill_graph_path: str = "models/skill_graph.pkl",
                 device: str = None):
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        
        self.model = DualEncoderModel()
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}. Using base weights.")
        self.model.to(self.device)
        self.model.eval()
        
        if Path(calibration_path).exists():
            self.calibrator = PlattCalibration.load(calibration_path)
            print(f"✓ Calibrator loaded")
        else:
            print("⚠ Calibrator not fitted. Using dummy calibration.")
            self.calibrator = PlattCalibration()
            import numpy as np
            self.calibrator.fit(np.array([0.2, 0.5, 0.8]), np.array([0, 0, 1]))
        
        if Path(skill_graph_path).exists():
            self.skill_graph = DynamicSkillGraph.load(skill_graph_path)
            print(f"✓ Skill graph loaded")
        else:
            esco = ESCOClient()
            self.skill_graph = DynamicSkillGraph(esco)
            print("⚠ Skill graph not trained. Using static ESCO.")
        
        self.scorer = HierarchicalScorer(
            model=self.model,
            calibrator=self.calibrator,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        
        self.gap_analyzer = SkillGapAnalyzer(self.skill_graph)
        
        self.rewriter = CounterfactualRewriter(
            scorer=self.scorer,
            tokenizer=self.tokenizer,
            model=self.model,
        )
    
    def analyze_from_file(self, resume_path: str, jd_text: str,
                          jd_skills: list[str] = None,
                          include_counterfactual: bool = True) -> FullAnalysis:
        parsed = parse_resume_file(resume_path)
        return self.analyze(parsed["sections"], parsed["skills_list"], 
                           jd_text, jd_skills, include_counterfactual)
    
    def analyze(self, resume_sections: dict, resume_skills: list[str],
                jd_text: str, jd_skills: list[str] = None,
                include_counterfactual: bool = True) -> FullAnalysis:
        
        if jd_skills is None:
            jd_skills = extract_skills_from_text(jd_text)
        
        match_score = self.scorer.compute_score(resume_sections, jd_text)
        skill_gaps = self.gap_analyzer.analyze(resume_skills, jd_skills)
        
        counterfactual = None
        if include_counterfactual:
            try:
                counterfactual = self.rewriter.rewrite_with_delta(
                    resume_sections, jd_text, skill_gaps.missing_skills[:3]
                )
            except Exception as e:
                print(f"Counterfactual failed: {e}")
        
        return FullAnalysis(
            match_score=match_score,
            skill_gaps=skill_gaps,
            counterfactual=counterfactual,
            attention_map=match_score.section_scores,
            resume_sections=resume_sections,
        )
