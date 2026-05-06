"""
Hierarchical matching scorer.
Computes section-level and overall match scores.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from src.model.dual_encoder import DualEncoderModel, SECTIONS
from src.model.calibration import PlattCalibration

@dataclass
class MatchScore:
    overall_score:       float         
    raw_similarity:      float         
    section_scores:      dict[str, float]  
    section_weights:     dict[str, float]  
    calibrated_message:  str           
    percentile:          float         


class HierarchicalScorer:
    
    def __init__(self, model: DualEncoderModel,
                 calibrator: PlattCalibration,
                 tokenizer,
                 device: str = "cpu"):
        self.model = model
        self.model.eval()
        self.calibrator = calibrator
        self.tokenizer = tokenizer
        self.device = device
    
    def encode_text(self, text: str, max_length: int = 256) -> tuple:
        enc = self.tokenizer(
            text if text else "[EMPTY]",
            max_length=max_length, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
    
    def encode_sections(self, sections: dict[str, str]) -> tuple:
        ids_list, masks_list = [], []
        for sec_name in SECTIONS:
            sec_text = sections.get(sec_name, "")
            ids, masks = self.encode_text(sec_text, max_length=128)
            ids_list.append(ids.squeeze(0))
            masks_list.append(masks.squeeze(0))
        
        sec_ids = torch.stack(ids_list).unsqueeze(0)    
        sec_masks = torch.stack(masks_list).unsqueeze(0)
        return sec_ids, sec_masks
    
    @torch.no_grad()
    def compute_score(self, resume_sections: dict[str, str], jd_text: str) -> MatchScore:
        sec_ids, sec_masks = self.encode_sections(resume_sections)
        jd_ids, jd_masks = self.encode_text(jd_text, max_length=256)
        
        r_emb, j_emb = self.model(sec_ids, sec_masks, jd_ids, jd_masks)
        
        raw_sim = (r_emb * j_emb).sum(dim=-1).item()
        
        section_scores = {}
        for sec_name in SECTIONS:
            sec_text = resume_sections.get(sec_name, "")
            if not sec_text:
                section_scores[sec_name] = 0.0
                continue
            
            single_ids, single_masks = self.encode_text(sec_text, max_length=128)
            sec_emb = self.model.jd_encoder(single_ids, single_masks)
            sec_sim = (sec_emb * j_emb).sum(dim=-1).item()
            section_scores[sec_name] = max(0.0, (sec_sim + 1) / 2)  
        
        learned_weights = self.model.get_section_weights()
        
        overall_pct = self.calibrator.predict_proba(raw_sim)
        percentile = self.calibrator.percentile_rank(raw_sim)
        
        message = f"You are a {overall_pct:.0f}% fit for this role"
        if percentile > 50:
            message += f" — stronger than {percentile:.0f}% of typical applicants."
        else:
            message += f". There is room to improve (bottom {100-percentile:.0f}% range)."
        
        return MatchScore(
            overall_score=overall_pct,
            raw_similarity=raw_sim,
            section_scores=section_scores,
            section_weights=learned_weights,
            calibrated_message=message,
            percentile=percentile,
        )
