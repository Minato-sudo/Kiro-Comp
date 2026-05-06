# training/fit_calibration.py
"""Run this AFTER Phase 6 to fit the Platt scaler on the validation set."""

import json
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.getcwd())

from src.model.dual_encoder import DualEncoderModel, SECTIONS
from src.model.calibration import PlattCalibration
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

BERT_MODEL = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "models/"

def collect_scores_and_labels(model_path: str, val_path: str):
    """Run trained model on val set, collect raw scores + true labels."""
    
    model = DualEncoderModel().to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return [], []
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Load ALL val pairs (positive and negative)
    all_pairs = []
    if not os.path.exists(val_path):
        print(f"Error: {val_path} not found.")
        return [], []
        
    with open(val_path) as f:
        for line in f:
            all_pairs.append(json.loads(line))
    
    scores = []
    labels = []
    
    print(f"Collecting scores for {len(all_pairs)} pairs...")
    with torch.no_grad():
        for pair in all_pairs:
            sections = pair.get("resume_sections", {})
            
            section_ids = []
            section_masks = []
            for sec_name in SECTIONS:
                sec_text = sections.get(sec_name, "")
                enc = tokenizer(
                    sec_text if sec_text else "[EMPTY]",
                    max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt"
                )
                section_ids.append(enc["input_ids"])
                section_masks.append(enc["attention_mask"])
            
            sec_ids = torch.stack([x.squeeze(0) for x in section_ids]).unsqueeze(0).to(DEVICE)
            sec_masks = torch.stack([x.squeeze(0) for x in section_masks]).unsqueeze(0).to(DEVICE)
            
            jd_enc = tokenizer(
                pair["jd_text"], max_length=256,
                padding="max_length", truncation=True, return_tensors="pt"
            )
            jd_ids = jd_enc["input_ids"].to(DEVICE)
            jd_masks = jd_enc["attention_mask"].to(DEVICE)
            
            r_emb, j_emb = model(sec_ids, sec_masks, jd_ids, jd_masks)
            similarity = (r_emb * j_emb).sum(dim=-1).item()
            
            scores.append(similarity)
            labels.append(pair["label"])
    
    return np.array(scores), np.array(labels)


def main():
    print("Collecting validation scores...")
    raw_scores, labels = collect_scores_and_labels(
        model_path=f"{SAVE_DIR}/best_model.pt",
        val_path="data/processed/val_pairs.jsonl"
    )
    
    if len(raw_scores) == 0:
        return
        
    print(f"Validation pairs: {len(raw_scores)}")
    print(f"Score range: {raw_scores.min():.3f} to {raw_scores.max():.3f}")
    print(f"Positive rate: {labels.mean():.1%}")
    
    calibrator = PlattCalibration()
    calibrator.fit(raw_scores, labels)
    calibrator.save("models/calibration.pkl")
    
    # Quick sanity check
    pos_scores = raw_scores[labels == 1]
    if len(pos_scores) > 0:
        test_score = pos_scores.mean()
        calibrated = calibrator.predict_proba(test_score)
        print(f"\nSanity check: avg positive raw score={test_score:.3f} → calibrated={calibrated:.1f}%")
        print("Calibration saved to models/calibration.pkl")


if __name__ == "__main__":
    main()
