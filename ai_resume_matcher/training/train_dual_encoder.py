"""
Main training script. Run this after Phases 1-4.
Uses InfoNCE contrastive loss.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.getcwd())

from src.model.dual_encoder import DualEncoderModel, SECTIONS, NUM_SECTIONS

# ── Config ──────────────────────────────────────────────────────────────────
BERT_MODEL   = "bert-base-uncased"
MAX_SEQ_LEN  = 128   # per section
MAX_JD_LEN   = 256
BATCH_SIZE   = 8
EPOCHS       = 3
LR           = 2e-5
TEMPERATURE  = 0.07
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR     = "models/"
# ────────────────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)


class ResumeJDDataset(Dataset):
    def __init__(self, pairs_path: str):
        self.pairs = []
        if not os.path.exists(pairs_path):
            print(f"Error: {pairs_path} not found.")
            return
            
        with open(pairs_path) as f:
            for line in f:
                p = json.loads(line)
                if p["label"] == 1:
                    self.pairs.append(p)
        print(f"Loaded {len(self.pairs)} positive training pairs from {pairs_path}")
    
    def tokenize_section(self, text: str) -> dict:
        return tokenizer(
            text if text else "[EMPTY]",
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        sections = pair.get("resume_sections", {})
        
        section_ids = []
        section_masks = []
        for sec_name in SECTIONS:
            sec_text = sections.get(sec_name, "")
            enc = self.tokenize_section(sec_text)
            section_ids.append(enc["input_ids"].squeeze(0))
            section_masks.append(enc["attention_mask"].squeeze(0))
        
        section_ids_tensor = torch.stack(section_ids)
        section_masks_tensor = torch.stack(section_masks)
        
        jd_enc = tokenizer(
            pair["jd_text"],
            max_length=MAX_JD_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "section_ids":    section_ids_tensor,
            "section_masks":  section_masks_tensor,
            "jd_ids":         jd_enc["input_ids"].squeeze(0),
            "jd_masks":       jd_enc["attention_mask"].squeeze(0),
            "llm_score":      torch.tensor(pair.get("llm_score", 80) / 100.0, dtype=torch.float),
        }


def info_nce_loss(resume_embs: torch.Tensor, jd_embs: torch.Tensor,
                  temperature: float = TEMPERATURE) -> torch.Tensor:
    sim = torch.matmul(resume_embs, jd_embs.T) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss_r2j = F.cross_entropy(sim, labels)
    loss_j2r = F.cross_entropy(sim.T, labels)
    return (loss_r2j + loss_j2r) / 2


def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        optimizer.zero_grad()
        
        r_emb, j_emb = model(
            batch["section_ids"].to(DEVICE),
            batch["section_masks"].to(DEVICE),
            batch["jd_ids"].to(DEVICE),
            batch["jd_masks"].to(DEVICE),
        )
        
        loss = info_nce_loss(r_emb, j_emb)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    all_resume_embs = []
    all_jd_embs = []
    
    with torch.no_grad():
        for batch in loader:
            r_emb, j_emb = model(
                batch["section_ids"].to(DEVICE),
                batch["section_masks"].to(DEVICE),
                batch["jd_ids"].to(DEVICE),
                batch["jd_masks"].to(DEVICE),
            )
            all_resume_embs.append(r_emb.cpu())
            all_jd_embs.append(j_emb.cpu())
    
    if not all_resume_embs:
        return 0.0, 0.0
        
    R = torch.cat(all_resume_embs)
    J = torch.cat(all_jd_embs)
    
    sims = torch.matmul(R, J.T)  
    labels = torch.arange(len(R))
    
    top1 = sims.argmax(dim=1)
    recall_at_1 = (top1 == labels).float().mean().item()
    
    top5 = sims.topk(min(5, len(R)), dim=1).indices
    recall_at_5 = sum(labels[i] in top5[i] for i in range(len(labels))) / len(labels)
    
    return recall_at_1, recall_at_5


def main():
    Path(SAVE_DIR).mkdir(exist_ok=True)
    
    print(f"Device: {DEVICE}")
    print("Loading datasets...")
    
    train_dataset = ResumeJDDataset("data/processed/train_pairs.jsonl")
    val_dataset   = ResumeJDDataset("data/processed/val_pairs.jsonl")
    
    if len(train_dataset) == 0:
        print("No training data. Exiting.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    
    print("Initializing model...")
    model = DualEncoderModel().to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    best_recall = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        r1, r5 = evaluate(model, val_loader)
        
        print(f"Epoch {epoch:2d} | Loss: {train_loss:.4f} | Recall@1: {r1:.3f} | Recall@5: {r5:.3f}")
        
        if r1 > best_recall:
            best_recall = r1
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pt")
            print(f"  ✓ New best model saved (Recall@1={r1:.3f})")
    
    print(f"\nTraining complete. Best Recall@1: {best_recall:.3f}")
    print(f"Model saved to {SAVE_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
