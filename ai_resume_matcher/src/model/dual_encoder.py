"""
Dual Encoder with Section-Aware Attention Pooling.
This is the core architecture. Do not simplify this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional

BERT_MODEL = "bert-base-uncased"
EMBED_DIM = 768
PROJ_DIM = 256
SECTIONS = ["summary", "skills", "experience", "education", "projects"]
NUM_SECTIONS = len(SECTIONS)


class SectionAttentionPooling(nn.Module):
    """
    Instead of using only the CLS token, this module:
    1. Encodes each resume section independently
    2. Learns a weighted combination based on which sections matter most
    3. Weights are learned per job-category during training
    
    This is the architectural novelty over vanilla BERT/SBERT.
    """
    
    def __init__(self, hidden_dim: int = EMBED_DIM):
        super().__init__()
        self.section_weights = nn.Parameter(torch.ones(NUM_SECTIONS) / NUM_SECTIONS)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, section_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            section_embeddings: (batch_size, num_sections, hidden_dim)
        Returns:
            pooled: (batch_size, hidden_dim)
        """
        weights = F.softmax(self.section_weights, dim=0)  # (num_sections,)
        weights = weights.unsqueeze(0).unsqueeze(-1)       # (1, num_sections, 1)
        
        weighted = section_embeddings * weights            # (batch, num_sections, hidden)
        pooled = weighted.sum(dim=1)                       # (batch, hidden)
        return self.layer_norm(pooled)
    
    def get_section_weights(self) -> dict:
        """Returns the learned importance weights per section (for explainability)."""
        weights = F.softmax(self.section_weights, dim=0).detach().cpu().numpy()
        return {section: float(w) for section, w in zip(SECTIONS, weights)}


class ResumeEncoder(nn.Module):
    """Encodes a resume by encoding each section separately then pooling."""
    
    def __init__(self, bert_model_name: str = BERT_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.section_pool = SectionAttentionPooling()
        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, PROJ_DIM),
            nn.GELU(),
            nn.LayerNorm(PROJ_DIM),
        )
        
    def encode_section(self, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a single section. Returns (batch, hidden_dim)."""
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        return summed / count
    
    def forward(self, section_input_ids: torch.Tensor,
                section_attention_masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            section_input_ids:       (batch, num_sections, seq_len)
            section_attention_masks: (batch, num_sections, seq_len)
        Returns:
            embedding: (batch, proj_dim) — L2 normalized
        """
        batch_size = section_input_ids.shape[0]
        section_embs = []
        
        for sec_idx in range(NUM_SECTIONS):
            emb = self.encode_section(
                section_input_ids[:, sec_idx, :],
                section_attention_masks[:, sec_idx, :]
            )
            section_embs.append(emb)
        
        # Stack: (batch, num_sections, hidden_dim)
        stacked = torch.stack(section_embs, dim=1)
        
        # Section-aware pooling
        pooled = self.section_pool(stacked)   # (batch, hidden_dim)
        
        # Project to lower dim
        projected = self.projection(pooled)   # (batch, proj_dim)
        
        # L2 normalize for cosine similarity
        return F.normalize(projected, p=2, dim=-1)


class JDEncoder(nn.Module):
    """
    Encodes job descriptions. Simpler than resume encoder since JDs
    are more uniform in structure.
    """
    
    def __init__(self, bert_model_name: str = BERT_MODEL):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, PROJ_DIM),
            nn.GELU(),
            nn.LayerNorm(PROJ_DIM),
        )
    
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns (batch, proj_dim) L2-normalized."""
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        cls_emb = summed / count
        projected = self.projection(cls_emb)
        return F.normalize(projected, p=2, dim=-1)


class DualEncoderModel(nn.Module):
    """Full dual encoder model combining resume + JD encoders."""
    
    def __init__(self):
        super().__init__()
        self.resume_encoder = ResumeEncoder()
        self.jd_encoder = JDEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    
    def forward(self, resume_section_ids, resume_section_masks,
                jd_ids, jd_masks):
        resume_emb = self.resume_encoder(resume_section_ids, resume_section_masks)
        jd_emb = self.jd_encoder(jd_ids, jd_masks)
        return resume_emb, jd_emb
    
    def get_similarity(self, resume_emb: torch.Tensor, 
                       jd_emb: torch.Tensor) -> torch.Tensor:
        """Returns cosine similarity (already L2 normalized = dot product)."""
        return (resume_emb * jd_emb).sum(dim=-1)
    
    def get_section_weights(self) -> dict:
        return self.resume_encoder.section_pool.get_section_weights()
