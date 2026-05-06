"""
N1: Asymmetric Cross-Attention
JD requirement sentences attend OVER resume section embeddings to find evidence.
This is the key architectural novelty over ConFit and all SBERT baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AsymmetricCrossAttention(nn.Module):
    """
    Each JD requirement sentence queries the resume sections to find evidence.
    
    Unlike self-attention (where both sides are the same document) or 
    global cosine similarity (one vector per document), this module lets
    individual JD requirements find their best matching resume evidence.
    """
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # JD requirements as queries
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Resume sections as keys and values
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, jd_embeddings: torch.Tensor,
                resume_section_embeddings: torch.Tensor,
                return_attention: bool = False):
        """
        Args:
            jd_embeddings:             (batch, num_jd_sentences, embed_dim)
            resume_section_embeddings: (batch, num_sections, embed_dim)  
        """
        batch_size, num_jd, _ = jd_embeddings.shape
        num_sections = resume_section_embeddings.shape[1]
        
        # Project
        Q = self.q_proj(jd_embeddings)              
        K = self.k_proj(resume_section_embeddings)  
        V = self.v_proj(resume_section_embeddings)  
        
        # Reshape
        def reshape_for_heads(x, seq_len):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim)\
                    .transpose(1, 2) 
        
        Q = reshape_for_heads(Q, num_jd)
        K = reshape_for_heads(K, num_sections)
        V = reshape_for_heads(V, num_sections)
        
        # Attention
        scale = self.head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        
        # Merge
        attended = attended.transpose(1, 2).contiguous()\
                           .view(batch_size, num_jd, self.embed_dim)
        
        output = self.out_proj(attended)
        output = self.layer_norm(output + jd_embeddings)  
        
        if return_attention:
            avg_attention = attention_weights.mean(dim=1)
            return output, avg_attention
        
        return output
    
    def get_requirement_evidence_map(self, jd_requirement_texts: list[str],
                                     resume_section_names: list[str],
                                     attention_weights: torch.Tensor) -> dict:
        weights = attention_weights[0].cpu().numpy()  
        
        evidence_map = {}
        for jd_idx, req_text in enumerate(jd_requirement_texts):
            section_scores = {}
            for sec_idx, sec_name in enumerate(resume_section_names):
                if sec_idx < weights.shape[1]:
                    section_scores[sec_name] = float(weights[jd_idx, sec_idx])
            
            best_section = max(section_scores, key=section_scores.get)
            evidence_map[req_text] = {
                "best_match_section": best_section,
                "section_scores": section_scores,
            }
        
        return evidence_map
