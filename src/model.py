import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
import numpy as np

class SectionAttentionPooling(nn.Module):
    """
    Computes a weighted sum of section-level embeddings.
    Weights are learned during training.
    """
    def __init__(self, hidden_size, num_sections=5):
        super().__init__()
        self.section_weights = nn.Parameter(torch.ones(num_sections) / num_sections)
        
    def forward(self, section_embeddings):
        # section_embeddings: (batch, num_sections, hidden_size)
        weights = F.softmax(self.section_weights, dim=0)
        weighted = section_embeddings * weights.unsqueeze(0).unsqueeze(-1)
        return weighted.sum(dim=1)

class AsymmetricCrossAttention(nn.Module):
    """
    N1: Asymmetric cross-attention.
    JD requirement sentences attend over resume sections to find the best supporting evidence.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, jd_sentences_emb, resume_sections_emb):
        # jd_sentences_emb: (batch, num_jd_sentences, hidden_size) - Queries
        # resume_sections_emb: (batch, num_sections, hidden_size) - Keys/Values
        
        Q = self.query_proj(jd_sentences_emb)
        K = self.key_proj(resume_sections_emb)
        V = self.value_proj(resume_sections_emb)

        # Attention scores: (batch, num_jd_sentences, num_sections)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)

        # Context vector for each JD sentence: (batch, num_jd_sentences, hidden_size)
        context = torch.matmul(attn_weights, V)
        
        # Aggregate across JD sentences (e.g., mean pooling)
        return context.mean(dim=1), attn_weights

class DualEncoderMatcher(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        self.section_pooling = SectionAttentionPooling(self.hidden_size)
        self.cross_attention = AsymmetricCrossAttention(self.hidden_size)
        self.projection = nn.Linear(self.hidden_size, 256)
        
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.encoder(**inputs)
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0, :]

    def forward(self, resume_sections, jd_sentences):
        """
        resume_sections: list of lists of strings (batch, num_sections)
        jd_sentences: list of lists of strings (batch, num_jd_sentences)
        """
        batch_size = len(resume_sections)
        num_sections = len(resume_sections[0])
        num_jd_sentences = len(jd_sentences[0])
        
        # Flatten and encode
        flat_resume = [sec for batch in resume_sections for sec in batch]
        flat_jd = [sent for batch in jd_sentences for sent in batch]
        
        resume_emb = self.encode_text(flat_resume).view(batch_size, num_sections, -1)
        jd_emb = self.encode_text(flat_jd).view(batch_size, num_jd_sentences, -1)
        
        # Apply asymmetric cross-attention
        attended_jd_emb, attn_weights = self.cross_attention(jd_emb, resume_emb)
        
        # Apply section pooling on resume
        pooled_resume_emb = self.section_pooling(resume_emb)
        
        # Project to common space
        res_proj = F.normalize(self.projection(pooled_resume_emb), p=2, dim=1)
        jd_proj = F.normalize(self.projection(attended_jd_emb), p=2, dim=1)
        
        return res_proj, jd_proj, attn_weights

class ScoreCalibration:
    """
    N4: Score Calibration using Platt scaling.
    Transforms raw cosine similarity into a probability [0, 1].
    """
    def __init__(self):
        self.calibrator = LogisticRegression()
        self.is_fitted = False
        
    def fit(self, raw_scores, labels):
        # raw_scores: array of cosine similarities [-1, 1]
        # labels: binary match (1) or no-match (0)
        X = np.array(raw_scores).reshape(-1, 1)
        y = np.array(labels)
        self.calibrator.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, raw_score):
        if not self.is_fitted:
            # Fallback if not calibrated
            return max(0.0, min(1.0, (raw_score + 1) / 2))
        X = np.array([[raw_score]])
        return self.calibrator.predict_proba(X)[0, 1]

def contrastive_loss(resume_emb, jd_emb, temperature=0.07):
    """InfoNCE loss for dual encoder."""
    sim_matrix = torch.matmul(resume_emb, jd_emb.T) / temperature
    labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
