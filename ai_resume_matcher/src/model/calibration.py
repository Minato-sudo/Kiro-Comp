"""
N4: Platt Scaling for score calibration.
Converts raw cosine similarity (-1 to 1) into calibrated probability (0 to 1).
"""

import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from pathlib import Path


class PlattCalibration:
    """
    Fits a logistic regression on raw similarity scores.
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0)
        self.is_fitted = False
        self.score_percentiles = None 
    
    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        raw_scores = raw_scores.reshape(-1, 1)
        self.calibrator.fit(raw_scores, labels)
        self.is_fitted = True
        
        self.score_percentiles = np.percentile(raw_scores, 
                                                [10, 25, 50, 75, 90])
        print("Platt calibration fitted successfully.")
        
    def predict_proba(self, raw_score: float) -> float:
        if not self.is_fitted:
            return float((raw_score + 1) * 50) # fallback
        prob = self.calibrator.predict_proba([[raw_score]])[0][1]
        return float(prob * 100)  
    
    def percentile_rank(self, raw_score: float) -> float:
        if self.score_percentiles is None:
            return 50.0
        rank = np.searchsorted(self.score_percentiles, raw_score) * 20.0
        return min(95.0, max(5.0, rank))
    
    def save(self, path: str = "models/calibration.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str = "models/calibration.pkl") -> "PlattCalibration":
        with open(path, "rb") as f:
            return pickle.load(f)
