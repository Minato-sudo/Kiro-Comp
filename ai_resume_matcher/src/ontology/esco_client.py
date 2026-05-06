"""Static ESCO skill lookups for skill normalization."""

import pandas as pd
from functools import lru_cache
from pathlib import Path

class ESCOClient:
    def __init__(self, csv_path: str = "data/esco/skills_en.csv"):
        if not Path(csv_path).exists():
            print(f"Warning: ESCO CSV not found at {csv_path}. Using empty fallback.")
            self.label_to_uri = {}
            self.uri_to_label = {}
            self.all_labels = []
            return
        
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        
        # Build lookup: preferred label → URI
        label_col = "preferredLabel" if "preferredLabel" in df.columns else df.columns[1]
        uri_col = "conceptUri" if "conceptUri" in df.columns else df.columns[0]
        
        self.label_to_uri = {}
        self.uri_to_label = {}
        self.all_labels = []
        
        for _, row in df.iterrows():
            label = str(row.get(label_col, "")).lower().strip()
            uri = str(row.get(uri_col, ""))
            if label and uri:
                self.label_to_uri[label] = uri
                self.uri_to_label[uri] = label
                self.all_labels.append(label)
        
        print(f"ESCO loaded: {len(self.all_labels)} skills")
    
    @lru_cache(maxsize=10000)
    def normalize(self, raw_skill: str) -> str:
        """Map a raw skill string to its ESCO canonical label."""
        raw_lower = raw_skill.lower().strip()
        
        # Exact match
        if raw_lower in self.label_to_uri:
            return raw_lower
        
        # Partial match (find longest ESCO label contained in skill)
        matches = [lbl for lbl in self.all_labels if lbl in raw_lower or raw_lower in lbl]
        if matches:
            return max(matches, key=len)
        
        # No match — return original
        return raw_lower
    
    def is_known(self, skill: str) -> bool:
        return self.normalize(skill) in self.label_to_uri
