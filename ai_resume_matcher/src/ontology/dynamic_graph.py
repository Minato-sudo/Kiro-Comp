"""
N2: Dynamic Skill Graph
Extends ESCO with new skill relationships learned from job posting co-occurrences.
"""

import json
import networkx as nx
from collections import defaultdict
from pathlib import Path
import pickle
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.getcwd())
from src.ontology.esco_client import ESCOClient


class DynamicSkillGraph:
    """
    Builds a skill similarity graph combining:
    1. ESCO taxonomy
    2. Co-occurrence edges
    """
    
    def __init__(self, esco_client: ESCOClient = None):
        self.graph = nx.Graph()
        self.esco = esco_client or ESCOClient()
        self.cooccurrence_counts = defaultdict(int)
        self._load_esco_base_graph()
    
    def _load_esco_base_graph(self):
        # Add all ESCO skills as nodes
        for label in self.esco.all_labels:  
            self.graph.add_node(label, source="esco")
        
        print(f"Base graph: {self.graph.number_of_nodes()} skill nodes")
    
    def update_from_jd_corpus(self, jd_pairs_path: str):
        """
        Learn skill co-occurrence from training JDs.
        """
        if not os.path.exists(jd_pairs_path):
            print(f"Warning: {jd_pairs_path} not found. Skipping co-occurrence update.")
            return

        with open(jd_pairs_path) as f:
            for line in f:
                pair = json.loads(line)
                skills = pair.get("jd_skills", [])
                normalized = [self.esco.normalize(s) for s in skills]
                
                for i in range(len(normalized)):
                    for j in range(i+1, len(normalized)):
                        s1, s2 = normalized[i], normalized[j]
                        if s1 and s2 and s1 != s2:
                            self.cooccurrence_counts[(s1, s2)] += 1
                            
                            count = self.cooccurrence_counts[(s1, s2)]
                            if count >= 1:  # lower threshold for demo
                                weight = min(1.0, count / 5.0)
                                if self.graph.has_edge(s1, s2):
                                    self.graph[s1][s2]["weight"] = weight
                                else:
                                    self.graph.add_edge(s1, s2, weight=weight, source="cooccurrence")
        
        print(f"Graph updated: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
    
    def skill_similarity(self, skill_a: str, skill_b: str) -> float:
        a = self.esco.normalize(skill_a)
        b = self.esco.normalize(skill_b)
        
        if a == b:
            return 1.0
        
        if not self.graph.has_node(a) or not self.graph.has_node(b):
            from difflib import SequenceMatcher
            return SequenceMatcher(None, a, b).ratio() * 0.5
        
        try:
            path_length = nx.shortest_path_length(self.graph, a, b)
            return 1.0 / (1.0 + path_length)
        except nx.NetworkXNoPath:
            return 0.0
    
    def get_related_skills(self, skill: str, top_k: int = 5) -> list[tuple[str, float]]:
        normalized = self.esco.normalize(skill)
        if not self.graph.has_node(normalized):
            return []
        
        neighbors = list(self.graph.neighbors(normalized))
        scored = [(n, self.skill_similarity(skill, n)) for n in neighbors]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]
    
    def save(self, path: str = "models/skill_graph.pkl"):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str = "models/skill_graph.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    # Script to build and save the graph
    esco = ESCOClient()
    dsg = DynamicSkillGraph(esco)
    dsg.update_from_jd_corpus("data/processed/train_pairs.jsonl")
    dsg.save()
    print("Skill graph saved.")
