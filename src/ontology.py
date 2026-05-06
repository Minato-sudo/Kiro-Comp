import networkx as nx
import json
from pathlib import Path

class DynamicSkillGraph:
    """
    N2: Dynamic skill graph that extends itself.
    Base ESCO taxonomy is represented as a NetworkX graph.
    New skills extracted from JDs are added dynamically based on co-occurrence.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self._load_base_ontology()
        
    def _load_base_ontology(self):
        """Mock loader for ESCO base ontology."""
        base_skills = [
            ("python", "numpy", 1.0),
            ("python", "pandas", 1.0),
            ("python", "machine learning", 0.8),
            ("machine learning", "deep learning", 0.9),
            ("deep learning", "pytorch", 0.9),
            ("javascript", "react", 0.9),
            ("react", "frontend", 0.8),
            ("frontend", "html", 0.7),
            ("frontend", "css", 0.7),
            ("data analysis", "pandas", 0.8),
            ("data analysis", "data science", 0.7),
        ]
        for skill1, skill2, weight in base_skills:
            self.graph.add_edge(skill1, skill2, weight=weight)
            
    def normalize_skill(self, raw_skill: str) -> str:
        """Simple normalizer to lowercase and strip."""
        return raw_skill.lower().strip()

    def add_co_occurrence(self, skill1: str, skill2: str, weight_increment=0.1):
        """
        Dynamic extension: Add or strengthen edges between skills that co-occur in JDs.
        e.g. ("cursor ai", "vs code")
        """
        s1 = self.normalize_skill(skill1)
        s2 = self.normalize_skill(skill2)
        
        if self.graph.has_edge(s1, s2):
            self.graph[s1][s2]['weight'] = min(1.0, self.graph[s1][s2]['weight'] + weight_increment)
        else:
            self.graph.add_edge(s1, s2, weight=weight_increment)

    def process_jd_skills(self, skills_list):
        """Process a list of skills from a new JD to dynamically update graph."""
        # Add edges between all pairs of skills in the same JD
        for i in range(len(skills_list)):
            for j in range(i + 1, len(skills_list)):
                self.add_co_occurrence(skills_list[i], skills_list[j])

    def skill_similarity(self, skill_a: str, skill_b: str) -> float:
        """
        Graph distance in the extended hierarchy.
        Provides partial credit for semantically related skills.
        """
        uri_a = self.normalize_skill(skill_a)
        uri_b = self.normalize_skill(skill_b)
        
        if uri_a not in self.graph or uri_b not in self.graph:
            return 0.0 # Unknown skill
            
        if uri_a == uri_b:
            return 1.0
            
        try:
            # Shortest path using weight as distance (invert weight so high weight = short distance)
            def weight_func(u, v, d):
                return 1.0 - d.get('weight', 0.1)
                
            path_length = nx.shortest_path_length(self.graph, uri_a, uri_b, weight=weight_func)
            return max(0.0, 1.0 / (1.0 + path_length))
        except nx.NetworkXNoPath:
            return 0.0

# Singleton instance
skill_graph = DynamicSkillGraph()
