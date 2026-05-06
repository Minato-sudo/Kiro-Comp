import sys
import os
sys.path.insert(0, os.getcwd())
from src.ontology.dynamic_graph import DynamicSkillGraph
from src.ontology.esco_client import ESCOClient

def main():
    esco = ESCOClient()
    dsg = DynamicSkillGraph(esco)
    dsg.update_from_jd_corpus("data/processed/train_pairs.jsonl")
    dsg.save()
    print("Skill graph saved correctly.")

if __name__ == "__main__":
    main()
