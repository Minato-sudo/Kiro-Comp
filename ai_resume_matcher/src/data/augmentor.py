"""
Augments training pairs using:
1. Section dropout: randomly remove 1-2 sections from resume
2. Skill synonym replacement: swap skill names with synonyms
3. Sentence shuffling within sections
"""

import json
import random
import re
from copy import deepcopy

SKILL_SYNONYMS = {
    "python":       ["python3", "py", "python programming"],
    "javascript":   ["js", "ecmascript", "es6"],
    "react":        ["react.js", "reactjs", "react framework"],
    "node.js":      ["nodejs", "node", "server-side javascript"],
    "machine learning": ["ml", "statistical learning", "predictive modeling"],
    "deep learning":    ["dl", "neural networks", "artificial neural networks"],
    "sql":          ["structured query language", "database querying", "relational databases"],
    "git":          ["version control", "github", "gitlab", "source control"],
    "docker":       ["containerization", "container technology"],
    "aws":          ["amazon web services", "cloud computing", "amazon cloud"],
}

def synonym_replace(text: str, probability: float = 0.3) -> str:
    result = text
    for skill, synonyms in SKILL_SYNONYMS.items():
        if random.random() < probability and skill.lower() in result.lower():
            replacement = random.choice(synonyms)
            result = re.sub(re.escape(skill), replacement, result, flags=re.IGNORECASE, count=1)
    return result

def section_dropout(sections: dict, dropout_rate: float = 0.2) -> dict:
    """Randomly remove non-critical sections to simulate incomplete resumes."""
    augmented = deepcopy(sections)
    droppable = ["achievements", "summary", "projects"]
    for section in droppable:
        if random.random() < dropout_rate:
            augmented[section] = ""
    return augmented

def shuffle_sentences(text: str) -> str:
    """Shuffle sentences within a section."""
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 10]
    if len(sentences) < 3:
        return text
    random.shuffle(sentences)
    return ". ".join(sentences) + "."

def augment_pair(pair: dict) -> list[dict]:
    """Generate 1-2 augmented versions of a training pair."""
    augmented_pairs = []
    
    # Augmentation 1: Skill synonym replacement
    aug1 = deepcopy(pair)
    aug1["resume_text"] = synonym_replace(pair["resume_text"])
    aug1["jd_text"] = synonym_replace(pair["jd_text"])
    aug1["augmented"] = True
    aug1["aug_type"] = "synonym"
    augmented_pairs.append(aug1)
    
    # Augmentation 2: Section dropout (only for positive pairs — 
    # we want model to handle incomplete resumes)
    if pair["label"] == 1 and random.random() < 0.5:
        aug2 = deepcopy(pair)
        aug2["resume_sections"] = section_dropout(pair["resume_sections"])
        # Rebuild resume text from augmented sections
        aug2["resume_text"] = " ".join(aug2["resume_sections"].values())
        aug2["augmented"] = True
        aug2["aug_type"] = "dropout"
        augmented_pairs.append(aug2)
    
    return augmented_pairs


def run_phase3(input_path: str = "data/processed/training_pairs.jsonl",
               output_path: str = "data/processed/augmented_pairs.jsonl"):
    import os
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run Phase 2 first.")
        return []
        
    original_pairs = []
    with open(input_path) as f:
        for line in f:
            original_pairs.append(json.loads(line))
    
    all_pairs = list(original_pairs)  # start with originals
    
    for pair in original_pairs:
        augmented = augment_pair(pair)
        all_pairs.extend(augmented)
    
    # Shuffle
    random.shuffle(all_pairs)
    
    with open(output_path, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")
    
    print(f"Original pairs:  {len(original_pairs)}")
    print(f"After augment:   {len(all_pairs)}")
    print(f"Output: {output_path}")
    return all_pairs


if __name__ == "__main__":
    run_phase3()
