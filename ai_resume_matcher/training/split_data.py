# training/split_data.py
import json
import random
import os
from pathlib import Path

def create_splits(input_path: str = "data/processed/augmented_pairs.jsonl",
                  train_ratio: float = 0.80,
                  val_ratio: float = 0.10):
    
    if not os.path.exists(input_path):
        # Fallback to training_pairs.jsonl if augmented doesn't exist yet
        input_path = "data/processed/training_pairs.jsonl"
        if not os.path.exists(input_path):
            print(f"Error: {input_path} not found. Run previous phases first.")
            return
            
    pairs = []
    with open(input_path) as f:
        for line in f:
            pairs.append(json.loads(line))
    
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(pairs)
    
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        "train": pairs[:train_end],
        "val":   pairs[train_end:val_end],
        "test":  pairs[val_end:],
    }
    
    for split_name, split_data in splits.items():
        path = f"data/processed/{split_name}_pairs.jsonl"
        with open(path, "w") as f:
            for p in split_data:
                f.write(json.dumps(p) + "\n")
        print(f"{split_name}: {len(split_data)} pairs → {path}")
    
    return splits

if __name__ == "__main__":
    create_splits()
