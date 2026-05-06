import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.pipeline import ResumeParser
from src.model import DualEncoderMatcher, contrastive_loss
import random
from tqdm import tqdm

class ResumeJDDataset(Dataset):
    def __init__(self, resume_paths):
        self.resume_paths = resume_paths
        self.parser = ResumeParser()
        self.data = []
        
        print(f"Loading and parsing {len(resume_paths)} resumes...")
        for path in tqdm(resume_paths):
            try:
                text = self.parser.extract_text(path)
                sections = self.parser.segment_sections(text)
                section_list = [
                    sections["skills"],
                    sections["experience"],
                    sections["education"],
                    sections["projects"],
                    sections["summary"]
                ]
                if len(text) > 200:
                    self.data.append(section_list)
            except Exception as e:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        resume_sections = self.data[idx]
        jd_proxy = [resume_sections[0], resume_sections[4]]
        return resume_sections, jd_proxy

def train_model(dataset_path="DataSet", epochs=5, batch_size=4, checkpoint_path="models/fine_tuned_matcher.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare Data
    all_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                 if f.endswith(('.pdf', '.docx'))]
    random.shuffle(all_files)
    
    split = int(0.8 * len(all_files))
    train_files = all_files[:split]
    
    train_ds = ResumeJDDataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: zip(*x))

    # 2. Initialize Model and Load Checkpoint
    model = DualEncoderMatcher().to(device)
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Lower LR for fine-tuning

    # 3. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for resume_batch, jd_batch in pbar:
            optimizer.zero_grad()
            
            res_proj, jd_proj, _ = model(list(resume_batch), list(jd_batch))
            loss = contrastive_loss(res_proj, jd_proj)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

    # 4. Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/fine_tuned_matcher.pth")
    print("✅ Extended training complete. Model saved.")

if __name__ == "__main__":
    train_model(epochs=5, batch_size=8) # Increased batch size for better contrastive signal
