import torch
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import sys
import datetime
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

# Configuration
STRATEGY = "simultaneous"
BLOCK_RANGE = (1, 3)  # Min and max values inclusive (max 3 blocks in ResNet34 layer4)
EPOCHS = 1  # For demonstration; increase for real training
BATCH_SIZE = 32
SUBSET_RATIO = 1

# Logging setup
LOG_DIR = "logs"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
pretty_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"logs-{pretty_timestamp}.log")
RAW_RESULTS = "raw-results.md"
PLOT_PREFIX = f"stage2-multiclass-{pretty_timestamp}"

def log(msg):
    msg = str(msg)
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# Custom Dataset for multi-class classification (breed recognition)
class PetBreedDataset(Dataset):
    def __init__(self, data_dir, split_file, indices=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.samples = []  # list of (img_path, label)
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name, classid, _, _ = parts
                label = int(classid) - 1  # 0-36
                self.samples.append((f"{name}.jpg", label))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        try:
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading image {fname}: {e}")
            return torch.zeros((3, 224, 224)), label

def set_parameter_requires_grad(model, num_blocks, strategy):
    """Set which parameters require gradient computation."""
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.fc.parameters():
        param.requires_grad = True
        
    if strategy == 'simultaneous':
        max_blocks = len(model.layer4)
        num_blocks = min(num_blocks, max_blocks)
        
        blocks_unfrozen = []
        for i in range(num_blocks):
            block_idx = max_blocks - 1 - i
            for param in model.layer4[block_idx].parameters():
                param.requires_grad = True
            blocks_unfrozen.append(block_idx)
            
        log(f"Unfrozen blocks in layer4: {sorted(blocks_unfrozen)}")

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()

def main():
    DATA_DIR = "dataset/oxford-iiit-pet"
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    
    # Load and split data
    with open(TRAIN_FILE) as f:
        lines = [line for line in f if len(line.strip().split()) >= 4]
    
    n_total = len(lines)
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    
    # Take subset of data
    subset_size = int(n_total * SUBSET_RATIO)
    indices = indices[:subset_size]
    
    # Split into train/val
    split = int(0.9 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets and dataloaders
    train_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=train_indices)
    val_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=val_indices)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() 
                         else "cpu")

    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Number of validation samples: {len(val_ds)}")
    log(f"[Telemetry] Batch size: {BATCH_SIZE}")
    log(f"[Telemetry] Using device: {device}")
    log(f"[Telemetry] Strategy: {STRATEGY}")
    log(f"[Telemetry] Using {SUBSET_RATIO*100}% of data")
    
    # Initialize model
    model = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)  # 37 breeds
    model = model.to(device)
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    set_parameter_requires_grad(model, BLOCK_RANGE[1], STRATEGY)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    # Training metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    log("\n[Telemetry] Starting training")
    
    for epoch in range(EPOCHS):
        log(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_dl)):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_acc += accuracy(outputs, labels) * imgs.size(0)
            
            # Validate every 50 batches
            if (batch_idx + 1) % 50 == 0:
                model.eval()
                val_loss, val_acc = 0, 0
                with torch.no_grad():
                    for val_imgs, val_labels in val_dl:
                        val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                        val_outputs = model(val_imgs)
                        val_loss += criterion(val_outputs, val_labels).item() * val_imgs.size(0)
                        val_acc += accuracy(val_outputs, val_labels) * val_imgs.size(0)
                
                val_loss /= len(val_ds)
                val_acc /= len(val_ds)
                log(f"Batch {batch_idx+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                model.train()
        
        # Epoch metrics
        train_loss /= len(train_ds)
        train_acc /= len(train_ds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        log(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    
    # Final validation
    model.eval()
    with torch.no_grad():
        val_loss, val_acc = 0, 0
        for val_imgs, val_labels in val_dl:
            val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
            val_outputs = model(val_imgs)
            val_loss += criterion(val_outputs, val_labels).item() * val_imgs.size(0)
            val_acc += accuracy(val_outputs, val_labels) * val_imgs.size(0)
        
        val_loss /= len(val_ds)
        val_acc /= len(val_ds)
    
    log(f"\n[Telemetry] Final Validation | Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-losses.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-accuracies.png"))
    plt.close()
    
    # Save results
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 2 Multi-class Classification ({pretty_timestamp})\n")
        f.write(f"### Training Configuration\n")
        f.write(f"- Strategy: {STRATEGY}\n")
        f.write(f"- Data subset: {SUBSET_RATIO*100}%\n")
        f.write(f"- Epochs: {EPOCHS}\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n\n")
        f.write(f"### Final Metrics\n")
        f.write(f"- Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"- Training Accuracy: {train_accs[-1]:.4f}\n")
        f.write(f"- Validation Loss: {val_loss:.4f}\n")
        f.write(f"- Validation Accuracy: {val_acc:.4f}\n\n")
        f.write(f"### Training Log\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main()