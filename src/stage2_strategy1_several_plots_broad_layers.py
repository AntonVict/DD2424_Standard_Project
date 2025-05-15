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


# ====== SETTINGS ======
# Choose strategy: 'simultaneous' or 'gradual'
STRATEGY = 'simultaneous'  # or 'gradual'
# Range of L values to iterate over (first layer to unfreeze, last layer to unfreeze)
L_RANGE = (1, 4)  # Min and max values inclusive
EPOCHS = 1  # For demonstration; increase for real training
BATCH_SIZE = 32

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
        # We'll handle transformations directly in __getitem__ to avoid errors

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        try:
            img_path = os.path.join(self.img_dir, fname)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
                
            # Create a simpler transformation pipeline to avoid numpy errors
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img) / 255.0  # Normalize to [0,1]
            img = img.transpose(2, 0, 1)  # HWC -> CHW format
            img = torch.FloatTensor(img)  # Convert to tensor
            
            # Apply mean/std normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            
            return img, label
        except Exception as e:
            print(f"Error loading image {fname}: {e}")
            # Create a dummy tensor as fallback
            return torch.zeros((3, 224, 224)), label


def set_parameter_requires_grad(model, l, strategy):
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    # Always unfreeze the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
    if strategy == 'simultaneous':
        # Unfreeze last l layers (layer4, layer3, ...)
        layers = [model.layer4, model.layer3, model.layer2, model.layer1]
        # Limit l to the number of available layers
        l = min(l, len(layers))
        for i in range(l):
            for param in layers[i].parameters():  # Access in regular order instead of negative indices
                param.requires_grad = True
            log(f"Unfreezing layer{4-i}")
    elif strategy == 'gradual':
        # Only classifier at first; will unfreeze more in training loop
        pass
    else:
        raise ValueError('Unknown strategy')

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()


def main():
    DATA_DIR = "dataset/oxford-iiit-pet"
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    # Split trainval.txt into train/val indices
    with open(TRAIN_FILE) as f:
        lines = [line for line in f if len(line.strip().split()) >= 4]
    n_total = len(lines)
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.95 * n_total)
    train_indices = indices[:split]
    # Quick-test: use only 10% of training data
    # num_quick = max(1, len(train_indices) // 5)
    # train_indices = train_indices[:num_quick]
    
    # Datasets and loaders
    train_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=train_indices)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Batch size: {BATCH_SIZE}")
    log(f"[Telemetry] Using device: {device}")
    log(f"[Telemetry] Strategy: {STRATEGY}")
    
    # Initialize storage for all L values
    all_iter_steps = {}
    all_iter_train_losses = {}
    all_iter_train_accs = {}
    all_train_losses = {}
    all_train_accs = {}
    
    # Run for each L value within the specified range
    for L in range(L_RANGE[0], L_RANGE[1] + 1):
        log(f"\n[Telemetry] Training with L={L} (unfreezing last {L} layers + classifier)")
        
        # Model
        resnet34 = models.resnet34(weights="IMAGENET1K_V1")
        num_ftrs = resnet34.fc.in_features
        resnet34.fc = nn.Linear(num_ftrs, 37)  # 37 breeds
        resnet34 = resnet34.to(device)

        # Set requires_grad according to strategy
        set_parameter_requires_grad(resnet34, L, STRATEGY)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4)

        # For plotting
        train_losses, train_accs = [], []
        iter_steps, iter_train_losses, iter_train_accs = [], [], []

        for epoch in range(EPOCHS):
            log(f"\n[Telemetry] L={L}, Starting epoch {epoch+1}/{EPOCHS}")
            resnet34.train()
            train_loss, train_acc, n = 0, 0, 0
            for batch_idx, (imgs, labels) in enumerate(tqdm(train_dl, total=len(train_dl))):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = resnet34(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                train_acc += accuracy(outputs, labels) * imgs.size(0)
                n += imgs.size(0)
                
                # Track training loss and accuracy every 10 iterations
                if (batch_idx) % 5 == 0:
                    curr_train_loss = loss.item()
                    curr_train_acc = accuracy(outputs, labels)
                    iter_train_losses.append(curr_train_loss)
                    iter_train_accs.append(curr_train_acc)
                    iter_steps.append(epoch * len(train_dl) + batch_idx + 1)
                    log(f"[Telemetry][L={L}][Epoch {epoch+1}][Iter {batch_idx+1}] Train Loss: {curr_train_loss:.4f} | Train Acc: {curr_train_acc:.4f}")
                
                # Log batch progress
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_dl):
                    log(f"[Telemetry][L={L}][Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Loss: {loss.item():.4f} | Acc: {accuracy(outputs, labels):.4f}")
                
            train_losses.append(train_loss/n)
            train_accs.append(train_acc/n)
            log(f"[Telemetry][L={L}] Epoch {epoch+1} | Train Loss: {train_loss/n:.4f} | Train Acc: {train_acc/n:.4f}")

        # Store results for this L value
        all_iter_steps[L] = iter_steps.copy()
        all_iter_train_losses[L] = iter_train_losses.copy()
        all_iter_train_accs[L] = iter_train_accs.copy()
        all_train_losses[L] = train_losses.copy()
        all_train_accs[L] = train_accs.copy()

    # Create the combined plot for training losses
    plt.figure(figsize=(10, 6))
    num_l_values = L_RANGE[1] - L_RANGE[0] + 1
    colors = plt.cm.viridis(np.linspace(0, 1, num_l_values))
    
    for i, L in enumerate(range(L_RANGE[0], L_RANGE[1] + 1)):
        plt.plot(all_iter_steps[L], all_iter_train_losses[L], 
                 label=f'L={L}',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Iteration for Different Layer Configurations')
    plt.legend()
    loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-l-comparison-loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Create the combined plot for training accuracies
    plt.figure(figsize=(10, 6))
    
    for i, L in enumerate(range(L_RANGE[0], L_RANGE[1] + 1)):
        plt.plot(all_iter_steps[L], all_iter_train_accs[L], 
                 label=f'L={L}',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy per Iteration for Different Layer Configurations')
    plt.legend()
    acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-l-comparison-acc.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 2 Multi-class Classification L Comparison Run ({pretty_timestamp})\n")
        f.write(f"### Training Loss\n")
        f.write(f"![](./{os.path.relpath(loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training Accuracy\n")
        f.write(f"![](./{os.path.relpath(acc_plot_path, os.getcwd())})\n\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main() 