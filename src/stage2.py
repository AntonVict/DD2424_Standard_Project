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

# ====== SETTINGS ======
# Choose strategy: 'simultaneous' or 'gradual'
STRATEGY = 'simultaneous'  # or 'gradual'
L = 2  # Number of last layers to fine-tune in 'simultaneous' mode
EPOCHS = 15  # For demonstration; increase for real training
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
        self.tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.tfms(img), label


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
        for i in range(l):
            for param in layers[-(i+1)].parameters():
                param.requires_grad = True
    elif strategy == 'gradual':
        # Only classifier at first; will unfreeze more in training loop
        pass
    else:
        raise ValueError('Unknown strategy')

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()


def main():
    # Paths
    DATA_DIR = "dataset/oxford-iiit-pet"
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    # Split trainval.txt into train/val indices
    with open(TRAIN_FILE) as f:
        lines = [line for line in f if len(line.strip().split()) >= 4]
    n_total = len(lines)
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * n_total)
    train_indices = indices[:split]
    val_indices = indices[split:]
    # Datasets and loaders
    train_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=train_indices)
    val_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=val_indices)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Number of validation samples: {len(val_ds)}")
    log(f"[Telemetry] Batch size: {BATCH_SIZE}")
    log(f"[Telemetry] Using device: {device}")
    log(f"[Telemetry] Strategy: {STRATEGY}")
    if STRATEGY == 'simultaneous':
        log(f"[Telemetry] Fine-tuning last {L} layers + classifier from start.")
    else:
        log(f"[Telemetry] Gradual unfreezing: starting with classifier only.")

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
    val_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        log(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
        resnet34.train()
        train_loss, train_acc, n = 0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet34(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_acc += accuracy(outputs, labels) * imgs.size(0)
            n += imgs.size(0)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_dl):
                log(f"[Telemetry][Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Loss: {loss.item():.4f} | Acc: {accuracy(outputs, labels):.4f}")
        train_losses.append(train_loss/n)
        train_accs.append(train_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Train Loss: {train_loss/n:.4f} | Train Acc: {train_acc/n:.4f}")

        # Gradual unfreezing: unfreeze one more layer each epoch
        if STRATEGY == 'gradual' and epoch+1 <= 4:
            # Unfreeze one more layer (layer4, then layer3, ...)
            layers = [resnet34.layer4, resnet34.layer3, resnet34.layer2, resnet34.layer1]
            for param in layers[-epoch-1].parameters():
                param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4)
            log(f"[Telemetry] Gradual: Unfroze layer {4-epoch}")

        resnet34.eval()
        val_loss, val_acc, n = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(outputs, labels) * imgs.size(0)
                n += imgs.size(0)
        val_losses.append(val_loss/n)
        val_accs.append(val_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Val Loss: {val_loss/n:.4f} | Val Acc: {val_acc/n:.4f}")

    # Plotting
    epochs = list(range(1, EPOCHS+1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-acc.png")
    plt.savefig(acc_plot_path)
    plt.close()

    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 2 Multi-class Classification Run ({pretty_timestamp})\n")
        f.write(f"![]({loss_plot_path})\n")
        f.write(f"![]({acc_plot_path})\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main() 