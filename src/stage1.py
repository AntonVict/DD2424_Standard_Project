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

# Logging setup
LOG_DIR = "logs"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
pretty_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"logs-{pretty_timestamp}.log")

# For raw results
RAW_RESULTS = "raw-results.md"

PLOT_PREFIX = f"stage1-binary-{pretty_timestamp}"

def log(msg):
    msg = str(msg)
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# Custom Dataset for binary classification (cat vs. dog)
class PetBinaryDataset(Dataset):
    def __init__(self, data_dir, split_file, indices=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.samples = []  # list of (img_path, label)
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name, _, species, _ = parts
                label = int(species) - 1  # 0=cat, 1=dog
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
        return self.tfms(img), torch.tensor(label, dtype=torch.float32)


def main():
    # Paths
    DATA_DIR = "dataset/oxford-iiit-pet"
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    TEST_FILE = os.path.join(DATA_DIR, "annotations", "test.txt")

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
    train_ds = PetBinaryDataset(DATA_DIR, TRAIN_FILE, indices=train_indices)
    val_ds = PetBinaryDataset(DATA_DIR, TRAIN_FILE, indices=val_indices)
    test_ds = PetBinaryDataset(DATA_DIR, TEST_FILE)
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Telemetry: Data and device info
    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Number of validation samples: {len(val_ds)}")
    log(f"[Telemetry] Number of test samples: {len(test_ds)}")
    log(f"[Telemetry] Batch size: {batch_size}")
    log(f"[Telemetry] Using device: {device}")

    # Model
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_ftrs, 1)  # Binary output
    resnet34 = resnet34.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(resnet34.parameters(), lr=1e-4)

    def accuracy(outputs, labels):
        preds = (torch.sigmoid(outputs) > 0.5).float()
        return (preds.squeeze() == labels).float().mean().item()

    # For plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    EPOCHS = 15  # Increase for real training
    for epoch in range(EPOCHS):
        log(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
        resnet34.train()
        train_loss, train_acc, n = 0, 0, 0
        for batch_idx, (imgs, labels) in enumerate(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet34(imgs).squeeze(1)
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

        resnet34.eval()
        val_loss, val_acc, n = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(outputs, labels) * imgs.size(0)
                n += imgs.size(0)
        val_losses.append(val_loss/n)
        val_accs.append(val_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Val Loss: {val_loss/n:.4f} | Val Acc: {val_acc/n:.4f}")

    # Test set evaluation
    test_loss, test_acc, n = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = resnet34(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            test_acc += accuracy(outputs, labels) * imgs.size(0)
            n += imgs.size(0)
    log(f"[Telemetry] Test Loss: {test_loss/n:.4f} | Test Acc: {test_acc/n:.4f}")

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
        f.write(f"\n\n## Stage 1 Binary Classification Run ({pretty_timestamp})\n")
        f.write(f"![]({loss_plot_path})\n")
        f.write(f"![]({acc_plot_path})\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main()