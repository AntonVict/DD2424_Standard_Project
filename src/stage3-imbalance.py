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
import random
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# ====== SETTINGS ======
EPOCHS = 3  # For demonstration; increase for real training
BATCH_SIZE = 32
CAT_BREED_IDS = set(range(1, 26))  # 1-25 are cat breeds
CAT_KEEP_FRAC = 0.2  # Keep only 20% of cat images

# Logging setup
LOG_DIR = "logs"
PLOT_DIR = os.path.join(LOG_DIR, "plots")
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
pretty_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = os.path.join(LOG_DIR, f"logs-{pretty_timestamp}.log")
RAW_RESULTS = "raw-results.md"
PLOT_PREFIX = f"stage3-imbalance-{pretty_timestamp}"

def log(msg):
    msg = str(msg)
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# Custom Dataset for multi-class classification (breed recognition, with imbalance)
class PetBreedImbalanceDataset(Dataset):
    def __init__(self, data_dir, split_file, cat_keep_frac=1.0):
        self.img_dir = os.path.join(data_dir, "images")
        self.samples = []  # list of (img_path, label, breed_id, species)
        breed_to_samples = defaultdict(list)
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name, classid, species, breedid = parts
                label = int(classid) - 1  # 0-36
                breedid = int(breedid)
                species = int(species)
                breed_to_samples[(label, breedid, species)].append((f"{name}.jpg", label, breedid, species))
        # For cat breeds, keep only a fraction
        for (label, breedid, species), samples in breed_to_samples.items():
            if species == 1:  # Cat
                k = max(1, int(len(samples) * cat_keep_frac))
                self.samples.extend(random.sample(samples, k))
            else:
                self.samples.extend(samples)
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
        fname, label, breedid, species = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.tfms(img), label, breedid, species

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item(), preds

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    # Paths
    DATA_DIR = "dataset/oxford-iiit-pet"
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    TEST_FILE = os.path.join(DATA_DIR, "annotations", "test.txt")

    # Datasets and loaders
    train_ds = PetBreedImbalanceDataset(DATA_DIR, TRAIN_FILE, cat_keep_frac=CAT_KEEP_FRAC)
    test_ds = PetBreedImbalanceDataset(DATA_DIR, TEST_FILE, cat_keep_frac=1.0)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print class distribution
    train_labels = [label for _, label, _, _ in train_ds.samples]
    label_counts = Counter(train_labels)
    log(f"[Telemetry] Training class distribution (label: count): {sorted(label_counts.items())}")
    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Number of test samples: {len(test_ds)}")
    log(f"[Telemetry] Batch size: {BATCH_SIZE}")
    log(f"[Telemetry] Using device: {device}")
    log(f"[Telemetry] Cat keep fraction: {CAT_KEEP_FRAC}")

    # Model
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_ftrs, 37)  # 37 breeds
    resnet34 = resnet34.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet34.parameters(), lr=1e-4)

    # For plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        log(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
        resnet34.train()
        train_loss, train_acc, n = 0, 0, 0
        for batch_idx, (imgs, labels, _, _) in enumerate(train_dl):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet34(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc, _ = accuracy(outputs, labels)
            train_loss += loss.item() * imgs.size(0)
            train_acc += acc * imgs.size(0)
            n += imgs.size(0)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_dl):
                log(f"[Telemetry][Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
        train_losses.append(train_loss/n)
        train_accs.append(train_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Train Loss: {train_loss/n:.4f} | Train Acc: {train_acc/n:.4f}")

        resnet34.eval()
        val_loss, val_acc, n = 0, 0, 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels, breedids, species in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs)
                loss = criterion(outputs, labels)
                acc, preds = accuracy(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += acc * imgs.size(0)
                n += imgs.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
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

    # Per-class accuracy
    per_class_correct = Counter()
    per_class_total = Counter()
    for label, pred in zip(all_labels, all_preds):
        per_class_total[label] += 1
        if label == pred:
            per_class_correct[label] += 1
    log("[Telemetry] Per-class accuracy:")
    for label in range(37):
        total = per_class_total[label]
        correct = per_class_correct[label]
        acc = correct / total if total > 0 else 0.0
        log(f"  Class {label+1:2d}: {acc:.4f} ({correct}/{total})")

    # Confusion matrix
    cm = np.zeros((37, 37), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1
    confmat_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-confmat.png")
    plot_confusion_matrix(cm, [str(i+1) for i in range(37)], confmat_path)
    log(f"[Telemetry] Confusion matrix saved to: {confmat_path}")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"stage3-imbalance-{pretty_timestamp}.pt")
    torch.save(resnet34.state_dict(), model_path)
    log(f"[Telemetry] Model saved to: {model_path}")

    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 3 Imbalanced Classification Run ({pretty_timestamp})\n")
        f.write(f"![]({loss_plot_path})\n")
        f.write(f"![]({acc_plot_path})\n")
        f.write(f"![]({confmat_path})\n")
        f.write(f"\n**Model saved at:** `{model_path}`\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main() 