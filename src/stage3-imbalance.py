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
    def __init__(self, data_dir, split_file, cat_keep_frac=1.0, indices=None, build_full=False):
        self.img_dir = os.path.join(data_dir, "images")
        self.samples = []  # list of (img_path, label, breed_id, species)
        self.label_to_info = {} # Map label (0-36) to (breed_id, species, breed_name)
        breed_to_samples = defaultdict(list)
        # Read breed information and samples
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                name, classid_str, species_str, breedid_str = parts
                label = int(classid_str) - 1  # 0-36
                breedid = int(breedid_str)
                species = int(species_str) # 1 for cat, 2 for dog
                # Infer breed name (simple approach: use the filename prefix)
                breed_name = "_".join(name.split('_')[:-1])
                self.label_to_info[label] = (breedid, species, breed_name)
                breed_to_samples[(label, breedid, species)].append((f"{name}.jpg", label, breedid, species))

        # For cat breeds, keep only a fraction
        for (label, breedid, species), samples in breed_to_samples.items():
            if species == 1:  # Cat
                k = max(1, int(len(samples) * cat_keep_frac))
                self.samples.extend(random.sample(samples, k))
            else:
                self.samples.extend(samples)

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
        fname, label, breedid, species = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        return self.tfms(img), label, breedid, species

def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item(), preds

def plot_confusion_matrix(cm, classes, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(12, 10)) # Increased figure size
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8) # Increased font size for species matrix
    plt.yticks(tick_marks, classes, fontsize=8) # Increased font size for species matrix
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_class_distribution(label_counts, label_to_info, save_path):
    labels = sorted(label_counts.keys())
    counts = [label_counts[label] for label in labels]
    # Create descriptive labels for the plot
    class_names = [f"{label+1}: {label_to_info[label][2]} ({'Cat' if label_to_info[label][1] == 1 else 'Dog'})" for label in labels]

    plt.figure(figsize=(15, 7)) # Increased figure size
    plt.bar(class_names, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Training Class Distribution")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def main():
    # Paths
    DATA_DIR = "dataset/oxford-iiit-pet" # Assuming dataset is in a 'dataset' subdirectory
    TRAIN_FILE = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    TEST_FILE = os.path.join(DATA_DIR, "annotations", "test.txt")

    # Build the full filtered sample list first to get consistent train/val split
    full_train_ds = PetBreedImbalanceDataset(DATA_DIR, TRAIN_FILE, cat_keep_frac=CAT_KEEP_FRAC)
    n_total = len(full_train_ds)
    indices = list(range(n_total))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.8 * n_total)
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Datasets and loaders
    train_ds = PetBreedImbalanceDataset(DATA_DIR, TRAIN_FILE, cat_keep_frac=CAT_KEEP_FRAC, indices=train_indices)
    val_ds = PetBreedImbalanceDataset(DATA_DIR, TRAIN_FILE, cat_keep_frac=CAT_KEEP_FRAC, indices=val_indices)
    test_ds = PetBreedImbalanceDataset(DATA_DIR, TEST_FILE, cat_keep_frac=1.0) # Use full test set

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print and plot class distribution
    train_labels = [label for _, label, _, _ in train_ds.samples]
    label_counts = Counter(train_labels)
    log(f"[Telemetry] Training class distribution (label: count): {sorted(label_counts.items())}")
    log(f"[Telemetry] Number of training samples: {len(train_ds)}")
    log(f"[Telemetry] Number of test samples: {len(test_ds)}")
    log(f"[Telemetry] Batch size: {BATCH_SIZE}")
    log(f"[Telemetry] Using device: {device}")
    log(f"[Telemetry] Cat keep fraction: {CAT_KEEP_FRAC}")

    # Plot class distribution
    class_dist_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-class-distribution.png")
    plot_class_distribution(label_counts, train_ds.label_to_info, class_dist_plot_path)
    log(f"[Telemetry] Training class distribution plot saved to: {class_dist_plot_path}")

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
    test_losses, test_accs = [], []

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

        # Validation
        resnet34.eval()
        val_loss, val_acc, n = 0, 0, 0
        with torch.no_grad():
            for imgs, labels, breedids, species in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs)
                loss = criterion(outputs, labels)
                acc, _ = accuracy(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += acc * imgs.size(0)
                n += imgs.size(0)
        val_losses.append(val_loss/n)
        val_accs.append(val_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Val Loss: {val_loss/n:.4f} | Val Acc: {val_acc/n:.4f}")
        # Test set evaluation (per epoch)
        test_loss, test_acc, n = 0, 0, 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels, breedids, species in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs)
                loss = criterion(outputs, labels)
                acc, preds = accuracy(outputs, labels)
                test_loss += loss.item() * imgs.size(0)
                test_acc += acc * imgs.size(0)
                n += imgs.size(0)
                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
        test_losses.append(test_loss/n)
        test_accs.append(test_acc/n)
        log(f"[Telemetry] Epoch {epoch+1} | Test Loss: {test_loss/n:.4f} | Test Acc: {test_acc/n:.4f}")

    # Plotting
    epochs = list(range(1, EPOCHS+1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation, and Test Loss')
    plt.legend()
    loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.plot(epochs, test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training, Validation, and Test Accuracy')
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
    # Create descriptive labels for per-class accuracy output
    class_accuracies = {}
    for label in range(37):
        total = per_class_total[label]
        correct = per_class_correct[label]
        acc = correct / total if total > 0 else 0.0
        class_name = f"{label+1}: {test_ds.label_to_info[label][2]} ({'Cat' if test_ds.label_to_info[label][1] == 1 else 'Dog'})"
        log(f"  {class_name}: {acc:.4f} ({correct}/{total})")
        class_accuracies[class_name] = acc

    # Plot per-class accuracy
    plt.figure(figsize=(15, 7))
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    plt.bar(classes, accuracies)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy on Test Set")
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    per_class_acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-per-class-accuracy.png")
    plt.savefig(per_class_acc_plot_path)
    plt.close()
    log(f"[Telemetry] Per-class accuracy plot saved to: {per_class_acc_plot_path}")


    # Confusion matrix (Breed Level)
    cm = np.zeros((37, 37), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    # Get sorted list of labels (cats first, then dogs) for breed confusion matrix
    sorted_labels = sorted(test_ds.label_to_info.keys(), key=lambda x: (test_ds.label_to_info[x][1], x))

    # Create a mapping from original label index to sorted label index
    original_to_sorted_index = {original_label: sorted_index for sorted_index, original_label in enumerate(sorted_labels)}

    # Create a new, sorted breed-level confusion matrix
    sorted_cm = np.zeros((37, 37), dtype=int)
    for i in range(37):
        for j in range(37):
            original_true_label = i
            original_predicted_label = j
            sorted_true_index = original_to_sorted_index[original_true_label]
            sorted_predicted_index = original_to_sorted_index[original_predicted_label]
            sorted_cm[sorted_true_index, sorted_predicted_index] = cm[original_true_label, original_predicted_label]


    # Create descriptive labels for breed confusion matrix axes based on the sorted order
    confusion_matrix_class_names_sorted = [f"{label+1}: {test_ds.label_to_info[label][2]} ({'Cat' if test_ds.label_to_info[label][1] == 1 else 'Dog'})" for label in sorted_labels]

    confmat_breed_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-confmat_breed_sorted.png")
    plot_confusion_matrix(sorted_cm, confusion_matrix_class_names_sorted, confmat_breed_path, title='Sorted Breed Confusion Matrix')
    log(f"[Telemetry] Sorted breed confusion matrix saved to: {confmat_breed_path}")


    # Confusion matrix (Species Level)
    # Species 1 is Cat, Species 2 is Dog. Map labels to 0 for Cat, 1 for Dog for a 2x2 matrix.
    species_cm = np.zeros((2, 2), dtype=int) # [[True Cat, Pred Cat], [True Cat, Pred Dog], [True Dog, Pred Cat], [True Dog, Pred Dog]]

    species_map = {1: 0, 2: 1} # Map original species ID to 0-indexed for matrix

    cat_total = 0
    dog_total = 0
    cat_correct_species = 0
    dog_correct_species = 0

    for true_label, pred_label in zip(all_labels, all_preds):
        true_species = test_ds.label_to_info[true_label][1]
        pred_species = test_ds.label_to_info[pred_label][1]

        species_cm[species_map[true_species], species_map[pred_species]] += 1

        if true_species == 1: # True Cat
            cat_total += 1
            if pred_species == 1: # Predicted Cat
                cat_correct_species += 1
        elif true_species == 2: # True Dog
            dog_total += 1
            if pred_species == 2: # Predicted Dog
                dog_correct_species += 1

    # Calculate species accuracy
    cat_species_accuracy = cat_correct_species / cat_total if cat_total > 0 else 0.0
    dog_species_accuracy = dog_correct_species / dog_total if dog_total > 0 else 0.0

    log(f"[Telemetry] Cat Species Accuracy: {cat_species_accuracy:.4f} ({cat_correct_species}/{cat_total})")
    log(f"[Telemetry] Dog Species Accuracy: {dog_species_accuracy:.4f} ({dog_correct_species}/{dog_total})")


    # Plot species confusion matrix
    species_classes = ['Cat', 'Dog']
    confmat_species_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-confmat_species.png")
    plot_confusion_matrix(species_cm, species_classes, confmat_species_path, title='Species Confusion Matrix')
    log(f"[Telemetry] Species confusion matrix saved to: {confmat_species_path}")


    # Save model
    model_path = os.path.join(MODEL_DIR, f"stage3-imbalance-{pretty_timestamp}.pt")
    torch.save(resnet34.state_dict(), model_path)
    log(f"[Telemetry] Model saved to: {model_path}")

    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 3 Imbalanced Classification Run ({pretty_timestamp})\n")
        f.write(f"### Training Class Distribution\n")
        f.write(f"![]({class_dist_plot_path})\n")
        f.write(f"### Training, Validation, and Test Loss\n")
        f.write(f"![]({loss_plot_path})\n")
        f.write(f"### Training, Validation, and Test Accuracy\n")
        f.write(f"![]({acc_plot_path})\n")
        f.write(f"### Per-Class Accuracy on Test Set\n")
        f.write(f"![]({per_class_acc_plot_path})\n")
        f.write(f"### Sorted Breed Confusion Matrix\n")
        f.write(f"![]({confmat_breed_path})\n")
        f.write(f"### Species Confusion Matrix\n")
        f.write(f"![]({confmat_species_path})\n")
        f.write(f"\n**Species Accuracy:**\n")
        f.write(f"- Cat Species Accuracy: {cat_species_accuracy:.4f}\n")
        f.write(f"- Dog Species Accuracy: {dog_species_accuracy:.4f}\n")
        f.write(f"\n**Model saved at:** `{model_path}`\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main()