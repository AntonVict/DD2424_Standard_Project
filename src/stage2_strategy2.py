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
from torchvision import transforms # == For data augmentation =====

# =========== DIFFERENT LEARNING RATES FOR DIFFERENT LAYERS ===============
# Every time a new block is unfreezed, we rebuild the optimizer
def build_optimizer(model, base_lr=1e-4, strategy="gradual"):
    """Constructs optimizer with different learning rates for each unfrozen block."""
    param_groups = []
    
    # Classifier head (always included)
    param_groups.append({'params': model.fc.parameters(), 'lr': base_lr})
    
    # Gradual unfreezing of layer4 blocks
    if hasattr(model, 'current_blocks'):
        for i in range(model.current_blocks):
            block_idx = len(model.layer4) - 1 - i
            block = model.layer4[block_idx]
            # Assign a smaller LR for earlier blocks
            block_lr = base_lr * (0.5 ** i)
            param_groups.append({'params': block.parameters(), 'lr': block_lr})
    
    return optim.Adam(param_groups, weight_decay=1e-4)  # ← includes L2 regularization!


# what we choose here doenst matter
# STRATEGY = 'gradual'
STRATEGY = "simultaneous"
BLOCK_RANGE = (1, 3)  # Min and max values inclusive (max 3 blocks in ResNet34 layer4)
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
    def __init__(self, data_dir, split_file, indices=None, train=True):
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

        # ====== DATA AUGMENTATION ===========================
        if train:
            self.tfms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
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


def set_parameter_requires_grad(model, num_blocks, strategy):
    """Set which parameters require gradient computation.
    For simultaneous strategy: unfreeze classifier and specified number of blocks in layer4.
    For gradual strategy: see training loop implementation.
    """
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Always unfreeze the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
        
    if strategy == 'simultaneous':
        # Unfreeze the last N blocks of layer4
        max_blocks = len(model.layer4)
        num_blocks = min(num_blocks, max_blocks)
        
        blocks_unfrozen = []
        for i in range(num_blocks):
            block_idx = max_blocks - 1 - i
            for param in model.layer4[block_idx].parameters():
                param.requires_grad = True
            blocks_unfrozen.append(block_idx)
            
        log(f"Unfrozen blocks in layer4: {sorted(blocks_unfrozen)}")
    else:
        # For gradual strategy, handled in the training loop
        pass

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
    split = int(0.9 * n_total)
    train_indices = indices[:split]
    val_indices = indices[split:]
    # Quick-test: use only 10% of training data
    # num_quick = max(1, len(train_indices) // 5)
    # train_indices = train_indices[:num_quick]
    
    # Datasets and loaders
    train_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=train_indices)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Validation dataset and loaders
    val_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=val_indices)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # Create a separate validation loader for iteration tracking with shuffling enabled
    val_iter_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

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
    
    # Initialize the model
    resnet34 = models.resnet34(weights="IMAGENET1K_V1")
    num_ftrs = resnet34.fc.in_features
    resnet34.fc = nn.Linear(num_ftrs, 37)  # 37 breeds
    resnet34 = resnet34.to(device)
    
    # Initialize storage for metrics
    all_iter_steps = []
    all_iter_train_losses = []
    all_iter_train_accs = []
    all_iter_val_losses = []
    all_iter_val_accs = []
    all_train_losses = []
    all_train_accs = []
    all_val_losses = []
    all_val_accs = []
    
    # Initialize the model with all layers frozen except the classifier
    set_parameter_requires_grad(resnet34, 0, 'gradual')
    
    # Initialize optimizer with only the classifier parameters
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4)
    # ===== L2 REGULARIZATION =========
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4, weight_decay=1e-4)
    
    # For plotting
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    iter_steps, iter_train_losses, iter_train_accs = [], [], []
    iter_val_losses, iter_val_accs = [], []
    
    # Track the current number of unfrozen blocks
    current_unfrozen_blocks = 0
    
    log("\n[Telemetry] Starting training with gradual unfreezing")

    for epoch in range(EPOCHS):
        log(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
        resnet34.train()
        train_loss, train_acc, n = 0, 0, 0
        total_batches = len(train_dl)
        
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_dl, total=total_batches)):
            # Determine when to unfreeze more blocks based on progress
            progress = (batch_idx + 1) / total_batches
            target_blocks = 1
            if progress >= 2/3:
                target_blocks = 3
            elif progress >= 1/3:
                target_blocks = 2
                
            # Update unfrozen blocks if needed
            if not hasattr(resnet34, 'current_blocks') or target_blocks != resnet34.current_blocks:
                # Freeze all parameters first
                for param in resnet34.parameters():
                    param.requires_grad = False
                
                # Unfreeze classifier and required blocks in layer4
                for param in resnet34.fc.parameters():
                    param.requires_grad = True
                    
                for i in range(target_blocks):
                    block_idx = len(resnet34.layer4) - 1 - i
                    for param in resnet34.layer4[block_idx].parameters():
                        param.requires_grad = True
                
                resnet34.current_blocks = target_blocks
                log(f"Progress: {progress*100:.1f}% - Unfrozen {target_blocks} blocks in layer4")
                
                # Update optimizer with new trainable parameters
                #optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4)
                optimizer = build_optimizer(resnet34) # ==== DIFFERENT LEARNING RATES FOR DIFFERENT LAYERS ===============

            
            # Training step
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = resnet34(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Update metrics
            train_loss += loss.item() * imgs.size(0)
            train_acc += accuracy(outputs, labels) * imgs.size(0)
            n += imgs.size(0)
            
            # Log progress periodically
            if (batch_idx) % 5 == 0:
                # Get current training metrics
                curr_train_loss = loss.item()
                curr_train_acc = accuracy(outputs, labels)
                
                # Get validation metrics (sample a random batch)
                resnet34.eval()
                with torch.no_grad():
                    # Sample from the validation loader with shuffling for better representation
                    val_imgs, val_labels = next(iter(val_iter_dl))
                    val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                    val_outputs = resnet34(val_imgs)
                    val_batch_loss = criterion(val_outputs, val_labels)
                    val_batch_acc = accuracy(val_outputs, val_labels)
                resnet34.train()  # Switch back to training mode
                
                # Store metrics
                iter_train_losses.append(curr_train_loss)
                iter_train_accs.append(curr_train_acc)
                iter_val_losses.append(val_batch_loss.item())
                iter_val_accs.append(val_batch_acc)
                iter_steps.append(epoch * len(train_dl) + batch_idx + 1)
                
                # Log metrics
                log(f"[Epoch {epoch+1}][Iter {batch_idx+1}] Train Loss: {curr_train_loss:.4f} | Train Acc: {curr_train_acc:.4f} | Val Loss: {val_batch_loss.item():.4f} | Val Acc: {val_batch_acc:.4f}")
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dl):
                log(f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Blocks: {resnet34.current_blocks} | Loss: {loss.item():.4f} | Acc: {accuracy(outputs, labels):.4f}")
        
        # Store epoch metrics
        epoch_loss = train_loss / n
        epoch_acc = train_acc / n
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Evaluate on validation set at the end of epoch
        resnet34.eval()
        val_loss, val_acc, val_n = 0, 0, 0
        with torch.no_grad():
            for val_imgs, val_labels in val_dl:
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                val_outputs = resnet34(val_imgs)
                val_batch_loss = criterion(val_outputs, val_labels)
                val_loss += val_batch_loss.item() * val_imgs.size(0)
                val_acc += accuracy(val_outputs, val_labels) * val_imgs.size(0)
                val_n += val_imgs.size(0)
            
        val_epoch_loss = val_loss/val_n
        val_epoch_acc = val_acc/val_n
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        log(f"[Telemetry] Epoch {epoch+1} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

    # Evaluate final validation metrics after all epochs
    resnet34.eval()
    with torch.no_grad():
        final_val_loss, final_val_acc, val_total = 0, 0, 0
        for val_imgs, val_labels in val_dl:
            val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
            val_outputs = resnet34(val_imgs)
            val_batch_loss = criterion(val_outputs, val_labels)
            final_val_loss += val_batch_loss.item() * val_imgs.size(0)
            final_val_acc += accuracy(val_outputs, val_labels) * val_imgs.size(0)
            val_total += val_imgs.size(0)
        final_val_loss /= val_total
        final_val_acc /= val_total
    
    log(f"\n[Telemetry] Final Model Validation | Loss: {final_val_loss:.4f} | Accuracy: {final_val_acc:.4f}")
    
    # Store final results
    all_iter_steps = iter_steps
    all_iter_train_losses = iter_train_losses
    all_iter_train_accs = iter_train_accs
    all_iter_val_losses = iter_val_losses
    all_iter_val_accs = iter_val_accs
    all_train_losses = train_losses
    all_train_accs = train_accs
    all_val_losses = val_losses
    all_val_accs = val_accs

    # Plot combined training and validation loss with block unfreezing markers
    plt.figure(figsize=(10, 6))
    plt.plot(all_iter_steps, all_iter_train_losses, color='blue', linestyle='-', label='Training')
    plt.plot(all_iter_steps, all_iter_val_losses, color='red', linestyle='--', label='Validation')
    
    # Mark when blocks were unfrozen
    # total_steps = len(all_iter_steps)
    total_steps = int(len(train_ds)/BATCH_SIZE)
    unfreeze_points = [(total_steps // 3, 'darkgreen', '2 blocks'), (2 * total_steps // 3, 'purple', '3 blocks')]
    for step, color, label in unfreeze_points:
        plt.axvline(x=step, color=color, linestyle='--', alpha=0.5, label=f'Unfreeze {label}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss with Gradual Unfreezing')
    plt.legend()
    loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-gradual-loss-combined.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Plot combined training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(all_iter_steps, all_iter_train_accs, color='green', linestyle='-', label='Training')
    plt.plot(all_iter_steps, all_iter_val_accs, color='purple', linestyle='--', label='Validation')
    
    # Add same vertical markers
    for step, color, label in unfreeze_points:
        plt.axvline(x=step, color=color, linestyle='--', alpha=0.5, label=f'Unfreeze {label}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy with Gradual Unfreezing')
    plt.legend()
    acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-gradual-acc-combined.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Create combined train vs validation loss plot
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(all_train_losses) + 1)
    plt.plot(epochs, all_train_losses, color='blue', marker='o', linestyle='-', label='Train')
    plt.plot(epochs, all_val_losses, color='red', marker='s', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss per Epoch')
    plt.legend()
    train_val_loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-gradual-train-vs-val-loss.png")
    plt.savefig(train_val_loss_plot_path)
    plt.close()
    
    # Create combined train vs validation accuracy plot
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, all_train_accs, color='green', marker='o', linestyle='-', label='Train')
    plt.plot(epochs, all_val_accs, color='purple', marker='s', linestyle='--', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy per Epoch')
    plt.legend()
    train_val_acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-gradual-train-vs-val-acc.png")
    plt.savefig(train_val_acc_plot_path)
    plt.close()
    
    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 2 Multi-class Classification with Gradual Unfreezing ({pretty_timestamp})\n")
        f.write(f"### Training vs Validation Loss per Iteration (Gradual Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training vs Validation Accuracy per Iteration (Gradual Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(acc_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training vs Validation Loss per Epoch (Gradual Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(train_val_loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training vs Validation Accuracy per Epoch (Gradual Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(train_val_acc_plot_path, os.getcwd())})\n\n")
        f.write(f"\n**Final Validation Metrics:**\n\n")
        f.write(f"- Loss: {final_val_loss:.4f}\n")
        f.write(f"- Accuracy: {final_val_acc:.4f}\n\n")
        f.write(f"\n**Training Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main()