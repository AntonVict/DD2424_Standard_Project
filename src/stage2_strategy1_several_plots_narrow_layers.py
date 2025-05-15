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
# Range of block values to iterate over (blocks in layer4 to unfreeze)
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


def set_parameter_requires_grad(model, num_blocks, strategy):
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # Always unfreeze the classifier
    for param in model.fc.parameters():
        param.requires_grad = True
        
    if strategy == 'simultaneous':
        # ResNet34's layer4 has 3 blocks (indexed 0, 1, 2)
        max_blocks = len(model.layer4)
        num_blocks = min(num_blocks, max_blocks)
        
        # Print total number of blocks for debugging
        log(f"Total blocks in layer4: {max_blocks}")
        
        # Unfreeze the last N blocks of layer4
        blocks_unfrozen = []
        for i in range(num_blocks):
            # Start from the last block (index 2) and work backwards
            block_idx = max_blocks - 1 - i
            for param in model.layer4[block_idx].parameters():
                param.requires_grad = True
            blocks_unfrozen.append(block_idx)
            log(f"Unfreezing layer4.block{block_idx}")
        
        # Log the final unfrozen blocks
        log(f"Unfrozen blocks in layer4: {sorted(blocks_unfrozen)}")
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
    split = int(0.9 * n_total)
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
    
    # Create validation dataset and loader
    val_indices = indices[split:]
    val_ds = PetBreedDataset(DATA_DIR, TRAIN_FILE, indices=val_indices)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # Create a separate validation loader for iteration tracking with shuffling enabled
    val_iter_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    log(f"[Telemetry] Number of validation samples: {len(val_ds)}")
    
    # Initialize storage for all L values
    all_iter_steps = {}
    all_iter_train_losses = {}
    all_iter_train_accs = {}
    all_iter_val_losses = {}
    all_iter_val_accs = {}
    all_train_losses = {}
    all_train_accs = {}
    all_val_losses = {}
    all_val_accs = {}
    
    # Run for each block value within the specified range
    for blocks in range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1):
        log(f"\n[Telemetry] Training with {blocks} blocks unfrozen in layer4 + classifier")
        
        # Model
        resnet34 = models.resnet34(weights="IMAGENET1K_V1")
        num_ftrs = resnet34.fc.in_features
        resnet34.fc = nn.Linear(num_ftrs, 37)  # 37 breeds
        resnet34 = resnet34.to(device)

        # Set requires_grad according to strategy - unfreeze specific number of blocks
        set_parameter_requires_grad(resnet34, blocks, STRATEGY)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet34.parameters()), lr=1e-4)

        # For plotting
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        iter_steps, iter_train_losses, iter_train_accs = [], [], []
        iter_val_losses, iter_val_accs = [], []  # New lists to store validation metrics

        for epoch in range(EPOCHS):
            log(f"\n[Telemetry] Blocks={blocks}, Starting epoch {epoch+1}/{EPOCHS}")
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
                
                # Track training and validation metrics every 5 iterations
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
                    log(f"[Telemetry][Blocks={blocks}][Epoch {epoch+1}][Iter {batch_idx+1}] Train Loss: {curr_train_loss:.4f} | Train Acc: {curr_train_acc:.4f} | Val Loss: {val_batch_loss.item():.4f} | Val Acc: {val_batch_acc:.4f}")
                
                # Log batch progress
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_dl):
                    log(f"[Telemetry][Blocks={blocks}][Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Loss: {loss.item():.4f} | Acc: {accuracy(outputs, labels):.4f}")
                
            train_losses.append(train_loss/n)
            train_accs.append(train_acc/n)
            
            # Evaluate on validation set
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
            
            log(f"[Telemetry][Blocks={blocks}] Epoch {epoch+1} | Train Loss: {train_loss/n:.4f} | Train Acc: {train_acc/n:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}")

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
        
        log(f"[Telemetry][Blocks={blocks}] Final Validation Metrics | Loss: {final_val_loss:.4f} | Accuracy: {final_val_acc:.4f}")
        
        # Store results for this block count
        all_iter_steps[blocks] = iter_steps.copy()
        all_iter_train_losses[blocks] = iter_train_losses.copy()
        all_iter_train_accs[blocks] = iter_train_accs.copy()
        all_iter_val_losses[blocks] = iter_val_losses.copy()
        all_iter_val_accs[blocks] = iter_val_accs.copy()
        all_train_losses[blocks] = train_losses.copy()
        all_train_accs[blocks] = train_accs.copy()
        all_val_losses[blocks] = val_losses.copy()
        all_val_accs[blocks] = val_accs.copy()

    # Create the combined plot for training losses
    plt.figure(figsize=(10, 6))
    num_block_values = BLOCK_RANGE[1] - BLOCK_RANGE[0] + 1
    colors = plt.cm.viridis(np.linspace(0, 1, num_block_values))
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        plt.plot(all_iter_steps[blocks], all_iter_train_losses[blocks], 
                 label=f'Blocks={blocks} (Train)',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss per Iteration for Different Block Configurations')
    plt.legend()
    loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-loss-train.png")
    plt.savefig(loss_plot_path)
    plt.close()
    
    # Create the combined plot for training accuracies
    plt.figure(figsize=(10, 6))
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        plt.plot(all_iter_steps[blocks], all_iter_train_accs[blocks], 
                 label=f'Blocks={blocks} (Train)',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Training Accuracy')
    plt.title('Training Accuracy per Iteration for Different Block Configurations')
    plt.legend()
    acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-acc-train.png")
    plt.savefig(acc_plot_path)
    plt.close()
    
    # Create the combined plot for validation losses
    plt.figure(figsize=(10, 6))
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        plt.plot(all_iter_steps[blocks], all_iter_val_losses[blocks], 
                 label=f'Blocks={blocks}',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss per Iteration for Different Block Configurations')
    plt.legend()
    val_loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-val-loss.png")
    plt.savefig(val_loss_plot_path)
    plt.close()
    
    # Create the combined plot for validation accuracies
    plt.figure(figsize=(10, 6))
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        plt.plot(all_iter_steps[blocks], all_iter_val_accs[blocks], 
                 label=f'Blocks={blocks}',
                 color=colors[i])
    
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Iteration for Different Block Configurations')
    plt.legend()
    val_acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-val-acc.png")
    plt.savefig(val_acc_plot_path)
    plt.close()
    
    # Create plots for epoch-level train vs validation metrics
    # Loss comparison (train vs val) for each block configuration
    plt.figure(figsize=(12, 8))
    line_styles = ['-', '--']
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        epochs = range(1, len(all_train_losses[blocks]) + 1)
        plt.plot(epochs, all_train_losses[blocks], 
                 label=f'Blocks={blocks} (Train)',
                 color=colors[i], linestyle=line_styles[0])
        plt.plot(epochs, all_val_losses[blocks], 
                 label=f'Blocks={blocks} (Val)',
                 color=colors[i], linestyle=line_styles[1])
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss per Epoch for Different Block Configurations')
    plt.legend()
    train_val_loss_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-train-vs-val-loss.png")
    plt.savefig(train_val_loss_plot_path)
    plt.close()
    
    # Accuracy comparison (train vs val) for each block configuration
    plt.figure(figsize=(12, 8))
    
    for i, blocks in enumerate(range(BLOCK_RANGE[0], BLOCK_RANGE[1] + 1)):
        epochs = range(1, len(all_train_accs[blocks]) + 1)
        plt.plot(epochs, all_train_accs[blocks], 
                 label=f'Blocks={blocks} (Train)',
                 color=colors[i], linestyle=line_styles[0])
        plt.plot(epochs, all_val_accs[blocks], 
                 label=f'Blocks={blocks} (Val)',
                 color=colors[i], linestyle=line_styles[1])
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy per Epoch for Different Block Configurations')
    plt.legend()
    train_val_acc_plot_path = os.path.join(PLOT_DIR, f"{PLOT_PREFIX}-blocks-comparison-train-vs-val-acc.png")
    plt.savefig(train_val_acc_plot_path)
    plt.close()
    
    # Append to raw-results.md
    with open(RAW_RESULTS, "a") as f:
        f.write(f"\n\n## Stage 2 Multi-class Classification Block Comparison Run ({pretty_timestamp})\n")
        f.write(f"### Training Loss per Iteration (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training Accuracy per Iteration (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(acc_plot_path, os.getcwd())})\n\n")
        f.write(f"### Validation Loss per Iteration (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(val_loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Validation Accuracy per Iteration (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(val_acc_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training vs Validation Loss per Epoch (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(train_val_loss_plot_path, os.getcwd())})\n\n")
        f.write(f"### Training vs Validation Accuracy per Epoch (Layer4 Block Unfreezing)\n")
        f.write(f"![](./{os.path.relpath(train_val_acc_plot_path, os.getcwd())})\n\n")
        f.write(f"\n**Log:**\n\n")
        with open(LOG_FILE) as logf:
            for line in logf:
                f.write(line)

if __name__ == "__main__":
    main() 