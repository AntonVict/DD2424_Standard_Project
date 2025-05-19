# stage2_strategy2_different_lr.py
import torch
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import datetime
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm

# Different learning rates
def build_optimizer(model, base_lr=1e-4):
    param_groups = []
    param_groups.append({'params': model.fc.parameters(), 'lr': base_lr})
    if hasattr(model, 'current_blocks'):
        for i in range(model.current_blocks):
            block_idx = len(model.layer4) - 1 - i
            block = model.layer4[block_idx]
            block_lr = base_lr * (0.5 ** i)
          
            param_groups.append({'params': block.parameters(), 'lr': block_lr})
    return optim.Adam(param_groups)

# dataset class
class PetBreedDataset(Dataset):
    def __init__(self, data_dir, split_file, indices=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts)<4: continue
                name,classid,_,_ = parts
                label = int(classid) - 1
                self.samples.append((f"{name}.jpg", label))
        if indices:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        image = np.array(image).transpose(2, 0, 1) / 255.0
        image = torch.FloatTensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        return image, label


def accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    return (preds == labels).float().mean().item()

def main(STRATEGY_TAG=None):
    train_file = os.path.join(DATA_DIR, "annotations", "trainval.txt")
    with open(train_file) as f:
        lines = [l for l in f if len(l.strip().split()) >= 4]
    indices = list(range(len(lines)))
    random.seed(42)
    random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_indices, val_indices = indices[:split], indices[split:]
    train_ds = PetBreedDataset(DATA_DIR, train_file, train_indices)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = models.resnet34(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 37)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    for p in model.parameters(): p.requires_grad = False
    for p in model.fc.parameters(): p.requires_grad = True
    model.current_blocks = 3
    for i in range(model.current_blocks):
        for p in model.layer4[i].parameters():
            p.requires_grad = True

  
    if STRATEGY_TAG.startswith("uniform"):
        base_lr = float(STRATEGY_TAG.replace("uniform_",""))
    elif STRATEGY_TAG == "layerwise":
        base_lr = 1e-4 # the default value for layerwise
    else:
        raise ValueError(f"Unrecongized STRATEGY_TAG: {STRATEGY_TAG}")

    optimizer = build_optimizer(model, base_lr=base_lr)
    #optimizer = build_optimizer(model, base_lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    iter_steps, iter_train_losses, iter_train_accs = [], [], []
    step = 0

    model.train()
    for epoch in range(EPOCHS):
        for imgs, labels in tqdm(train_dl):
            imgs, labels = imgs.to(model.fc.weight.device), labels.to(model.fc.weight.device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = accuracy(outputs,labels)
            iter_steps.append(step)
            iter_train_losses.append(loss.item())
            iter_train_accs.append(acc)
            step += 1
    # Save metrics
    save_path = os.path.join(LOG_DIR, f"stage2-{STRATEGY_TAG}-metrics.pt")
    torch.save({
        "tag": STRATEGY_TAG,
        "iter_steps": iter_steps,
        "iter_train_losses": iter_train_losses,
        "iter_train_accs": iter_train_accs,
    }, save_path)

    # Plot training loss and accuracy vs iteration for all strategies
    runs = {
        "Uniform LR (1e-6)": "logs/stage2-uniform_1e-6-metrics.pt",
        "Uniform LR (1e-5)": "logs/stage2-uniform_1e-5-metrics.pt",
        "Uniform LR (1e-4)": "logs/stage2-uniform_1e-4-metrics.pt",
        "Uniform LR (1e-3)": "logs/stage2-uniform_1e-3-metrics.pt",
        "Uniform LR (1e-2)": "logs/stage2-uniform_1e-2-metrics.pt",
        "Layerwise Decay": "logs/stage2-layerwise-metrics.pt"
    }
  
    plot(runs)
    

def plot(runs):
    # Training loss vs iteration
    plt.figure(figsize=(10, 6))
    for label, path in runs.items():
        if not os.path.exists(path):
            print(f"[!] Missing file: {path}")
            continue
        data = torch.load(path)
        plt.plot(data["iter_steps"], data["iter_train_losses"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/compare_training_loss_vs_iteration.png")
    plt.close()

    # training accuracy vs iteration
    plt.figure(figsize=(10, 6))
    for label, path in runs.items():
        if not os.path.exists(path):
            print(f"[!] Missing file: {path}")
            continue
        data = torch.load(path)
        plt.plot(data["iter_steps"], data["iter_train_accs"], label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/compare_training_accuracy_vs_iteration.png")
    plt.close()    

if __name__ == "__main__":
    for STRATEGY_TAG in ["uniform_1e-6", "uniform_1e-5", "uniform_1e-4", "uniform_1e-3", "uniform_1e-2", "layerwise"]:
        
        BATCH_SIZE = 32
        EPOCHS = 1
        DATA_DIR = "dataset/oxford-iiit-pet"
        LOG_DIR = "logs"
        os.makedirs(LOG_DIR, exist_ok=True)
        pretty_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        PLOT_PREFIX = f"{STRATEGY_TAG}-{pretty_timestamp}"
        print(f"\n=== Running strategy: {STRATEGY_TAG} ===")
        main(STRATEGY_TAG=STRATEGY_TAG)


