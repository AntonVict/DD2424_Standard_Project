import torch
import torchvision.models as models
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# Custom Dataset for binary classification (cat vs. dog)
class PetBinaryDataset(Dataset):
    def __init__(self, data_dir, split_file):
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

    # Datasets and loaders
    train_ds = PetBinaryDataset(DATA_DIR, TRAIN_FILE)
    test_ds = PetBinaryDataset(DATA_DIR, TEST_FILE)
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Telemetry: Data and device info
    print(f"[Telemetry] Number of training samples: {len(train_ds)}")
    print(f"[Telemetry] Number of test samples: {len(test_ds)}")
    print(f"[Telemetry] Batch size: {batch_size}")
    print(f"[Telemetry] Using device: {device}")

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

    # Training loop
    EPOCHS = 3  # Increase for real training
    for epoch in range(EPOCHS):
        print(f"\n[Telemetry] Starting epoch {epoch+1}/{EPOCHS}")
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
                print(f"[Telemetry][Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_dl)} | Loss: {loss.item():.4f} | Acc: {accuracy(outputs, labels):.4f}")
        print(f"[Telemetry] Epoch {epoch+1} | Train Loss: {train_loss/n:.4f} | Train Acc: {train_acc/n:.4f}")

        resnet34.eval()
        val_loss, val_acc, n = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = resnet34(imgs).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_acc += accuracy(outputs, labels) * imgs.size(0)
                n += imgs.size(0)
        print(f"[Telemetry] Epoch {epoch+1} | Val Loss: {val_loss/n:.4f} | Val Acc: {val_acc/n:.4f}")


if __name__ == "__main__":
    main()