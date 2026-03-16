"""
vgg16.py
---------
Step 2 of 3: VGG16 with Transfer Learning for Land Use Classification

What is VGG16?
    Published by Oxford's Visual Geometry Group in 2014.
    Uses 16 layers of 3×3 convolutions — very deep but uniform architecture.
    Pre-trained on ImageNet (1.2M images, 1000 classes).

What is Transfer Learning?
    Instead of training from scratch, we reuse weights already learned on ImageNet.
    The conv layers have learned to detect edges, textures, shapes → useful for
    satellite images too.
    We FREEZE conv layers (don't update them) and only train the final classifier.

Run:
    python vgg16.py
"""

import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# ← UPDATE THIS to your actual dataset path
DATA_DIR      = Path("/home/142502007/Downloads/EuroSAT_RGB")

CKPT_DIR      = Path("outputs/checkpoints")
PLOT_DIR      = Path("outputs/plots")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE    = 32    # VGG16 is large — use smaller batch to fit in memory
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
SEED          = 42
IMG_SIZE      = 64
NUM_CLASSES   = 10

CLASS_NAMES   = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),       # rotate ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # vary brightness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("\nLoading dataset...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
print(f"Total images : {len(full_dataset)}")

total   = len(full_dataset)
n_test  = int(total * TEST_SPLIT)
n_val   = int(total * VAL_SPLIT)
n_train = total - n_val - n_test

generator = torch.Generator().manual_seed(SEED)
train_ds, val_ds, test_ds = random_split(
    full_dataset, [n_train, n_val, n_test], generator=generator
)

val_ds.dataset        = copy.copy(full_dataset)
val_ds.dataset.transform  = val_transform
test_ds.dataset       = copy.copy(full_dataset)
test_ds.dataset.transform = val_transform

print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# ════════════════════════════════════════════════════════════════════════════
# MODEL — VGG16 with Transfer Learning
# ════════════════════════════════════════════════════════════════════════════

print("\nLoading VGG16 with ImageNet weights...")

# Load pre-trained VGG16
# weights=VGG16_Weights.DEFAULT loads the best available ImageNet weights
vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

# VGG16 architecture:
#   features   : 13 conv layers (3×3 filters) + 5 max-pool layers
#   avgpool    : adaptive average pooling → (batch, 512, 7, 7)
#   classifier : FC(25088→4096) → ReLU → Dropout
#                FC(4096→4096)  → ReLU → Dropout
#                FC(4096→1000)  ← we replace this

# FREEZE the feature extractor (conv layers)
# requires_grad=False means these weights will NOT be updated during training
# Why? They already know how to detect edges, textures, shapes from ImageNet
for param in vgg16.features.parameters():
    param.requires_grad = False

# Also freeze the first two FC layers in classifier
# Only fine-tune the last FC layer
for i, param in enumerate(vgg16.classifier.parameters()):
    if i < 4:   # first 2 FC layers have 2 params each (weight + bias)
        param.requires_grad = False

# Replace the final layer: FC(4096→1000) → FC(4096→10)
# in_features of the last layer = 4096
in_features = vgg16.classifier[-1].in_features   # = 4096
vgg16.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
# This new layer has requires_grad=True by default → will be trained

vgg16 = vgg16.to(DEVICE)

# Count parameters
total_params    = sum(p.numel() for p in vgg16.parameters())
trainable_params = sum(p.numel() for p in vgg16.parameters() if p.requires_grad)
frozen_params   = total_params - trainable_params
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}  (only these get updated)")
print(f"Frozen params   : {frozen_params:,}   (ImageNet weights, kept fixed)")


# ════════════════════════════════════════════════════════════════════════════
# LOSS, OPTIMIZER, SCHEDULER
# ════════════════════════════════════════════════════════════════════════════

criterion = nn.CrossEntropyLoss()

# Only pass trainable parameters to optimizer
# filter(lambda p: p.requires_grad, ...) skips frozen layers
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, vgg16.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4    # L2 regularisation: penalises large weights → less overfitting
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)


# ════════════════════════════════════════════════════════════════════════════
# TRAINING AND VALIDATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    """One full pass over training data with weight updates."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping: prevents exploding gradients in deep networks
        # Clips gradient norm to max_norm=1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model — no weight updates."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(dim=1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, 100.0 * correct / total


# ════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  Training VGG16 — {NUM_EPOCHS} epochs")
print(f"{'='*60}")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(vgg16, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = evaluate(vgg16, val_loader, criterion, DEVICE)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch [{epoch:>2}/{NUM_EPOCHS}] "
          f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% "
          f"| {time.time()-t0:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "epoch": epoch,
            "model_state": vgg16.state_dict(),
            "val_acc": val_acc,
        }, CKPT_DIR / "vgg16_best.pth")
        print(f"  ✓ Best model saved (val_acc={val_acc:.2f}%)")

print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")


# ════════════════════════════════════════════════════════════════════════════
# PLOT TRAINING CURVES
# ════════════════════════════════════════════════════════════════════════════

epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("VGG16 — Training History", fontsize=14, fontweight='bold')

ax1.plot(epochs, history["train_loss"], 'b-o', label="Train", markersize=4)
ax1.plot(epochs, history["val_loss"],   'r-o', label="Val",   markersize=4)
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, history["train_acc"], 'b-o', label="Train", markersize=4)
ax2.plot(epochs, history["val_acc"],   'r-o', label="Val",   markersize=4)
ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "vgg16_training_history.png", dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION
# ════════════════════════════════════════════════════════════════════════════

ckpt = torch.load(CKPT_DIR / "vgg16_best.pth", map_location=DEVICE)
vgg16.load_state_dict(ckpt["model_state"])

all_preds, all_labels = [], []
vgg16.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = vgg16(images.to(DEVICE))
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = 100 * accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc:.2f}%")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix — VGG16", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "vgg16_confusion_matrix.png", dpi=150)
plt.show()

print(f"\n{'='*60}")
print(f"  DONE — Test Accuracy: {test_acc:.2f}%")
print(f"  Next: run python resnet34.py")
print(f"{'='*60}")
