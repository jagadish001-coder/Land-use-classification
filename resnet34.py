"""
resnet34.py
-----------
Step 3 of 3: ResNet34 with Transfer Learning for Land Use Classification

What is ResNet34?
    Published by Microsoft Research in 2015.
    34 layers deep with RESIDUAL (SKIP) CONNECTIONS.

What are Residual Connections?
    Normal CNN:  output = F(x)           — just the transformation
    ResNet:      output = F(x) + x       — transformation PLUS original input

    Why does this help?
    - Gradients can flow directly through the skip connection
    - Solves the vanishing gradient problem in deep networks
    - If F(x)=0 is best, the block just passes x through unchanged
    - Result: we can train much deeper networks effectively

Why ResNet34 vs VGG16?
    - Fewer parameters (~21M vs ~138M) but usually better accuracy
    - Faster to train
    - Residual connections = more stable training

Run:
    python resnet34.py
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

BATCH_SIZE    = 64    # ResNet34 is lighter than VGG16, can use larger batch
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
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
# MODEL — ResNet34 with Transfer Learning
# ════════════════════════════════════════════════════════════════════════════

print("\nLoading ResNet34 with ImageNet weights...")

# Load pre-trained ResNet34
resnet34 = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

# ResNet34 architecture:
#   conv1        : 7×7 conv, stride 2
#   bn1, relu    : batch norm + activation
#   maxpool      : 3×3 max pool, stride 2
#   layer1       : 3 residual blocks (64 filters)
#   layer2       : 4 residual blocks (128 filters)
#   layer3       : 6 residual blocks (256 filters)
#   layer4       : 3 residual blocks (512 filters)
#   avgpool      : global average pooling → (batch, 512)
#   fc           : Linear(512, 1000)  ← we replace this

# FREEZE strategy: freeze layers 1-3, fine-tune layer4 + fc
# layer4 is closest to the output → most task-specific features
# Fine-tuning it helps the model adapt to satellite imagery
for name, param in resnet34.named_parameters():
    if "layer4" not in name and "fc" not in name:
        # Freeze everything except layer4 and fc
        param.requires_grad = False

# Replace final FC layer: Linear(512, 1000) → Linear(512, 10)
in_features  = resnet34.fc.in_features   # = 512
resnet34.fc  = nn.Linear(in_features, NUM_CLASSES)
# New fc layer has requires_grad=True by default

resnet34 = resnet34.to(DEVICE)

total_params     = sum(p.numel() for p in resnet34.parameters())
trainable_params = sum(p.numel() for p in resnet34.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params
print(f"Total params    : {total_params:,}")
print(f"Trainable params: {trainable_params:,}")
print(f"Frozen params   : {frozen_params:,}")


# ════════════════════════════════════════════════════════════════════════════
# LOSS, OPTIMIZER, SCHEDULER
# ════════════════════════════════════════════════════════════════════════════

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, resnet34.parameters()),
    lr=LEARNING_RATE,
    weight_decay=1e-4
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
print(f"  Training ResNet34 — {NUM_EPOCHS} epochs")
print(f"{'='*60}")

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()

    train_loss, train_acc = train_one_epoch(resnet34, train_loader, optimizer, criterion, DEVICE)
    val_loss,   val_acc   = evaluate(resnet34, val_loader, criterion, DEVICE)
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
            "model_state": resnet34.state_dict(),
            "val_acc": val_acc,
        }, CKPT_DIR / "resnet34_best.pth")
        print(f"  ✓ Best model saved (val_acc={val_acc:.2f}%)")

print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")


# ════════════════════════════════════════════════════════════════════════════
# PLOT TRAINING CURVES
# ════════════════════════════════════════════════════════════════════════════

epochs = range(1, NUM_EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ResNet34 — Training History", fontsize=14, fontweight='bold')

ax1.plot(epochs, history["train_loss"], 'b-o', label="Train", markersize=4)
ax1.plot(epochs, history["val_loss"],   'r-o', label="Val",   markersize=4)
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(epochs, history["train_acc"], 'b-o', label="Train", markersize=4)
ax2.plot(epochs, history["val_acc"],   'r-o', label="Val",   markersize=4)
ax2.set_title("Accuracy (%)"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "resnet34_training_history.png", dpi=150)
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# FINAL TEST EVALUATION
# ════════════════════════════════════════════════════════════════════════════

ckpt = torch.load(CKPT_DIR / "resnet34_best.pth", map_location=DEVICE)
resnet34.load_state_dict(ckpt["model_state"])

all_preds, all_labels = [], []
resnet34.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = resnet34(images.to(DEVICE))
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_title("Confusion Matrix — ResNet34", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "resnet34_confusion_matrix.png", dpi=150)
plt.show()

print(f"\n{'='*60}")
print(f"  DONE — Test Accuracy: {test_acc:.2f}%")
print(f"  All 3 models complete! Compare results:")
print(f"  Logistic Regression → outputs/checkpoints/logistic_best.pth")
print(f"  VGG16               → outputs/checkpoints/vgg16_best.pth")
print(f"  ResNet34            → outputs/checkpoints/resnet34_best.pth")
print(f"{'='*60}")
