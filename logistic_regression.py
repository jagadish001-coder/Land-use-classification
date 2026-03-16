"""
logistic_regression.py
-----------------------
Step 1 of 3: Logistic Regression baseline for Land Use Classification

What is Logistic Regression?
    Despite the name, it is a CLASSIFICATION model, not regression.
    It learns a single linear mapping from input pixels to class scores.
    
    Formula:  output = W * x + b
        W = weight matrix (learned during training)
        x = flattened input image (3 * 64 * 64 = 12288 values)
        b = bias term
    
    It is the simplest possible neural network — just ONE layer, no hidden layers.
    We use it as a BASELINE — if our fancy CNN can't beat this, something is wrong.

Run:
    python logistic_regression.py

Output:
    - Training progress printed every epoch
    - Test accuracy and classification report
    - Confusion matrix saved to outputs/plots/lr_confusion_matrix.png
    - Model checkpoint saved to outputs/checkpoints/logistic_best.pth
"""

# ── Standard library imports ───────────────────────────────────────────────
import os
import time
from pathlib import Path

# ── Numerical computing ────────────────────────────────────────────────────
import numpy as np

# ── PyTorch: the deep learning framework we use ────────────────────────────
import torch                        # core tensor operations
import torch.nn as nn               # neural network building blocks
import torch.optim as optim         # optimizers (SGD, Adam, etc.)
from torch.utils.data import DataLoader, random_split  # data pipeline

# ── Torchvision: datasets and image transforms ────────────────────────────
from torchvision import datasets, transforms

# ── Scikit-learn: for evaluation metrics ──────────────────────────────────
from sklearn.metrics import (
    classification_report,   # precision, recall, F1 per class
    confusion_matrix,        # matrix showing which classes get confused
    accuracy_score,          # overall accuracy
)

# ── Plotting ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns               # prettier confusion matrix heatmap


# ════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION — all hyperparameters in one place
# ════════════════════════════════════════════════════════════════════════════

# Path to your EuroSAT dataset — change this to match your actual path
DATA_DIR    = Path("/home/142502007/Downloads/EuroSAT_RGB")  # ← update if needed

# Output directories for saving results
CKPT_DIR    = Path("outputs/checkpoints")
PLOT_DIR    = Path("outputs/plots")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters — values that control how training works
BATCH_SIZE  = 64      # number of images processed together in one step
                      # larger = faster but needs more memory
NUM_EPOCHS  = 20      # how many full passes over the training data
                      # more epochs = more learning (up to a point)
LEARNING_RATE = 1e-3  # how big each weight update step is
                      # too high = unstable, too low = slow learning
VAL_SPLIT   = 0.15   # 15% of data used for validation
TEST_SPLIT  = 0.15   # 15% of data used for final testing
SEED        = 42      # random seed for reproducibility

# Image dimensions
IMG_SIZE    = 64      # EuroSAT images are 64x64 pixels
NUM_CLASSES = 10      # 10 land use categories
# Input size = 3 channels (RGB) × 64 × 64 pixels = 12288
INPUT_SIZE  = 3 * IMG_SIZE * IMG_SIZE   # = 12288

# Class names in the same order as folder names (alphabetical)
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# Device: use GPU if available, otherwise CPU
# GPU is ~10-50x faster for deep learning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════════════
# 2. DATA PIPELINE
# ════════════════════════════════════════════════════════════════════════════

# Transforms: operations applied to every image before feeding to model
# These ensure all images have the same format and value range

# Training transform: includes data augmentation
# Augmentation = artificially creating variations of images
# → makes model more robust (less overfitting)
train_transform = transforms.Compose([
    # Resize to 64x64 in case any images have different sizes
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    
    # Randomly flip image left-right with 50% probability
    # Satellite images look the same flipped → safe augmentation
    transforms.RandomHorizontalFlip(),
    
    # Randomly flip image top-bottom with 50% probability
    transforms.RandomVerticalFlip(),
    
    # Convert PIL image to PyTorch tensor
    # Also rescales pixel values from [0, 255] → [0.0, 1.0]
    # Changes shape from (H, W, C) → (C, H, W) which PyTorch expects
    transforms.ToTensor(),
    
    # Normalise each channel: pixel = (pixel - mean) / std
    # These mean/std values come from ImageNet dataset
    # Centering values near 0 helps gradient descent converge faster
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # mean for R, G, B channels
        std=[0.229, 0.224, 0.225]     # std  for R, G, B channels
    ),
])

# Validation/Test transform: NO augmentation
# We want to evaluate on clean images, not randomly flipped ones
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load the full dataset from the folder structure
# ImageFolder expects: root/class_name/image.jpg
# It automatically assigns integer labels based on folder names (alphabetical)
print("\nLoading dataset...")
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
print(f"Total images: {len(full_dataset)}")
print(f"Classes: {full_dataset.classes}")

# Split dataset into train / validation / test
# We do this manually so we control the exact sizes
total   = len(full_dataset)
n_test  = int(total * TEST_SPLIT)    # 15% → ~4050 images
n_val   = int(total * VAL_SPLIT)     # 15% → ~4050 images
n_train = total - n_val - n_test     # 70% → ~18900 images

# random_split randomly divides indices — seed ensures same split every run
generator = torch.Generator().manual_seed(SEED)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [n_train, n_val, n_test], generator=generator
)

# Override transform for val/test — no augmentation
# We need to make a copy to avoid changing the training set's transform
import copy
val_dataset.dataset  = copy.copy(full_dataset)
val_dataset.dataset.transform  = val_transform
test_dataset.dataset = copy.copy(full_dataset)
test_dataset.dataset.transform = val_transform

print(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

# DataLoader: handles batching, shuffling, and parallel loading
# shuffle=True for train: see images in random order each epoch
# num_workers: parallel processes for loading data (speeds up training)
train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = True,       # shuffle training data every epoch
    num_workers = 2,          # 2 parallel workers for loading
    pin_memory  = True        # speeds up CPU→GPU transfer
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,      # no need to shuffle validation data
    num_workers = 2,
    pin_memory  = True
)
test_loader = DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 2,
    pin_memory  = True
)


# ════════════════════════════════════════════════════════════════════════════
# 3. MODEL DEFINITION
# ════════════════════════════════════════════════════════════════════════════

class LogisticRegression(nn.Module):
    """
    Logistic Regression model for image classification.
    
    Architecture:
        Input  : (batch_size, 3, 64, 64)  — batch of RGB images
        Flatten: (batch_size, 12288)       — stretch each image into a vector
        Linear : (batch_size, 10)          — one score per class
        Output : raw scores (logits) — CrossEntropyLoss applies softmax internally
    
    nn.Module is the base class for all PyTorch models.
    We must define:
        __init__  : define the layers
        forward   : define how data flows through the layers
    """
    
    def __init__(self, input_size: int, num_classes: int):
        """
        Define the layers of the model.
        
        Args:
            input_size  : number of input features (3 * 64 * 64 = 12288)
            num_classes : number of output classes (10)
        """
        # Always call super().__init__() first — required by PyTorch
        super(LogisticRegression, self).__init__()
        
        # nn.Flatten: reshapes (batch, 3, 64, 64) → (batch, 12288)
        # This converts the 2D image into a 1D feature vector
        self.flatten = nn.Flatten()
        
        # nn.Linear(in, out): a fully connected layer
        # Learns a weight matrix W of shape (12288, 10) and bias b of shape (10,)
        # Output = W^T * x + b
        # Each of the 10 outputs represents the score for one class
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass — how data flows through the model.
        PyTorch automatically computes the backward pass (gradients) from this.
        
        Args:
            x : input tensor of shape (batch_size, 3, 64, 64)
        
        Returns:
            logits : tensor of shape (batch_size, 10) — raw class scores
        """
        x = self.flatten(x)   # (batch, 3, 64, 64) → (batch, 12288)
        x = self.linear(x)    # (batch, 12288)      → (batch, 10)
        return x              # return raw logits (not probabilities)


# Create model instance and move to device (GPU/CPU)
model = LogisticRegression(INPUT_SIZE, NUM_CLASSES).to(DEVICE)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: Logistic Regression")
print(f"Total parameters: {total_params:,}")
# Expected: 12288 * 10 weights + 10 biases = 122,890 parameters


# ════════════════════════════════════════════════════════════════════════════
# 4. LOSS FUNCTION AND OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════

# Loss function: measures how wrong the model's predictions are
# CrossEntropyLoss = Softmax + Negative Log Likelihood
# It expects raw logits (not probabilities) as input
# Lower loss = better predictions
criterion = nn.CrossEntropyLoss()

# Optimizer: algorithm that updates model weights to reduce loss
# Adam (Adaptive Moment Estimation):
#   - Maintains separate learning rates for each parameter
#   - Usually converges faster than plain SGD
#   - lr = learning rate (step size for weight updates)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: reduces LR when validation loss stops improving
# ReduceLROnPlateau:
#   - monitors validation loss
#   - if no improvement for `patience` epochs, multiply LR by `factor`
#   - helps fine-tune near convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode    = 'min',    # we want to MINIMIZE the loss
    factor  = 0.5,      # new_lr = old_lr * 0.5
    patience= 3,        # wait 3 epochs before reducing
)


# ════════════════════════════════════════════════════════════════════════════
# 5. TRAINING AND VALIDATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one full pass over the training data.
    
    One epoch = the model sees every training image exactly once.
    
    Steps for each mini-batch:
        1. Load batch of images and labels
        2. Move data to GPU/CPU
        3. Zero out gradients (PyTorch accumulates them by default)
        4. Forward pass: compute predictions
        5. Compute loss: how wrong are the predictions?
        6. Backward pass: compute gradients via chain rule (backprop)
        7. Update weights: move in direction that reduces loss
    
    Returns:
        avg_loss : average loss over all batches
        accuracy : percentage of correct predictions
    """
    model.train()   # set model to training mode
                    # (affects dropout and batchnorm — not used here but good habit)
    
    total_loss = 0.0   # accumulate loss across all batches
    correct    = 0     # count correct predictions
    total      = 0     # count total predictions
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to the device (GPU if available)
        images = images.to(device)   # shape: (batch_size, 3, 64, 64)
        labels = labels.to(device)   # shape: (batch_size,) — integer class indices
        
        # Step 3: Zero gradients
        # Without this, gradients from previous batch would accumulate
        optimizer.zero_grad()
        
        # Step 4: Forward pass
        # model(images) calls model.forward(images)
        outputs = model(images)      # shape: (batch_size, 10) — raw logits
        
        # Step 5: Compute loss
        # criterion compares predictions (logits) with true labels
        loss = criterion(outputs, labels)
        
        # Step 6: Backward pass
        # Computes gradient of loss with respect to every parameter
        # gradient = direction to move weights to reduce loss
        loss.backward()
        
        # Step 7: Update weights
        # moves each weight in the direction that reduces loss
        # step size = learning_rate * gradient
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item() * images.size(0)  # loss.item() gets scalar value
        
        # Get predicted class: index of highest score
        # dim=1 means take argmax across class dimension
        predictions = outputs.argmax(dim=1)
        
        # Compare predictions with true labels
        correct += (predictions == labels).sum().item()
        total   += labels.size(0)
    
    avg_loss = total_loss / total           # average loss per image
    accuracy = 100.0 * correct / total      # accuracy as percentage
    return avg_loss, accuracy


@torch.no_grad()   # decorator: disables gradient computation during evaluation
                   # saves memory and speeds up evaluation (no backprop needed)
def evaluate(model, loader, criterion, device):
    """
    Evaluate model on validation or test data.
    
    No weight updates here — we just measure performance.
    @torch.no_grad() means PyTorch won't track gradients → faster + less memory.
    
    Returns:
        avg_loss : average loss
        accuracy : percentage of correct predictions
    """
    model.eval()    # set model to evaluation mode
    
    total_loss = 0.0
    correct    = 0
    total      = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass only — no backward pass
        outputs     = model(images)
        loss        = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        
        total_loss += loss.item() * images.size(0)
        correct    += (predictions == labels).sum().item()
        total      += labels.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ════════════════════════════════════════════════════════════════════════════
# 6. TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"  Training Logistic Regression for {NUM_EPOCHS} epochs")
print(f"{'='*60}")

# Track history for plotting later
history = {
    "train_loss": [], "val_loss": [],
    "train_acc" : [], "val_acc" : [],
}

best_val_acc  = 0.0          # track best validation accuracy
best_epoch    = 0            # track which epoch was best

for epoch in range(1, NUM_EPOCHS + 1):
    t_start = time.time()
    
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE
    )
    
    # Evaluate on validation set
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
    
    # Update learning rate scheduler
    # It looks at val_loss and reduces LR if no improvement
    scheduler.step(val_loss)
    
    # Save metrics to history
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    
    elapsed = time.time() - t_start
    
    # Print progress
    print(
        f"Epoch [{epoch:>2}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f}  Acc: {train_acc:.2f}% "
        f"| Val Loss: {val_loss:.4f}  Acc: {val_acc:.2f}% "
        f"| {elapsed:.1f}s"
    )
    
    # Save model checkpoint if this is the best validation accuracy so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        
        # Save model state dict (weights) to disk
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),   # all learned weights
            "val_acc"    : val_acc,
            "val_loss"   : val_loss,
        }, CKPT_DIR / "logistic_best.pth")
        
        print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")

print(f"\nBest validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")


# ════════════════════════════════════════════════════════════════════════════
# 7. PLOT TRAINING CURVES
# ════════════════════════════════════════════════════════════════════════════

epochs = range(1, NUM_EPOCHS + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Logistic Regression — Training History", fontsize=14, fontweight='bold')

# Loss curve
ax1.plot(epochs, history["train_loss"], 'b-o', label="Train Loss", markersize=4)
ax1.plot(epochs, history["val_loss"],   'r-o', label="Val Loss",   markersize=4)
ax1.set_title("Loss per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend()
ax1.grid(alpha=0.3)

# Accuracy curve
ax2.plot(epochs, history["train_acc"], 'b-o', label="Train Acc", markersize=4)
ax2.plot(epochs, history["val_acc"],   'r-o', label="Val Acc",   markersize=4)
ax2.set_title("Accuracy per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_DIR / "lr_training_history.png", dpi=150)
plt.show()
print(f"Training curves saved → {PLOT_DIR}/lr_training_history.png")


# ════════════════════════════════════════════════════════════════════════════
# 8. FINAL EVALUATION ON TEST SET
# ════════════════════════════════════════════════════════════════════════════

# Load the best checkpoint
print("\nLoading best model for final evaluation...")
checkpoint = torch.load(CKPT_DIR / "logistic_best.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

# Collect all predictions on test set
print("Running inference on test set...")
all_preds  = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# Overall accuracy
test_accuracy = 100 * accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Per-class report
# Precision = of all images predicted as class X, how many were actually X?
# Recall    = of all actual class X images, how many did we correctly find?
# F1        = harmonic mean of precision and recall (balance between the two)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))


# ════════════════════════════════════════════════════════════════════════════
# 9. CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════════

# Confusion matrix: rows = true class, columns = predicted class
# Diagonal = correct predictions
# Off-diagonal = errors (e.g. Forest predicted as HerbaceousVegetation)

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm,
    annot      = True,          # show numbers in each cell
    fmt        = 'd',           # format as integer
    cmap       = 'Blues',       # colour scheme
    xticklabels= CLASS_NAMES,
    yticklabels= CLASS_NAMES,
    linewidths = 0.5,
    ax         = ax,
)
ax.set_title("Confusion Matrix — Logistic Regression", fontsize=14, fontweight='bold')
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(PLOT_DIR / "lr_confusion_matrix.png", dpi=150)
plt.show()
print(f"Confusion matrix saved → {PLOT_DIR}/lr_confusion_matrix.png")

print("\n" + "="*60)
print(f"  LOGISTIC REGRESSION COMPLETE")
print(f"  Test Accuracy : {test_accuracy:.2f}%")
print(f"  Best Val Acc  : {best_val_acc:.2f}%")
print(f"  Next step     : run vgg16.py to compare with a deep CNN")
print("="*60)
