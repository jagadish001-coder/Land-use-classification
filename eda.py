"""
eda.py  —  Exploratory Data Analysis for EuroSAT Dataset
=========================================================
Run this from your project root:
    python eda.py

What this script analyses:
  1. Folder structure & class names
  2. Class distribution (are classes balanced?)
  3. Image sizes & consistency check
  4. Pixel value statistics (mean, std per channel)
  5. Sample image grid (one row per class)
  6. Average colour per class (useful for understanding separability)
  7. Brightness distribution across classes
  8. RGB channel histograms
  9. Duplicate/corrupt image detection

All plots saved to: outputs/eda/
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_PATH = Path.home() / "Downloads" / "EuroSAT_RGB"   # change if needed
OUTPUT_DIR   = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════

def load_image_paths(root: Path) -> dict:
    """Return {class_name: [list of image paths]} from a folder of subfolders."""
    class_paths = {}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            imgs = [p for p in cls_dir.iterdir()
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif')]
            if imgs:
                class_paths[cls_dir.name] = imgs
    return class_paths


# ════════════════════════════════════════════════════════════════════════════
# 1. FOLDER STRUCTURE
# ════════════════════════════════════════════════════════════════════════════

def analyse_structure(class_paths: dict):
    print("\n" + "="*60)
    print("  1. DATASET STRUCTURE")
    print("="*60)
    total = 0
    for cls, paths in class_paths.items():
        print(f"  {cls:<30}  {len(paths):>5} images")
        total += len(paths)
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<30}  {total:>5} images")
    print(f"  Number of classes: {len(class_paths)}")
    return total


# ════════════════════════════════════════════════════════════════════════════
# 2. CLASS DISTRIBUTION PLOT
# ════════════════════════════════════════════════════════════════════════════

def plot_class_distribution(class_paths: dict):
    """
    A balanced dataset is important — if one class has far more images,
    the model can bias toward it and ignore rare classes.
    EuroSAT is roughly balanced (~2000-3000 per class), which is good.
    """
    classes = list(class_paths.keys())
    counts  = [len(v) for v in class_paths.values()]
    colours = plt.cm.Set3(np.linspace(0, 1, len(classes)))

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(classes, counts, color=colours, edgecolor='white', linewidth=0.8)

    # Add count labels on top of each bar
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Class Distribution — EuroSAT", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Images")
    ax.set_xlabel("Land Use Class")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.xticks(rotation=30, ha='right', fontsize=9)

    # Add mean line
    mean_count = np.mean(counts)
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(len(classes)-0.5, mean_count + 30, f'mean={mean_count:.0f}',
            color='red', fontsize=9, ha='right')

    plt.tight_layout()
    path = OUTPUT_DIR / "1_class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  ✅  Saved → {path}")
    print(f"  Imbalance ratio (max/min): {max(counts)/min(counts):.2f}x")
    print(f"  Verdict: {'⚠️  Imbalanced' if max(counts)/min(counts) > 2 else '✅  Balanced'}")


# ════════════════════════════════════════════════════════════════════════════
# 3. IMAGE SIZE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyse_image_sizes(class_paths: dict):
    """
    Check if all images are the same size.
    Mixed sizes require resizing (handled by transforms.Resize in our pipeline).
    Also checks for corrupt images that can't be opened.
    """
    print("\n" + "="*60)
    print("  3. IMAGE SIZE ANALYSIS")
    print("="*60)

    size_counter = defaultdict(int)
    corrupt      = []
    # Sample max 50 images per class for speed
    sample_paths = []
    for paths in class_paths.values():
        sample_paths.extend(paths[:50])

    for p in tqdm(sample_paths, desc="  Checking sizes"):
        try:
            with Image.open(p) as img:
                size_counter[img.size] += 1
        except Exception:
            corrupt.append(p)

    print(f"\n  Unique sizes found: {len(size_counter)}")
    for size, count in sorted(size_counter.items(), key=lambda x: -x[1]):
        pct = 100 * count / sum(size_counter.values())
        print(f"    {str(size):<15}  {count:>5} images  ({pct:.1f}%)")

    if corrupt:
        print(f"\n  ⚠️  Corrupt/unreadable images: {len(corrupt)}")
        for p in corrupt[:5]:
            print(f"    → {p}")
    else:
        print(f"\n  ✅  No corrupt images found")

    dominant = max(size_counter, key=size_counter.get)
    print(f"\n  Dominant size: {dominant[0]}×{dominant[1]} px")
    if len(size_counter) == 1:
        print(f"  ✅  All images are the same size — no resizing issues")
    else:
        print(f"  ℹ️  Mixed sizes detected — transforms.Resize(64,64) will standardise them")
    return corrupt


# ════════════════════════════════════════════════════════════════════════════
# 4. PIXEL STATISTICS PER CLASS
# ════════════════════════════════════════════════════════════════════════════

def compute_pixel_stats(class_paths: dict):
    """
    Compute per-channel mean & std for each class.
    Why this matters:
      - Mean tells us the dominant colour of a class (Forest = high green)
      - Std tells us texture richness (Highway = low std = uniform grey)
      - These stats are what transforms.Normalize uses to centre the data
    """
    print("\n" + "="*60)
    print("  4. PIXEL STATISTICS PER CHANNEL")
    print("="*60)
    print(f"  {'Class':<25} {'R_mean':>7} {'G_mean':>7} {'B_mean':>7} {'R_std':>7} {'G_std':>7} {'B_std':>7}")
    print(f"  {'─'*65}")

    class_stats = {}
    all_means   = []

    for cls, paths in class_paths.items():
        # Sample up to 100 images per class
        sample = paths[:100]
        pixels = []
        for p in sample:
            try:
                img = np.array(Image.open(p).convert('RGB').resize((64, 64)))
                pixels.append(img)
            except:
                continue

        if not pixels:
            continue

        arr = np.stack(pixels, axis=0).astype(np.float32)  # (N, 64, 64, 3)
        means = arr.mean(axis=(0,1,2))   # mean across all pixels, per channel
        stds  = arr.std(axis=(0,1,2))

        class_stats[cls] = {'means': means, 'stds': stds}
        all_means.append(means)

        print(f"  {cls:<25} "
              f"{means[0]:>7.1f} {means[1]:>7.1f} {means[2]:>7.1f} "
              f"{stds[0]:>7.1f} {stds[1]:>7.1f} {stds[2]:>7.1f}")

    # Compute global stats (used in transforms.Normalize)
    global_mean = np.mean([s['means'] for s in class_stats.values()], axis=0) / 255.0
    global_std  = np.mean([s['stds']  for s in class_stats.values()], axis=0) / 255.0
    print(f"\n  Global mean (0-1 scale): R={global_mean[0]:.3f}  G={global_mean[1]:.3f}  B={global_mean[2]:.3f}")
    print(f"  Global std  (0-1 scale): R={global_std[0]:.3f}  G={global_std[1]:.3f}  B={global_std[2]:.3f}")
    print(f"\n  (ImageNet values used in our pipeline:  mean=[0.485, 0.456, 0.406])")

    return class_stats


# ════════════════════════════════════════════════════════════════════════════
# 5. SAMPLE IMAGE GRID
# ════════════════════════════════════════════════════════════════════════════

def plot_sample_grid(class_paths: dict, n_per_class: int = 5):
    """Show n sample images per class in a grid so we can visually inspect quality."""
    classes = list(class_paths.keys())
    n_cls   = len(classes)

    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(n_per_class * 2.5, n_cls * 2.2))
    fig.suptitle(f"Sample Images — {n_per_class} per class", fontsize=14, fontweight='bold', y=1.01)

    for row, cls in enumerate(classes):
        # Set row label
        axes[row][0].set_ylabel(cls, fontsize=8, fontweight='bold', rotation=0,
                                 labelpad=80, va='center')
        sample = class_paths[cls][:n_per_class]
        for col in range(n_per_class):
            ax = axes[row][col]
            if col < len(sample):
                try:
                    img = Image.open(sample[col]).convert('RGB')
                    ax.imshow(img)
                except:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.tight_layout()
    path = OUTPUT_DIR / "2_sample_grid.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"\n  ✅  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 6. AVERAGE COLOUR PER CLASS
# ════════════════════════════════════════════════════════════════════════════

def plot_average_colours(class_stats: dict):
    """
    The average colour patch per class tells us how visually distinct the classes are.
    Very different colours = the model can separate them easily.
    Similar colours = harder — model needs to learn texture, shape, etc.
    """
    classes = list(class_stats.keys())
    colours = [class_stats[c]['means'] / 255.0 for c in classes]  # normalise 0-1

    fig, axes = plt.subplots(2, len(classes)//2 + len(classes)%2, figsize=(14, 4))
    axes = axes.flatten()
    fig.suptitle("Average Colour per Class", fontsize=13, fontweight='bold')

    for i, (cls, colour) in enumerate(zip(classes, colours)):
        axes[i].add_patch(plt.Rectangle((0,0), 1, 1, color=np.clip(colour, 0, 1)))
        axes[i].set_title(cls, fontsize=8, fontweight='bold')
        axes[i].set_xlim(0,1); axes[i].set_ylim(0,1)
        axes[i].axis('off')

        # Add RGB values as text
        axes[i].text(0.5, 0.15,
                     f"R:{colour[0]*255:.0f}\nG:{colour[1]*255:.0f}\nB:{colour[2]*255:.0f}",
                     ha='center', va='bottom', fontsize=7,
                     color='white' if np.mean(colour) < 0.5 else 'black')

    # Hide extra axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    path = OUTPUT_DIR / "3_average_colours.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  ✅  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. BRIGHTNESS DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

def plot_brightness_distribution(class_paths: dict):
    """
    Brightness = average pixel value across all channels.
    This shows whether some classes are systematically darker/brighter
    (e.g. SeaLake = dark blue/black, Desert = bright yellow).
    If distributions overlap heavily → harder for model to distinguish.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(class_paths)))

    for (cls, paths), color in zip(class_paths.items(), colors):
        brightnesses = []
        for p in paths[:80]:   # sample 80 per class
            try:
                img = np.array(Image.open(p).convert('RGB').resize((64,64)))
                brightnesses.append(img.mean())
            except:
                continue

        ax.hist(brightnesses, bins=30, alpha=0.55, label=cls,
                color=color, density=True, edgecolor='none')

    ax.set_title("Brightness Distribution by Class", fontsize=13, fontweight='bold')
    ax.set_xlabel("Mean Pixel Value (0–255)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "4_brightness_distribution.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  ✅  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 8. RGB CHANNEL HISTOGRAMS (overall dataset)
# ════════════════════════════════════════════════════════════════════════════

def plot_rgb_histograms(class_paths: dict):
    """
    Shows the overall distribution of R, G, B pixel values across the full dataset.
    A healthy dataset should have values spread across the full 0–255 range.
    The overlap between channels tells us about image colouring.
    """
    r_vals, g_vals, b_vals = [], [], []

    all_paths = []
    for paths in class_paths.values():
        all_paths.extend(paths[:30])   # 30 per class

    for p in tqdm(all_paths, desc="  Computing RGB histograms"):
        try:
            img = np.array(Image.open(p).convert('RGB').resize((32,32)))
            r_vals.extend(img[:,:,0].flatten().tolist())
            g_vals.extend(img[:,:,1].flatten().tolist())
            b_vals.extend(img[:,:,2].flatten().tolist())
        except:
            continue

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(0, 256, 4)
    ax.hist(r_vals, bins=bins, color='#E24B4A', alpha=0.6, label='Red',   density=True)
    ax.hist(g_vals, bins=bins, color='#639922', alpha=0.6, label='Green', density=True)
    ax.hist(b_vals, bins=bins, color='#378ADD', alpha=0.6, label='Blue',  density=True)
    ax.set_title("RGB Pixel Value Distribution (full dataset)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Pixel Value (0–255)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = OUTPUT_DIR / "5_rgb_histograms.png"
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"\n  ✅  Saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 9. DUPLICATE DETECTION
# ════════════════════════════════════════════════════════════════════════════

def detect_duplicates(class_paths: dict):
    """
    Hash every image (MD5) to detect exact duplicates.
    Duplicates that appear in both train and test splits cause data leakage
    — the model memorises them and reports inflated accuracy.
    """
    print("\n" + "="*60)
    print("  9. DUPLICATE DETECTION")
    print("="*60)

    hashes = defaultdict(list)
    all_paths = [p for paths in class_paths.values() for p in paths]

    for p in tqdm(all_paths[:500], desc="  Hashing images (sample)"):
        try:
            with open(p, 'rb') as f:
                h = hashlib.md5(f.read()).hexdigest()
            hashes[h].append(p)
        except:
            continue

    duplicates = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    if duplicates:
        print(f"\n  ⚠️  Found {len(duplicates)} duplicate groups:")
        for h, paths in list(duplicates.items())[:5]:
            print(f"    Hash {h[:8]}... → {[str(p.name) for p in paths]}")
    else:
        print(f"\n  ✅  No duplicates found in sampled images")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n🛰️  EuroSAT EDA — Dataset path: {DATASET_PATH}")

    # Check path exists
    if not DATASET_PATH.exists():
        print(f"\n❌  Dataset not found at: {DATASET_PATH}")
        print(f"    Please update DATASET_PATH at the top of this script.")
        print(f"    Example:  DATASET_PATH = Path('/your/actual/path/EuroSAT_RGB')")
        sys.exit(1)

    # Load all image paths
    class_paths = load_image_paths(DATASET_PATH)
    if not class_paths:
        print("❌  No class folders found. Check the dataset structure.")
        sys.exit(1)

    # Run all analyses
    total         = analyse_structure(class_paths)
    plot_class_distribution(class_paths)
    corrupt       = analyse_image_sizes(class_paths)
    class_stats   = compute_pixel_stats(class_paths)
    print(f"\n  ⏳  Generating sample grid (may take ~30s) ...")
    plot_sample_grid(class_paths, n_per_class=5)
    plot_average_colours(class_stats)
    plot_brightness_distribution(class_paths)
    plot_rgb_histograms(class_paths)
    detect_duplicates(class_paths)

    # Final summary
    print("\n" + "="*60)
    print("  EDA COMPLETE — Key Takeaways")
    print("="*60)
    counts = [len(v) for v in class_paths.values()]
    print(f"  • Total images        : {total:,}")
    print(f"  • Number of classes   : {len(class_paths)}")
    print(f"  • Min images in class : {min(counts)}")
    print(f"  • Max images in class : {max(counts)}")
    print(f"  • Imbalance ratio     : {max(counts)/min(counts):.2f}x")
    print(f"  • Corrupt images      : {len(corrupt)}")
    print(f"\n  All plots saved to: {OUTPUT_DIR.resolve()}")
    print(f"\n  ✅  Ready to train! Run: python train.py --no_download --data_dir \"{DATASET_PATH}\"")


if __name__ == "__main__":
    main()
