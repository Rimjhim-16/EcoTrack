import random
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

RAW = Path("raw")
OUT = Path("dataset")
CLASSES = ["plastic", "paper", "metal", "organic", "ewaste"]
MAX_PER_CLASS = 450
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}

random.seed(42)
for s in SPLITS:
    (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

counts = {s: {c: 0 for c in CLASSES} for s in SPLITS}

for cid, cname in enumerate(CLASSES):
    class_folder = RAW / cname
    if not class_folder.exists():
        print(f"Warning: {class_folder} not found! Skipping.")
        continue
    imgs = [p for p in class_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.shuffle(imgs)
    imgs = imgs[:MAX_PER_CLASS]
    n = len(imgs)
    
    train_cut = int(0.7 * n)
    val_cut = int(0.9 * n)
    
    for i, p in enumerate(imgs):
        split = "train" if i < train_cut else "val" if i < val_cut else "test"
        new_name = f"{cname}_{i:04d}{p.suffix.lower()}"
        shutil.copy(p, OUT / "images" / split / new_name)
        
        # Write dummy full-image bounding box format: class_id center_x center_y width height
        (OUT / "labels" / split / f"{cname}_{i:04d}.txt").write_text(f"{cid} 0.5 0.5 0.95 0.95\n")
        counts[split][cname] += 1

total = {c: sum(counts[s][c] for s in SPLITS) for c in CLASSES}
print("Dataset split counts:", counts)
print("Total per class:", total)

# Generate class distribution chart for slides
plt.figure(figsize=(7, 4))
plt.bar(total.keys(), total.values(), color=["#2E86AB", "#F6AE2D", "#9A9A9A", "#58A55C", "#C0392B"])
plt.title("EcoTrack Dataset Distribution")
plt.ylabel("Images")
for i, v in enumerate(total.values()):
    plt.text(i, v + 5, str(v), ha="center")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=200)
print("Saved class_distribution.png successfully!")