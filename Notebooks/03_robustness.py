"""
Step: Robustness testing
How much do classification accuracy and retrieval P@5 drop when test MRIs are
corrupted (noise, blur, brightness, contrast, rotation, JPEG compression)?

Compares:
  - Classifier accuracy (augmented ResNet18)
  - Retrieval P@5 with frozen ResNet embeddings
  - Retrieval P@5 with the SupCon head
Gallery = clean training images (from 02_contrastive_head.py cache).
Test duplicates of training images are removed.

Outputs:
  ../results/robustness.csv
  ../results/robustness.png
  ../results/corruption_examples.png
Runtime on CPU: ~5-8 minutes.
"""
import hashlib
import io
import os

import matplotlib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import datasets, models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("MRI_ROOT", "..")
TRAIN_DIR = f"{ROOT}/Data/brain_tumor/Training"
TEST_DIR = f"{ROOT}/Data/brain_tumor/Testing"
MODELS = f"{ROOT}/models"
RESULTS = f"{ROOT}/results"
CKPT = os.environ.get("MRI_CKPT", "best_brain_mri_resnet18_augmented.pth")
os.makedirs(RESULTS, exist_ok=True)
torch.manual_seed(0)
device = torch.device("cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ---------- Models ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 4)
model.load_state_dict(torch.load(CKPT, map_location=device))
classifier_fc = model.fc
model.fc = nn.Identity()          # model now outputs 512-d features
model.eval(); classifier_fc.eval()


class Head(nn.Module):
    def __init__(self, d_in=512, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


head = Head()
head.load_state_dict(torch.load(f"{MODELS}/supcon_head.pth", map_location=device))
head.eval()

Xtr, ytr, _, _ = torch.load(f"{MODELS}/frozen_feats.pt")
with torch.no_grad():
    gallery_frozen = F.normalize(Xtr, dim=1)
    gallery_supcon = head(Xtr)

# ---------- Clean test images (duplicates removed) ----------
def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

train_hashes = {md5(p) for p, _ in datasets.ImageFolder(TRAIN_DIR).samples}
test_samples = [(p, y) for p, y in datasets.ImageFolder(TEST_DIR).samples if md5(p) not in train_hashes]
classes = datasets.ImageFolder(TEST_DIR).classes
print(f"Test images after removing duplicates: {len(test_samples)}")

to_tensor = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
images = torch.stack([to_tensor(Image.open(p).convert("RGB")) for p, _ in test_samples])  # [N,3,224,224] in [0,1]
labels = torch.tensor([y for _, y in test_samples])


# ---------- Corruptions (applied in [0,1] pixel space) ----------
def jpeg(x, quality):
    out = []
    for img in x:
        buf = io.BytesIO()
        TF.to_pil_image(img).save(buf, format="JPEG", quality=quality)
        out.append(TF.to_tensor(Image.open(buf).convert("RGB")))
    return torch.stack(out)

CORRUPTIONS = {
    "gaussian_noise": ([0.05, 0.10, 0.20], lambda x, s: x + s * torch.randn_like(x)),
    "gaussian_blur":  ([1.0, 2.0, 4.0],    lambda x, s: TF.gaussian_blur(x, kernel_size=int(2 * round(3 * s) + 1), sigma=s)),
    "brightness":     ([0.8, 0.6, 0.4],    lambda x, s: TF.adjust_brightness(x, s)),
    "contrast":       ([0.8, 0.6, 0.4],    lambda x, s: TF.adjust_contrast(x, s)),
    "rotation":       ([10, 20, 30],       lambda x, s: TF.rotate(x, s)),
    "jpeg":           ([50, 20, 10],       lambda x, s: jpeg(x, s)),
}


# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(x, k=5):
    feats, logits = [], []
    for batch in x.split(32):
        f = model(TF.normalize(batch.clamp(0, 1), MEAN, STD))
        feats.append(f); logits.append(classifier_fc(f))
    feats, logits = torch.cat(feats), torch.cat(logits)
    acc = (logits.argmax(1) == labels).float().mean().item()
    glioma_recall = (logits.argmax(1)[labels == 0] == 0).float().mean().item()

    def p_at_k(q, g):
        idx = (q @ g.T).topk(k, dim=1).indices
        return (ytr[idx] == labels[:, None]).float().mean().item()

    return {
        "accuracy": acc,
        "glioma_recall": glioma_recall,
        "p@5_frozen": p_at_k(F.normalize(feats, dim=1), gallery_frozen),
        "p@5_supcon": p_at_k(head(feats), gallery_supcon),
    }


rows = [{"corruption": "clean", "severity": 0, **evaluate(images)}]
print(f"clean: {rows[0]}")
for name, (levels, fn) in CORRUPTIONS.items():
    for level_idx, s in enumerate(levels, start=1):
        r = evaluate(fn(images, s))
        rows.append({"corruption": name, "severity": level_idx, "param": s, **r})
        print(f"{name:15s} sev {level_idx} ({s}): acc {r['accuracy']:.3f}  "
              f"P@5 frozen {r['p@5_frozen']:.3f}  P@5 supcon {r['p@5_supcon']:.3f}")

df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/robustness.csv", index=False)

# ---------- Summary: average drop per corruption ----------
clean = df.iloc[0]
metrics = ["accuracy", "glioma_recall", "p@5_frozen", "p@5_supcon"]
summary = (df[df.corruption != "clean"].groupby("corruption")[metrics].mean() - clean[metrics].astype(float)) * 100
print("\n=== Average drop vs clean (percentage points, mean over 3 severities) ===")
print(summary.round(1).to_string())
print("\nClean:", {m: round(float(clean[m]), 3) for m in metrics})

# ---------- Plot ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
for ax, metric, title in zip(axes, ["accuracy", "p@5_frozen", "p@5_supcon"],
                             ["Classifier accuracy", "Retrieval P@5 (frozen ResNet)", "Retrieval P@5 (+ SupCon head)"]):
    for name in CORRUPTIONS:
        sub = df[df.corruption == name]
        ax.plot([0] + sub.severity.tolist(), [clean[metric]] + sub[metric].tolist(), marker="o", label=name)
    ax.set_title(title); ax.set_xlabel("Severity"); ax.set_xticks([0, 1, 2, 3]); ax.grid(alpha=0.3)
axes[0].set_ylabel("Score"); axes[-1].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(f"{RESULTS}/robustness.png", dpi=150); plt.close()

# ---------- Example grid for README ----------
sample = images[:1]
fig, axes = plt.subplots(len(CORRUPTIONS), 4, figsize=(9, 2.3 * len(CORRUPTIONS)))
for r, (name, (levels, fn)) in enumerate(CORRUPTIONS.items()):
    axes[r, 0].imshow(sample[0].permute(1, 2, 0)); axes[r, 0].set_ylabel(name, fontsize=9)
    for c, s in enumerate(levels, start=1):
        axes[r, c].imshow(fn(sample, s)[0].clamp(0, 1).permute(1, 2, 0))
        if r == 0: axes[r, c].set_title(f"severity {c}", fontsize=9)
    if r == 0: axes[r, 0].set_title("clean", fontsize=9)
    for ax in axes[r]: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(f"{RESULTS}/corruption_examples.png", dpi=120); plt.close()

print(f"\nSaved: {RESULTS}/robustness.csv, robustness.png, corruption_examples.png")
