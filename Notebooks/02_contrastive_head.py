"""
Step: Contrastive learning -> Better retrieval
Trains a supervised-contrastive (SupCon) projection head on top of the frozen,
fine-tuned ResNet18 backbone, then compares retrieval before vs after.

Fair evaluation:
  gallery = Training images, queries = Testing images
  test images that are exact duplicates of training images (MD5) are removed.

Run from the same folder as your notebook (paths match your layout).
CPU-friendly: feature extraction ~5 min, head training ~1 min.
"""
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

torch.manual_seed(42); np.random.seed(42)
TRAIN_DIR = "../Data/brain_tumor/Training"
TEST_DIR = "../Data/brain_tumor/Testing"
CKPT = "best_brain_mri_resnet18_augmented.pth"
OUT = "../models"
device = torch.device("cpu")
os.makedirs(OUT, exist_ok=True)

# ---------- 1. Frozen backbone ----------
backbone = models.resnet18(weights=None)
backbone.fc = nn.Linear(512, 4)
backbone.load_state_dict(torch.load(CKPT, map_location=device))
backbone.fc = nn.Identity()
backbone.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])  # NOTE: must match the test_transform you trained with; drop Normalize if you didn't use it


def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@torch.no_grad()
def extract(ds):
    feats, labels = [], []
    for x, y in DataLoader(ds, batch_size=32, shuffle=False):
        feats.append(backbone(x)); labels.append(y)
    return torch.cat(feats), torch.cat(labels)


train_ds = datasets.ImageFolder(TRAIN_DIR, transform=tf)
test_ds = datasets.ImageFolder(TEST_DIR, transform=tf)

# Remove test images that also appear in training (data leakage)
train_hashes = {md5(p) for p, _ in train_ds.samples}
keep = [i for i, (p, _) in enumerate(test_ds.samples) if md5(p) not in train_hashes]
print(f"Test images: {len(test_ds)} -> {len(keep)} after removing duplicates")
test_ds = torch.utils.data.Subset(test_ds, keep)

cache = f"{OUT}/frozen_feats.pt"
if os.path.exists(cache):
    Xtr, ytr, Xte, yte = torch.load(cache)
else:
    print("Extracting features (one-time)...")
    Xtr, ytr = extract(train_ds)
    Xte, yte = extract(test_ds)
    torch.save((Xtr, ytr, Xte, yte), cache)


# ---------- 2. Retrieval metrics ----------
def retrieval_metrics(q, qy, g, gy, k=5):
    q, g = F.normalize(q, dim=1), F.normalize(g, dim=1)
    sims = q @ g.T
    order = sims.argsort(dim=1, descending=True)
    rel = (gy[order] == qy[:, None]).float()          # relevance matrix
    p_at_k = rel[:, :k].mean().item()
    # mean average precision
    cum = rel.cumsum(1)
    ranks = torch.arange(1, rel.shape[1] + 1).float()
    ap = ((cum / ranks) * rel).sum(1) / rel.sum(1).clamp(min=1)
    # per-class P@k
    per_class = {c: rel[qy == c, :k].mean().item() for c in qy.unique().tolist()}
    return p_at_k, ap.mean().item(), per_class


# ---------- 3. SupCon projection head ----------
class Head(nn.Module):
    def __init__(self, d_in=512, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, d_out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def supcon_loss(z, y, t=0.1):
    sim = z @ z.T / t
    self_mask = torch.eye(len(y), dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1).clamp(min=1)).mean()


# Hold out 20% of training features to pick the best epoch (never touch test for that)
perm = torch.randperm(len(Xtr)); n_val = len(Xtr) // 5
va_idx, tr_idx = perm[:n_val], perm[n_val:]

head = Head()
opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
best, best_state = -1, None
for epoch in range(30):
    head.train()
    for b in torch.randperm(len(tr_idx)).split(256):
        idx = tr_idx[b]
        x = Xtr[idx] + 0.05 * torch.randn_like(Xtr[idx])   # light feature noise = cheap augmentation
        loss = supcon_loss(head(x), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        _, val_map, _ = retrieval_metrics(head(Xtr[va_idx]), ytr[va_idx], head(Xtr[tr_idx]), ytr[tr_idx])
    if val_map > best:
        best, best_state = val_map, {k: v.clone() for k, v in head.state_dict().items()}
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  loss {loss.item():.4f}  val mAP {val_map:.4f}")

head.load_state_dict(best_state); head.eval()
torch.save(head.state_dict(), f"{OUT}/supcon_head.pth")

# ---------- 4. Before vs after (test queries -> train gallery) ----------
classes = datasets.ImageFolder(TRAIN_DIR).classes
with torch.no_grad():
    before = retrieval_metrics(Xte, yte, Xtr, ytr)
    after = retrieval_metrics(head(Xte), yte, head(Xtr), ytr)

print("\n=== Retrieval: test queries vs train gallery (duplicates removed) ===")
print(f"{'':22s}{'Frozen ResNet':>15s}{'+ SupCon head':>15s}")
print(f"{'P@5':22s}{before[0]:15.3f}{after[0]:15.3f}")
print(f"{'mAP':22s}{before[1]:15.3f}{after[1]:15.3f}")
for c in sorted(before[2]):
    print(f"{'P@5 ' + classes[c]:22s}{before[2][c]:15.3f}{after[2][c]:15.3f}")

# Save gallery embeddings for the app / RAG step
with torch.no_grad():
    np.save(f"{OUT}/brain_mri_embeddings_supcon.npy", head(Xtr).numpy())
print(f"\nSaved head + new gallery embeddings to {OUT}/")
