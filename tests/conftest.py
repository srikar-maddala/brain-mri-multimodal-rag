"""Shared test fixtures.

The tests never download models or need the MRI dataset:
  - FakeEncoder replaces the sentence-transformers embedding model (deterministic, numpy only)
  - fake_project builds a tiny project folder with a randomly initialised ResNet18,
    a SupCon head, a 20-image gallery and a knowledge-base index
"""
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(ROOT, "knowledge_base")
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


class FakeEncoder:
    """Hashes character trigrams into a 256-d vector. Similar texts -> similar vectors."""

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            t = f"  {text.lower()}  "
            for j in range(len(t) - 2):
                out[i, zlib.crc32(t[j:j + 3].encode()) % 256] += 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)


@pytest.fixture
def searcher():
    from kb_search import HybridSearcher, load_chunks

    chunks = load_chunks(KB_DIR)
    s = HybridSearcher(chunks, None, "fake-encoder")
    s._encoder = FakeEncoder()
    s.embeddings = s._encoder.encode([c["indexed_text"] for c in chunks])
    return s


@pytest.fixture(scope="session")
def fake_project(tmp_path_factory):
    """<tmp>/models/..., <tmp>/Data/..., <tmp>/Notebooks/model.pth - same layout as the real project."""
    from kb_search import load_chunks

    root = tmp_path_factory.mktemp("project")
    (root / "models").mkdir()
    (root / "Notebooks").mkdir()
    torch.manual_seed(0)

    # classifier with random weights (architecture identical to the real one)
    clf = models.resnet18(weights=None)
    clf.fc = nn.Linear(512, 4)
    ckpt = root / "Notebooks" / "model.pth"
    torch.save(clf.state_dict(), ckpt)

    # SupCon head (same key names as the trained head: "net.0.weight", ...)
    head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128))
    torch.save({f"net.{k}": v for k, v in head.state_dict().items()}, root / "models" / "supcon_head.pth")

    # gallery: 5 small images per class; metadata paths use WINDOWS separators like the real file
    rng = np.random.default_rng(0)
    rows = []
    for label_id, cls in enumerate(CLASSES):
        folder = root / "Data" / "brain_tumor" / "Training" / cls
        folder.mkdir(parents=True)
        for i in range(5):
            Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(folder / f"{i}.jpg")
            rows.append({"path": f"{root}/Data/brain_tumor/Training\\{cls}\\{i}.jpg",
                         "label_id": label_id, "label": cls})
    pd.DataFrame(rows).to_csv(root / "models" / "brain_mri_metadata.csv", index=False)
    np.save(root / "models" / "brain_mri_embeddings_supcon.npy",
            rng.normal(size=(len(rows), 128)).astype(np.float32))

    # knowledge-base index built with the fake encoder
    chunks = load_chunks(KB_DIR)
    emb = FakeEncoder().encode([c["indexed_text"] for c in chunks])
    with open(root / "models" / "kb_index.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": emb, "model_name": "fake-encoder"}, f)
    return root, ckpt


@pytest.fixture(scope="session")
def assistant(fake_project):
    from mri_assistant import MRIAssistant

    root, ckpt = fake_project
    a = MRIAssistant(root=str(root), ckpt=str(ckpt), llm_model="qwen3:4b-instruct",
                     ollama_url="http://127.0.0.1:9")          # nothing listens there -> LLM fallback
    a.kb._encoder = FakeEncoder()
    return a


@pytest.fixture
def mri_image():
    rng = np.random.default_rng(1)
    return Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
