"""
Step: Document RAG - build the knowledge-base index and evaluate retrieval.

Compares BM25 vs dense vs hybrid (RRF) [vs hybrid + reranker] on 30 labeled
questions (25 English, 5 German) using hit@1, hit@3 and MRR.

Run from the Notebooks folder:
    python 04_hybrid_search.py              # BM25 / dense / hybrid
    python 04_hybrid_search.py --rerank     # + cross-encoder reranker (extra ~2.3 GB download)
    python 04_hybrid_search.py --model BAAI/bge-m3

Outputs:
    ../models/kb_index.pkl
    ../results/kb_retrieval_eval.csv
"""
import argparse
import os
import time

import pandas as pd
from kb_search import DEFAULT_EMBED_MODEL, HybridSearcher

parser = argparse.ArgumentParser()
parser.add_argument("--kb", default="../knowledge_base")
parser.add_argument("--model", default=DEFAULT_EMBED_MODEL)
parser.add_argument("--rerank", action="store_true")
parser.add_argument("--min-hit3", type=float, default=None,
                    help="quality gate for CI: exit with an error if adaptive-hybrid hit@3 is below this")
args = parser.parse_args()

os.makedirs("../models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ---------- Build + save index ----------
t0 = time.time()
searcher = HybridSearcher.build(args.kb, model_name=args.model)
searcher.save("../models/kb_index.pkl")
print(f"Indexed {len(searcher.chunks)} chunks from {len(set(c['doc_id'] for c in searcher.chunks))} "
      f"documents with {args.model} in {time.time() - t0:.1f}s")

# ---------- Labeled evaluation questions ----------
# Each question lists the relevant (doc_id, section-heading-prefix) pairs.
QUESTIONS = [
    ("What is a glioma?", [("glioma", "What is a glioma")], "en"),
    ("Which cells do gliomas start from?", [("glioma", "What is a glioma")], "en"),
    ("What is the difference between low-grade and high-grade glioma?", [("glioma", "Glioma grades")], "en"),
    ("How is glioblastoma treated?", [("glioma", "Treatment of glioma")], "en"),
    ("What molecular markers are tested in a glioma biopsy?", [("glioma", "How glioma is diagnosed")], "en"),
    ("What affects survival in glioma patients?", [("glioma", "Prognosis of glioma")], "en"),
    ("Is a meningioma cancer?", [("meningioma", "What is a meningioma")], "en"),
    ("Are meningiomas more common in women?", [("meningioma", "Who gets meningioma")], "en"),
    ("Can a small meningioma just be monitored instead of operated?", [("meningioma", "Treatment of meningioma")], "en"),
    ("What is an atypical grade 2 meningioma?", [("meningioma", "Meningioma grades")], "en"),
    ("Why do pituitary tumors cause vision loss?", [("pituitary_tumors", "Symptoms of pituitary tumors")], "en"),
    ("What is a prolactinoma?", [("pituitary_tumors", "Functioning vs nonfunctioning")], "en"),
    ("How do surgeons reach the pituitary gland?", [("pituitary_tumors", "Treatment of pituitary tumors")], "en"),
    ("Difference between a microadenoma and a macroadenoma", [("pituitary_tumors", "Types of pituitary tumors")], "en"),
    ("Which hormone tests are done for a pituitary tumor?", [("pituitary_tumors", "How pituitary tumors are diagnosed")], "en"),
    ("What does gadolinium contrast do in an MRI?", [("brain_mri", "Contrast agent"), ("glioma", "How glioma is diagnosed")], "en"),
    ("Can I have an MRI if I have a metal implant?", [("brain_mri", "Preparing for a brain MRI")], "en"),
    ("Who reads my MRI scan results?", [("brain_mri", "Who interprets")], "en"),
    ("Can an MRI alone tell the exact tumor type?", [("brain_mri", "Limitations of MRI"), ("brain_tumors_overview", "How brain tumors are diagnosed")], "en"),
    ("What is a metastatic brain tumor?", [("brain_tumors_overview", "Primary vs secondary")], "en"),
    ("What is the most common symptom of a brain tumor?", [("brain_tumors_overview", "Symptoms of brain tumors")], "en"),
    ("How accurate is this model on the test set?", [("model_card", "Performance")], "en"),
    ("Why should I not trust a no tumor prediction on a noisy image?", [("model_card", "Robustness"), ("model_card", "How to interpret")], "en"),
    ("What data leakage was found in the dataset?", [("model_card", "Data leakage")], "en"),
    ("Is this tool approved for medical diagnosis?", [("model_card", "Intended use")], "en"),
    # German queries -> English documents (cross-lingual retrieval)
    ("Was ist ein Gliom?", [("glioma", "What is a glioma")], "de"),
    ("Wie wird ein Meningeom behandelt?", [("meningioma", "Treatment of meningioma")], "de"),
    ("Warum verursachen Hypophysentumoren Sehstörungen?", [("pituitary_tumors", "Symptoms of pituitary tumors")], "de"),
    ("Ist ein MRT mit Kontrastmittel gefährlich?", [("brain_mri", "Contrast agent")], "de"),
    ("Wie zuverlässig ist das Modell?", [("model_card", "Performance"), ("model_card", "Robustness"), ("model_card", "How to interpret")], "de"),
]


def is_relevant(chunk, relevant):
    return any(chunk["doc_id"] == d and chunk["section"].startswith(s) for d, s in relevant)


# sanity check: every label must match at least one chunk
for q, rel, _ in QUESTIONS:
    assert any(is_relevant(c, rel) for c in searcher.chunks), f"Label matches no chunk: {q} -> {rel}"

modes = [("BM25", "bm25", False), ("Dense", "dense", False), ("Hybrid (RRF)", "hybrid", False),
         ("Hybrid (adaptive)", "adaptive", False)]
if args.rerank:
    modes.append(("Adaptive + rerank", "adaptive", True))

rows = []
for label, mode, rerank in modes:
    t0 = time.time()
    for q, rel, lang in QUESTIONS:
        hits = searcher.search(q, k=10, mode=mode, rerank=rerank)
        ranks = [i + 1 for i, h in enumerate(hits) if is_relevant(h, rel)]
        first = ranks[0] if ranks else None
        rows.append({"method": label, "question": q, "lang": lang,
                     "hit@1": int(first == 1), "hit@3": int(first is not None and first <= 3),
                     "mrr": 1.0 / first if first else 0.0,
                     "top1": f"{hits[0]['doc_id']} / {hits[0]['section']}"})
    print(f"{label:16s} evaluated in {time.time() - t0:.1f}s")

df = pd.DataFrame(rows)
df.to_csv("../results/kb_retrieval_eval.csv", index=False)

order = [m[0] for m in modes]
print("\n=== Knowledge-base retrieval (30 questions) ===")
print(df.groupby("method")[["hit@1", "hit@3", "mrr"]].mean().loc[order].round(3).to_string())
print("\n=== By language (hit@3) ===")
print(df.pivot_table(index="method", columns="lang", values="hit@3").loc[order].round(3).to_string())

missed = df[(df.method == "Hybrid (adaptive)") & (df["hit@3"] == 0)]
if len(missed):
    print("\nAdaptive-hybrid misses (check these - bad label or real retrieval failure?):")
    for _, r in missed.iterrows():
        print(f"  - {r.question}  ->  got: {r.top1}")

# ---------- Demo ----------
print("\n=== Demo: 'How is a meningioma treated?' ===")
for h in searcher.search("How is a meningioma treated?", k=3):
    print(f"  [{h['doc_id']} / {h['section']}]  ({h['source']})")
print("\nSaved ../models/kb_index.pkl and ../results/kb_retrieval_eval.csv")

# ---------- Quality gate (used by the CI pipeline) ----------
if args.min_hit3 is not None:
    score = df[df.method == "Hybrid (adaptive)"]["hit@3"].mean()
    if score < args.min_hit3:
        raise SystemExit(f"QUALITY GATE FAILED: adaptive hybrid hit@3 = {score:.3f} < {args.min_hit3}")
    print(f"Quality gate passed: adaptive hybrid hit@3 = {score:.3f} >= {args.min_hit3}")
