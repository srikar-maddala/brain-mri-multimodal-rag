"""
Hybrid search over the knowledge base: BM25 (keywords) + dense embeddings,
fused with Reciprocal Rank Fusion (RRF), optional cross-encoder reranking.

Usage:
    from kb_search import HybridSearcher
    searcher = HybridSearcher.build("../knowledge_base")      # or .load(path)
    hits = searcher.search("how is a pituitary tumor treated?", k=3)
"""
import glob
import os
import pickle
import re

import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-small"   # ~470 MB, CPU-friendly, supports German
# Alternative (better, bigger ~2.2 GB): "BAAI/bge-m3"
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

STOPWORDS = set("""a an the and or of to in on for is are was were be been by with as at from that this
these those it its what which who how why when do does can could should would i my me you your
about into than then there their them they not no""".split())


def _stem(t):
    # very light stemming so "prolactinomas" matches "prolactinoma", "tumors" matches "tumor"
    return t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith("ss") else t


def tokenize(text):
    return [_stem(t) for t in re.findall(r"[a-z0-9äöüß]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


# ---------------- Chunking ----------------
def parse_markdown(path):
    """Split a KB file into (metadata, sections). Front matter between --- lines."""
    raw = open(path, encoding="utf-8").read()
    meta = {}
    if raw.startswith("---"):
        _, fm, raw = raw.split("---", 2)
        for line in fm.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    meta.setdefault("title", os.path.basename(path))
    meta["doc_id"] = os.path.splitext(os.path.basename(path))[0]

    chunks = []
    for block in re.split(r"\n(?=## )", raw):
        block = block.strip()
        if not block.startswith("## "):
            continue                                      # skip the H1 title block
        heading, _, body = block.partition("\n")
        chunks.append({
            "doc_id": meta["doc_id"],
            "title": meta["title"],
            "section": heading.lstrip("# ").strip(),
            "text": body.strip(),
            "source": meta.get("source", ""),
            "url": meta.get("url", ""),
            "topic": meta.get("topic", ""),
        })
    return chunks


def load_chunks(kb_dir):
    chunks = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.md"))):
        chunks.extend(parse_markdown(path))
    for i, c in enumerate(chunks):
        c["chunk_id"] = i
        # Title + section heading are prepended so each chunk is self-contained
        c["indexed_text"] = f"{c['title']} - {c['section']}. {c['text']}"
    return chunks


# ---------------- Searcher ----------------
class HybridSearcher:
    def __init__(self, chunks, embeddings, model_name, use_reranker=False):
        self.chunks = chunks
        self.embeddings = embeddings                         # [N, d], L2-normalized
        self.model_name = model_name
        self.bm25 = BM25Okapi([tokenize(c["indexed_text"]) for c in chunks])
        self._encoder = None
        self._reranker = None
        self.use_reranker = use_reranker

    # --- models are loaded lazily so importing this file is fast ---
    @property
    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name, device="cpu")
        return self._encoder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(DEFAULT_RERANKER, device="cpu", max_length=512)
        return self._reranker

    @staticmethod
    def _prefix(model_name, texts, kind):
        # E5 models expect "query: " / "passage: " prefixes
        if "e5" in model_name.lower():
            return [f"{kind}: {t}" for t in texts]
        return texts

    @classmethod
    def build(cls, kb_dir, model_name=DEFAULT_EMBED_MODEL, use_reranker=False):
        chunks = load_chunks(kb_dir)
        obj = cls(chunks, None, model_name, use_reranker)
        texts = cls._prefix(model_name, [c["indexed_text"] for c in chunks], "passage")
        obj.embeddings = obj.encoder.encode(texts, normalize_embeddings=True, batch_size=16,
                                            show_progress_bar=True)
        return obj

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings,
                         "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path, use_reranker=False):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["chunks"], d["embeddings"], d["model_name"], use_reranker)

    # --- individual retrievers: return chunk indices, best first ---
    def bm25_rank(self, query, n=20):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:n])

    def dense_rank(self, query, n=20):
        q = self.encoder.encode(self._prefix(self.model_name, [query], "query"), normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        return list(np.argsort(-scores)[:n])

    @staticmethod
    def rrf(rankings, k=60, weights=None):
        """(Weighted) Reciprocal Rank Fusion: score = sum_i  w_i / (k + rank_i)."""
        weights = weights or [1.0] * len(rankings)
        scores = {}
        for ranking, w in zip(rankings, weights):
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    def bm25_coverage(self, query):
        """Share of query words that exist in the knowledge-base vocabulary (0..1).
        German questions over English documents score ~0: BM25 has nothing to match."""
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        return sum(t in self.bm25.idf for t in tokens) / len(tokens)

    def search(self, query, k=3, mode="adaptive", candidates=20, rerank=None):
        """mode: 'bm25' | 'dense' | 'hybrid' (plain RRF) | 'adaptive' (RRF with BM25 weighted
        by query-word coverage - the default). Returns list of chunk dicts with 'score'."""
        if mode == "bm25":
            ranked = self.bm25_rank(query, candidates)
        elif mode == "dense":
            ranked = self.dense_rank(query, candidates)
        elif mode == "hybrid":
            ranked = self.rrf([self.bm25_rank(query, candidates), self.dense_rank(query, candidates)])
        else:
            w_bm25 = self.bm25_coverage(query)
            ranked = self.rrf([self.bm25_rank(query, candidates), self.dense_rank(query, candidates)],
                              weights=[w_bm25, 1.0])

        rerank = self.use_reranker if rerank is None else rerank
        if rerank:
            pool = ranked[:candidates]
            scores = self.reranker.predict([(query, self.chunks[i]["indexed_text"]) for i in pool])
            ranked = [pool[i] for i in np.argsort(-scores)]
            final_scores = sorted(scores, reverse=True)
        else:
            final_scores = [None] * len(ranked)

        results = []
        for idx, s in zip(ranked[:k], final_scores[:k]):
            c = dict(self.chunks[idx])
            c["score"] = None if s is None else float(s)
            results.append(c)
        return results
