"""
MRI Assistant = multimodal RAG over one brain MRI.

  image ──► ResNet18 classifier ──► prediction + confidence
        ├─► Grad-CAM heatmap      ──► where the model looked
        └─► SupCon embedding      ──► 5 most similar training cases (image retrieval)
  question ──► hybrid search over knowledge_base/ (text retrieval)
  all of the above ──► local LLM via Ollama ──► grounded answer with [1], [2] citations

Educational demo only - NOT a diagnostic tool.

Usage:
    from mri_assistant import MRIAssistant
    assistant = MRIAssistant()
    analysis = assistant.analyze(Image.open("scan.jpg"))
    reply = assistant.ask("What is this tumor type and how is it treated?", analysis)
    print(reply["answer"], reply["sources"])
"""
import json
import re
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from kb_search import HybridSearcher
from PIL import Image
from torchvision import models, transforms

CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
GERMAN_HINTS = {"der", "die", "das", "und", "ist", "wie", "was", "wird", "ein", "eine", "nicht",
                "mit", "warum", "welche", "dieses", "diese", "sind", "kann", "wer"}
READABLE = {"glioma_tumor": "glioma", "meningioma_tumor": "meningioma",
            "no_tumor": "no tumor", "pituitary_tumor": "pituitary tumor"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

SYSTEM_PROMPT = """You are an educational assistant in a brain MRI research demo.
Rules:
- You are NOT a doctor and this tool is NOT a diagnostic device. Never state a diagnosis as fact.
- Use ONLY the numbered context passages (and the model analysis, if one is given). Do not use outside knowledge.
- If the passages do not contain the answer, reply only: "The knowledge base does not cover this question."
  (in German: "Die Wissensbasis deckt diese Frage nicht ab.") Do not guess numbers, doses or statistics.
- Cite passages inline like [1] or [2] after the sentences that use them.
- When talking about the model's prediction, mention its confidence and whether similar cases agree.
- If the analysis shows a warning, say clearly that the prediction is unreliable.
- Answer directly. Do not describe your reasoning. At most 150 words."""


class MRIAssistant:
    def __init__(self, root="..", ckpt="best_brain_mri_resnet18_augmented.pth",
                 llm_model="qwen3:4b-instruct", ollama_url="http://localhost:11434", top_k_cases=5):
        self.root = root
        self.llm_model = llm_model
        self.ollama_url = ollama_url.rstrip("/")
        self.top_k_cases = top_k_cases
        self.device = torch.device("cpu")

        # ---- classifier (keeps its fc layer so we get logits + features in one pass) ----
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, len(CLASSES))
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()

        # hooks: layer4 activations/gradients for Grad-CAM, avgpool output = 512-d embedding
        self._acts, self._grads, self._feat = None, None, None
        self.model.layer4.register_forward_hook(self._save_acts)
        self.model.avgpool.register_forward_hook(lambda m, i, o: setattr(self, "_feat", o.flatten(1)))

        # ---- SupCon head + gallery for similar-case retrieval ----
        self.head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128))
        state = torch.load(f"{root}/models/supcon_head.pth", map_location=self.device)
        self.head.load_state_dict({k.replace("net.", ""): v for k, v in state.items()})
        self.head.eval()
        self.gallery = F.normalize(torch.tensor(
            np.load(f"{root}/models/brain_mri_embeddings_supcon.npy"), dtype=torch.float32), dim=1)
        self.metadata = pd.read_csv(f"{root}/models/brain_mri_metadata.csv")
        # paths were saved on Windows ("Training\\glioma_tumor\\x.jpg"); make them work on Linux/Docker too
        self.metadata["path"] = self.metadata["path"].str.replace("\\", "/", regex=False)

        # ---- text knowledge base ----
        self.kb = HybridSearcher.load(f"{root}/models/kb_index.pkl")

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    # ------------------------------------------------------------------ image side
    def _save_acts(self, module, inp, out):
        self._acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def analyze(self, image: Image.Image):
        """Classify, explain (Grad-CAM) and retrieve similar cases for one MRI image."""
        image = image.convert("RGB")
        x = self.preprocess(image).unsqueeze(0).requires_grad_(False)

        # forward WITH gradients (needed for Grad-CAM)
        self.model.zero_grad()
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach()
        pred_idx = int(probs.argmax())
        logits[0, pred_idx].backward()

        # Grad-CAM: weight each layer4 channel by its average gradient
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # similar cases via SupCon embedding
        with torch.no_grad():
            z = F.normalize(self.head(self._feat.detach()), dim=1)
            sims = (z @ self.gallery.T)[0]
            scores, idx = sims.topk(self.top_k_cases)
        cases = self.metadata.iloc[idx.numpy()].copy().reset_index(drop=True)
        cases["similarity"] = scores.numpy().round(3)

        vote_counts = cases["label"].value_counts()
        vote_label, vote_share = vote_counts.index[0], vote_counts.iloc[0] / len(cases)
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        warnings = []
        if confidence < 0.70:
            warnings.append(f"Low classifier confidence ({confidence:.0%}).")
        if vote_label != pred_label:
            warnings.append(f"Similar cases mostly show {READABLE[vote_label]}, "
                            f"but the classifier predicts {READABLE[pred_label]}.")
        if pred_label in ("meningioma_tumor", "no_tumor"):
            warnings.append("This model often mislabels gliomas as meningioma or no tumor "
                            "(glioma recall only about 24% on the test set).")

        return {
            "prediction": pred_label,
            "prediction_readable": READABLE[pred_label],
            "confidence": confidence,
            "probabilities": {READABLE[c]: float(p) for c, p in zip(CLASSES, probs)},
            "similar_cases": cases,
            "neighbor_vote": READABLE[vote_label],
            "neighbor_agreement": float(vote_share),
            "warnings": warnings,
            "gradcam": cam.numpy(),
            "gradcam_overlay": self.overlay(image, cam.numpy()),
        }

    @staticmethod
    def overlay(image, cam, alpha=0.45):
        """Blend a jet-colored heatmap onto the resized MRI. Returns uint8 RGB array."""
        import matplotlib.cm as cm
        base = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32) / 255.0
        heat = cm.jet(cam)[..., :3]
        return (np.clip((1 - alpha) * base + alpha * heat, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def summarize(analysis):
        """Plain-text summary of the image analysis that is passed to the LLM."""
        probs = ", ".join(f"{k} {v:.0%}" for k, v in
                          sorted(analysis["probabilities"].items(), key=lambda kv: -kv[1]))
        cases = ", ".join(f"{READABLE[lab]} ({s:.2f})" for lab, s in
                          zip(analysis["similar_cases"]["label"], analysis["similar_cases"]["similarity"]))
        lines = [
            f"Classifier prediction: {analysis['prediction_readable']} "
            f"(confidence {analysis['confidence']:.0%}). All class probabilities: {probs}.",
            f"Top {len(analysis['similar_cases'])} similar training cases (label, cosine similarity): {cases}.",
            f"Similar-case vote: {analysis['neighbor_vote']} ({analysis['neighbor_agreement']:.0%} agreement).",
        ]
        if analysis["warnings"]:
            lines.append("WARNINGS: " + " ".join(analysis["warnings"]))
        return "\n".join(lines)

    # ------------------------------------------------------------------ text side
    def retrieve(self, question, analysis=None, k=4):
        """Hybrid search; if an image was analyzed, also search for the predicted tumor type."""
        groups = []
        if analysis is not None and analysis["prediction"] != "no_tumor":
            # passages about the predicted tumor type get priority
            groups.append(self.kb.search(f"{question} {analysis['prediction_readable']}", k=2))
        groups.append(self.kb.search(question, k=3))
        if analysis is not None and analysis["warnings"]:
            groups.append(self.kb.search("How to interpret a prediction model reliability", k=1))

        # round-robin over groups so every group contributes, then de-duplicate
        seen, unique = set(), []
        for rank in range(3):
            for g in groups:
                if rank < len(g) and g[rank]["chunk_id"] not in seen:
                    seen.add(g[rank]["chunk_id"]); unique.append(g[rank])
        return unique[:k]

    @staticmethod
    def is_german(text):
        words = set(re.findall(r"[a-zäöüß]+", text.lower()))
        return bool(re.search(r"[äöüß]", text.lower())) or len(words & GERMAN_HINTS) >= 2

    def build_prompt(self, question, analysis, passages):
        context = "\n\n".join(f"[{i}] {p['title']} - {p['section']} (source: {p['source']})\n{p['text']}"
                              for i, p in enumerate(passages, start=1))
        parts = []
        if analysis:                         # only mention an image when there is one (it distracts otherwise)
            parts.append(f"MODEL ANALYSIS OF THE UPLOADED MRI:\n{self.summarize(analysis)}")
        parts.append(f"CONTEXT PASSAGES:\n{context}")
        parts.append(f"QUESTION: {question}")
        instructions = ["Answer in German." if self.is_german(question) else "Answer in English."]
        if analysis:
            instructions.append("End with one sentence recommending that a radiologist reviews the scan.")
        parts.append(" ".join(instructions))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ LLM (Ollama)
    OPTIONS = {"temperature": 0.2,
               "num_ctx": 2048,          # smaller context = faster prompt reading on CPU
               "num_predict": 350}       # cap answer length (~250 words)

    @staticmethod
    def _qwen3_no_think_prompt(messages):
        """Qwen3's official non-thinking chat template: the assistant turn starts with an EMPTY
        <think></think> block, so the model skips reasoning. Sent in raw mode, this works on every
        Ollama version (older versions ignore the "think": false flag and the /no_think switch)."""
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
        return "\n".join(parts) + "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def _request(self, messages):
        if "qwen3" in self.llm_model.lower() and "instruct" not in self.llm_model.lower():
            # hybrid Qwen3 builds: force the non-thinking template
            url = f"{self.ollama_url}/api/generate"
            body = {"model": self.llm_model, "prompt": self._qwen3_no_think_prompt(messages), "raw": True,
                    "stream": True, "keep_alive": "30m",
                    "options": {**self.OPTIONS, "stop": ["<|im_end|>", "<|im_start|>"]}}
        else:
            url = f"{self.ollama_url}/api/chat"
            body = {"model": self.llm_model, "messages": messages, "stream": True, "think": False,
                    "keep_alive": "30m", "options": self.OPTIONS}
        return urllib.request.Request(url, data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"})

    @staticmethod
    def clean(text):
        """Remove reasoning blocks / word counts. Safe to call on partial (streaming) text."""
        if "</think>" in text:
            text = text.split("</think>")[-1]
        elif text.lstrip().startswith("<think>"):
            return ""                                      # still inside a reasoning block
        text = re.sub(r"[*_]*\(\s*word count[^)]*\)[*_]*", "", text, flags=re.I)
        return text.strip()

    def _stream_ollama(self, messages):
        """Yield the growing answer text token by token (works for /api/chat and /api/generate)."""
        raw = ""
        with urllib.request.urlopen(self._request(messages), timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                raw += chunk.get("response") or chunk.get("message", {}).get("content", "")
                yield self.clean(raw)
                if chunk.get("done"):
                    break

    def warmup(self):
        """Load the LLM into memory at startup so the first question is not extra slow."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/generate",
                                         data=json.dumps({"model": self.llm_model, "keep_alive": "30m"}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=120).read()
        except Exception:
            pass

    def ollama_available(self):
        try:
            with urllib.request.urlopen(f"{self.ollama_url}/api/tags", timeout=3) as r:
                names = [m["name"] for m in json.loads(r.read()).get("models", [])]
            wanted = self.llm_model if ":" in self.llm_model else self.llm_model + ":latest"
            return wanted in names
        except Exception:
            return False

    # ------------------------------------------------------------------ public API
    def _prepare(self, question, analysis, history):
        passages = self.retrieve(question, analysis, k=3)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or [])[-2:]:                    # last exchange only (keeps the prompt short)
            messages.append({"role": turn["role"], "content": turn["content"][:600]})
        messages.append({"role": "user", "content": self.build_prompt(question, analysis, passages)})
        sources = [{"n": i, "title": p["title"], "section": p["section"],
                    "source": p["source"], "url": p["url"]} for i, p in enumerate(passages, start=1)]
        return passages, messages, sources

    def _fallback(self, analysis, passages):
        text = ("(LLM not reachable - start Ollama with `ollama serve` and "
                f"`ollama pull {self.llm_model}`. Showing retrieved evidence instead.)\n\n")
        if analysis:
            text += self.summarize(analysis) + "\n\n"
        return text + "\n\n".join(f"[{i}] {p['section']}: {p['text'][:300]}..."
                                   for i, p in enumerate(passages, start=1))

    def ask_stream(self, question, analysis=None, history=None):
        """Generator: yields (partial_answer, sources) while the LLM is writing."""
        passages, messages, sources = self._prepare(question, analysis, history)
        try:
            for partial in self._stream_ollama(messages):
                yield partial, sources
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            yield self._fallback(analysis, passages), sources

    def ask(self, question, analysis=None, history=None):
        """Answer a question grounded in the image analysis + knowledge base (non-streaming)."""
        passages, messages, sources = self._prepare(question, analysis, history)
        try:
            answer = ""
            for answer in self._stream_ollama(messages):
                pass
            used_llm = True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            answer, used_llm = self._fallback(analysis, passages), False
        return {"answer": answer, "sources": sources, "used_llm": used_llm,
                "prompt": messages[-1]["content"]}
