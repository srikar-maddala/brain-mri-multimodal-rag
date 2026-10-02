# CLAUDE.md

Explainable brain MRI assistant: ResNet18 classifier + Grad-CAM + SupCon similar-case retrieval + hybrid
text retrieval over `knowledge_base/` + local LLM (Qwen3-4B-Instruct via Ollama). CPU-only, offline.
Educational project, **not a diagnostic tool**. Keep that disclaimer in any user-facing text or prompt.

## Commands

```bash
pytest                                   # 19 tests; no dataset, GPU, model download or Ollama needed
ruff check .                             # line length 120, rules E/F/W/I (see pyproject.toml)

# Scripts must be run FROM Notebooks/ (they use relative paths like ../models, ../knowledge_base)
cd Notebooks
python 04_hybrid_search.py               # rebuild models/kb_index.pkl + results/kb_retrieval_eval.csv
python 04_hybrid_search.py --min-hit3 0.9  # CI quality gate
python app.py                            # Gradio app on http://127.0.0.1:7860 (needs Ollama for chat)
python 06_llm_eval.py                    # LLM eval suite, ~10 min on CPU, needs Ollama running

docker compose -f docker-compose.host-ollama.yml up --build   # app in Docker, Ollama on host
```

Install: CPU torch first (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`),
then `pip install -r requirements.txt`.

## Layout

- `Notebooks/mri_assistant.py` – `MRIAssistant`: `analyze()` (prediction, Grad-CAM via layer4 hooks,
  512-d avgpool embedding → SupCon head → cosine search over gallery), `retrieve()`, `build_prompt()`,
  `ask()` / `ask_stream()` against Ollama, with a non-LLM fallback when Ollama is unreachable.
  `SYSTEM_PROMPT` lives here; changes to it should be re-checked with `06_llm_eval.py`.
- `Notebooks/kb_search.py` – `HybridSearcher`: BM25 + `intfloat/multilingual-e5-small`, fused with
  query-adaptive weighted RRF (BM25 weight = share of query tokens in corpus vocab, so German queries
  rely on dense search). Index is pickled to `models/kb_index.pkl`.
- `Notebooks/app.py` – Gradio UI. Config via CLI or env: `LLM_MODEL`, `OLLAMA_URL`, `GRADIO_SERVER_NAME`, `PORT`.
- `Notebooks/0N_*.py` – numbered experiment/eval scripts; `01_dataset_exploration.ipynb` does training.
- `knowledge_base/*.md` – cited source docs with `---` front matter (`title`, `source`, `url`, `topic`).
  Each `## ` section becomes one chunk; text before the first `## ` is ignored. Rebuild the index after editing.
- `models/` – runtime artifacts (SupCon head, gallery embeddings, metadata CSV, KB index).
  Classifier checkpoint is `Notebooks/best_brain_mri_resnet18_augmented.pth`.
- `results/` – evaluation CSVs/figures referenced by the README.
- `tests/conftest.py` – `FakeEncoder` (deterministic, replaces sentence-transformers) and `fake_project`
  (tiny random ResNet18 + gallery + KB index). New tests should use these fixtures, not real models.

## Conventions and gotchas

- Class order is fixed: `["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]`.
- `models/brain_mri_metadata.csv` stores Windows paths; `MRIAssistant` converts `\` to `/` for Linux/Docker.
- The MRI dataset (`Data/brain_tumor/{Training,Testing}`) is not in the repo and is optional for the app.
- The Dockerfile copies specific files explicitly; add new runtime modules/artifacts to it.
- CI (`.github/workflows/ci.yml`): ruff → pytest → retrieval gate (hit@3 ≥ 0.9) → Docker build, pushed to GHCR on `main`.
- Reported numbers (69.3% test accuracy on the de-duplicated test set, etc.) come from `results/`; update the
  README if you re-run evaluations.
