"""
Step: Multimodal RAG assistant - end-to-end test.

For one test MRI per class:
  1. classify + Grad-CAM + similar cases
  2. ask the assistant an English and a German question
Saves a Grad-CAM + similar-cases figure per class to ../results/assistant_<class>.png

Before running:
  - run 04_hybrid_search.py once (creates ../models/kb_index.pkl)
  - install Ollama, then:  ollama pull qwen3:4b-instruct
Run:  python 05_test_assistant.py            (or --llm llama3.2:3b)
"""
import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mri_assistant import READABLE, MRIAssistant
from PIL import Image
from torchvision import datasets

parser = argparse.ArgumentParser()
parser.add_argument("--llm", default="qwen3:4b-instruct")
parser.add_argument("--no-llm-questions", action="store_true", help="only test the image side")
args = parser.parse_args()
os.makedirs("../results", exist_ok=True)

t0 = time.time()
assistant = MRIAssistant(llm_model=args.llm)
print(f"Assistant loaded in {time.time() - t0:.1f}s")
if assistant.ollama_available():
    print(f"Ollama OK - using {args.llm}")
else:
    print(f"WARNING: Ollama or model '{args.llm}' not found -> answers will fall back to retrieved text.\n"
          f"         Fix: install Ollama, then run  ollama pull {args.llm}")

test = datasets.ImageFolder("../Data/brain_tumor/Testing")
QUESTIONS = [
    "What does the model see in this scan, and how reliable is that prediction?",
    "Wie wird diese Art von Tumor normalerweise behandelt?",   # German: how is this type of tumor usually treated?
]

for class_idx, class_name in enumerate(test.classes):
    path = next(p for p, y in test.samples if y == class_idx)
    image = Image.open(path)

    t0 = time.time()
    a = assistant.analyze(image)
    print("\n" + "=" * 80)
    print(f"TRUE LABEL: {READABLE[class_name]}   |   image: {os.path.basename(path)}   "
          f"|   analysis {time.time() - t0:.2f}s")
    print(assistant.summarize(a))

    # ---- figure: original | Grad-CAM | 5 similar cases ----
    cases = a["similar_cases"]
    fig, axes = plt.subplots(1, 2 + len(cases), figsize=(2.6 * (2 + len(cases)), 3.2))
    axes[0].imshow(image.convert("RGB")); axes[0].set_title(f"Query (true: {READABLE[class_name]})", fontsize=8)
    axes[1].imshow(a["gradcam_overlay"])
    axes[1].set_title(f"Grad-CAM\npred: {a['prediction_readable']} {a['confidence']:.0%}", fontsize=8)
    for ax, (_, r) in zip(axes[2:], cases.iterrows()):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(f"{READABLE[r['label']]}\nsim {r['similarity']:.2f}", fontsize=8,
                     color="green" if r["label"] == class_name else "red")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(f"../results/assistant_{class_name}.png", dpi=120); plt.close()

    if args.no_llm_questions:
        continue
    for q in QUESTIONS:
        t0 = time.time()
        reply = assistant.ask(q, a)
        print(f"\nQ: {q}\n[{time.time() - t0:.1f}s, LLM={'yes' if reply['used_llm'] else 'NO (fallback)'}]")
        print(reply["answer"])
        print("Sources: " + "; ".join(f"[{s['n']}] {s['title']} - {s['section']}" for s in reply["sources"]))

# ---- a question without any image ----
print("\n" + "=" * 80)
reply = assistant.ask("Is this tool approved to diagnose brain tumors?")
print("Q (no image): Is this tool approved to diagnose brain tumors?\n" + reply["answer"])
print("\nFigures saved to ../results/assistant_<class>.png")
