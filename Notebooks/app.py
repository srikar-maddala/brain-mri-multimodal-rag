"""
Brain MRI Multimodal RAG Assistant - Gradio web app.

Upload an MRI -> prediction + Grad-CAM + 5 similar cases -> chat with a local LLM
that answers from the image analysis and a cited medical knowledge base.

Before running (from the Notebooks folder):
    pip install gradio
    python 04_hybrid_search.py        # once, builds ../models/kb_index.pkl
    ollama pull qwen3:4b-instruct              # local LLM (Ollama must be running)
Run:
    python app.py                     # opens http://127.0.0.1:7860
"""
import argparse
import glob
import os

import gradio as gr
from mri_assistant import READABLE, MRIAssistant
from PIL import Image, ImageDraw

# Settings can come from the command line or from environment variables (used by Docker)
parser = argparse.ArgumentParser()
parser.add_argument("--llm", default=os.getenv("LLM_MODEL", "qwen3:4b-instruct"))
parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
parser.add_argument("--host", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
parser.add_argument("--share", action="store_true", help="create a temporary public link")
args = parser.parse_args()

print("Loading models...")
assistant = MRIAssistant(llm_model=args.llm, ollama_url=args.ollama_url)
LLM_OK = assistant.ollama_available()
print(f"Ollama ({args.llm} at {args.ollama_url}): "
      f"{'OK' if LLM_OK else 'NOT FOUND - chat will show retrieved text only'}")
if LLM_OK:
    print("Loading the LLM into memory (one-time warm-up)...")
    assistant.warmup()


def find_examples():
    """One test image per class if the dataset is available, else the images in ../test_images."""
    test_dir = "../Data/brain_tumor/Testing"
    if os.path.isdir(test_dir):
        classes = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
        found = [sorted(glob.glob(os.path.join(test_dir, c, "*")))[:1] for c in classes]
        return [f for f in found if f]
    return [[p] for p in sorted(glob.glob("../test_images/*"))]


def load_case_image(path, caption):
    """Similar-case thumbnail; a labelled placeholder if the dataset is not mounted (e.g. in the cloud)."""
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    img = Image.new("RGB", (224, 224), (40, 40, 40))
    ImageDraw.Draw(img).text((12, 100), caption, fill=(230, 230, 230))
    return img


EXAMPLES = find_examples()

DISCLAIMER = ("⚠️ **Educational research demo, not a medical device.** Predictions can be wrong "
              "(test accuracy 69%, glioma recall about 24%). Real scans must be reviewed by a radiologist.")


# ------------------------------------------------------------------ callbacks
def run_analysis(image):
    if image is None:
        raise gr.Error("Please upload an MRI image first.")
    a = assistant.analyze(image)

    gallery = []
    for i, (_, r) in enumerate(a["similar_cases"].iterrows()):
        caption = f"#{i + 1} {READABLE[r['label']]} (sim {r['similarity']:.2f})"
        gallery.append((load_case_image(r["path"], caption), caption))

    status = (f"**Prediction:** {a['prediction_readable']} ({a['confidence']:.0%})  \n"
              f"**Similar-case vote:** {a['neighbor_vote']} ({a['neighbor_agreement']:.0%} agree)")
    if a["warnings"]:
        status += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in a["warnings"])
    else:
        status += "\n\n- ✅ Classifier and similar cases agree."

    greeting = [{"role": "assistant",
                 "content": f"I analyzed the scan. The model predicts **{a['prediction_readable']}** "
                            f"with {a['confidence']:.0%} confidence. Ask me what this means, how reliable "
                            f"it is, or how this tumor type is usually diagnosed and treated. "
                            f"You can also ask in German."}]
    return a["probabilities"], a["gradcam_overlay"], gallery, status, a, greeting, ""


def format_sources(sources):
    return "**Sources**\n\n" + "\n".join(
        f"- **[{s['n']}]** {s['title']} - *{s['section']}*"
        + (f" ([link]({s['url']}))" if s["url"].startswith("http") else " (project model card)")
        for s in sources)


def chat(message, history, analysis):
    """Streams the answer into the chat as the LLM writes it."""
    if not message.strip():
        yield history, "", gr.update()
        return
    llm_history = [{"role": m["role"], "content": m["content"]} for m in history[-2:]
                   if isinstance(m.get("content"), str)]
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "⏳ Reading the scan analysis and sources..."}]
    yield history, "", gr.update()

    for partial, sources in assistant.ask_stream(message, analysis, history=llm_history):
        if partial:
            history[-1] = {"role": "assistant", "content": partial}
            yield history, "", format_sources(sources)


def make_chatbot():
    try:                                   # Gradio 4/5 need type="messages"; Gradio 6 uses it by default
        return gr.Chatbot(type="messages", height=380, label="Assistant")
    except TypeError:
        return gr.Chatbot(height=380, label="Assistant")


# ------------------------------------------------------------------ layout
with gr.Blocks(title="Brain MRI Multimodal RAG") as demo:
    gr.Markdown("# 🧠 Brain MRI Multimodal RAG Assistant\n"
                "ResNet18 classifier · Grad-CAM · SupCon similar-case retrieval · "
                "hybrid text search (BM25 + multilingual embeddings) · local LLM via Ollama")
    gr.Markdown(DISCLAIMER)
    if not LLM_OK:
        gr.Markdown(f"🟡 Ollama model `{args.llm}` not found: the chat will show retrieved passages "
                    f"instead of generated answers. Run `ollama pull {args.llm}` and restart.")

    analysis_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(type="pil", label="Upload MRI", height=280)
            analyze_btn = gr.Button("Analyze scan", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=image_in,
                            label="Example test images (glioma, meningioma, no tumor, pituitary)")
        with gr.Column(scale=1):
            probs_out = gr.Label(label="Class probabilities", num_top_classes=4)
            status_out = gr.Markdown()
        with gr.Column(scale=1):
            cam_out = gr.Image(label="Grad-CAM (where the model looked)", height=280)

    cases_out = gr.Gallery(label="5 most similar training cases (image retrieval)", columns=5, height=220)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = make_chatbot()
            with gr.Row():
                msg = gr.Textbox(placeholder="e.g. How reliable is this prediction?  /  Wie wird das behandelt?",
                                 show_label=False, scale=5)
                send = gr.Button("Send", scale=1)
        with gr.Column(scale=1):
            sources_out = gr.Markdown("**Sources** appear here after each answer.")

    analyze_btn.click(run_analysis, inputs=image_in,
                      outputs=[probs_out, cam_out, cases_out, status_out, analysis_state, chatbot, sources_out])
    for trigger in (send.click, msg.submit):
        trigger(chat, inputs=[msg, chatbot, analysis_state], outputs=[chatbot, msg, sources_out])

if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
