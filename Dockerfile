# Brain MRI Multimodal RAG - app container (CPU only)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=7860

WORKDIR /app

# 1) Dependencies first, so Docker can cache this slow layer when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# 2) Bake the text-embedding model into the image (no download at startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# 3) Model artifacts needed at runtime (the dataset is NOT copied - it is mounted or optional)
COPY models/supcon_head.pth models/brain_mri_embeddings_supcon.npy \
     models/brain_mri_metadata.csv models/kb_index.pkl /app/models/
COPY knowledge_base/ /app/knowledge_base/
COPY test_images/ /app/test_images/
COPY Notebooks/best_brain_mri_resnet18_augmented.pth Notebooks/mri_assistant.py \
     Notebooks/kb_search.py Notebooks/app.py /app/Notebooks/

# 4) Run as a non-root user (security best practice)
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
WORKDIR /app/Notebooks

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860', timeout=4)"
CMD ["python", "app.py"]
