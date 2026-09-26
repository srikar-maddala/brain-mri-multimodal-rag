"""Tests for the multimodal assistant: image analysis, prompt building, LLM plumbing."""
import numpy as np
from mri_assistant import MRIAssistant


# ---------------------------------------------------------------- pure helpers
def test_clean_strips_reasoning_and_word_count():
    raw = "<think>\nlong reasoning\n</think>\n\nThe answer [1].\n\n*(Word count: 4)*"
    assert MRIAssistant.clean(raw) == "The answer [1]."
    assert MRIAssistant.clean("<think>still thinking") == ""        # partial stream inside reasoning
    assert MRIAssistant.clean("Plain answer.") == "Plain answer."


def test_is_german():
    assert MRIAssistant.is_german("Wie wird ein Meningeom behandelt?")
    assert MRIAssistant.is_german("Warum verursachen Hypophysentumoren Sehstörungen?")
    assert not MRIAssistant.is_german("How is a meningioma treated?")


def test_qwen3_prompt_disables_thinking():
    prompt = MRIAssistant._qwen3_no_think_prompt([{"role": "system", "content": "S"},
                                                  {"role": "user", "content": "Q"}])
    assert prompt.startswith("<|im_start|>system\nS<|im_end|>")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_endpoint_depends_on_model(assistant):
    msgs = [{"role": "user", "content": "hi"}]
    assistant.llm_model = "qwen3:4b-instruct"
    assert assistant._request(msgs).full_url.endswith("/api/chat")
    assistant.llm_model = "qwen3:4b"                                 # hybrid thinking model
    assert assistant._request(msgs).full_url.endswith("/api/generate")
    assistant.llm_model = "qwen3:4b-instruct"


# ---------------------------------------------------------------- image side
def test_analyze_output(assistant, mri_image):
    a = assistant.analyze(mri_image)
    assert a["prediction"] in ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
    assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-4
    assert 0.0 <= a["confidence"] <= 1.0
    assert a["gradcam"].shape == (224, 224)
    assert a["gradcam"].min() >= 0.0 and a["gradcam"].max() <= 1.0
    assert a["gradcam_overlay"].shape == (224, 224, 3) and a["gradcam_overlay"].dtype == np.uint8
    assert len(a["similar_cases"]) == 5
    assert 0.0 < a["neighbor_agreement"] <= 1.0


def test_windows_paths_are_normalised(assistant):
    assert not assistant.metadata["path"].str.contains("\\\\", regex=False).any()


def test_low_confidence_triggers_warning(assistant, mri_image):
    a = assistant.analyze(mri_image)
    if a["confidence"] < 0.70:                       # random weights -> usually low confidence
        assert any("confidence" in w.lower() for w in a["warnings"])


# ---------------------------------------------------------------- text side
def test_retrieve_returns_unique_passages(assistant, mri_image):
    a = assistant.analyze(mri_image)
    passages = assistant.retrieve("How is this tumor treated?", a, k=3)
    assert 1 <= len(passages) <= 3
    assert len({p["chunk_id"] for p in passages}) == len(passages)


def test_prompt_without_image_does_not_mention_scan(assistant):
    passages = assistant.retrieve("What is a glioma?", None, k=2)
    prompt = assistant.build_prompt("What is a glioma?", None, passages)
    assert "MODEL ANALYSIS" not in prompt
    assert "Answer in English." in prompt
    assert "radiologist" not in prompt


def test_prompt_with_image_and_german_question(assistant, mri_image):
    a = assistant.analyze(mri_image)
    q = "Wie wird das behandelt?"
    prompt = assistant.build_prompt(q, a, assistant.retrieve(q, a, k=2))
    assert "MODEL ANALYSIS" in prompt
    assert "Answer in German." in prompt
    assert "radiologist" in prompt


def test_ask_falls_back_when_llm_is_unreachable(assistant):
    reply = assistant.ask("What is a meningioma?")
    assert reply["used_llm"] is False
    assert "LLM not reachable" in reply["answer"]
    assert reply["sources"] and all("title" in s for s in reply["sources"])
    assert assistant.ollama_available() is False
