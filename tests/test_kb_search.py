"""Tests for the knowledge-base chunking and hybrid search."""
from conftest import KB_DIR
from kb_search import HybridSearcher, load_chunks, tokenize


def test_tokenize_removes_stopwords_and_stems_plurals():
    assert tokenize("What are the tumors?") == ["tumor"]
    assert tokenize("prolactinomas") == ["prolactinoma"]
    assert "class" in tokenize("class")                    # words ending in "ss" are kept


def test_chunks_are_one_per_section_with_metadata():
    chunks = load_chunks(KB_DIR)
    assert len(chunks) >= 30
    for c in chunks:
        assert c["text"] and c["section"] and c["doc_id"]
        assert c["indexed_text"].startswith(c["title"])     # title is prepended for context
    doc_ids = {c["doc_id"] for c in chunks}
    assert {"glioma", "meningioma", "pituitary_tumors", "brain_mri", "model_card"} <= doc_ids


def test_every_chunk_cites_a_source():
    for c in load_chunks(KB_DIR):
        assert c["source"], f"{c['doc_id']} has no source"


def test_rrf_rewards_agreement_between_rankers():
    # worked example: A is 1st and 3rd, B is 5th and 1st, C only appears once
    bm25 = ["A", "x1", "x2", "x3", "B"]
    dense = ["B", "C", "A"]
    fused = HybridSearcher.rrf([bm25, dense])
    assert [x for x in fused if x in {"A", "B", "C"}] == ["A", "B", "C"]
    assert fused[0] == "A"                                  # found by both rankers -> top


def test_weighted_rrf_with_zero_weight_equals_other_ranking():
    bm25, dense = ["x", "y", "z"], ["z", "y", "x"]
    assert HybridSearcher.rrf([bm25, dense], weights=[0.0, 1.0]) == dense


def test_bm25_coverage_is_zero_for_german_questions(searcher):
    assert searcher.bm25_coverage("Wie wird ein Meningeom behandelt?") == 0.0
    assert searcher.bm25_coverage("How is a meningioma treated?") > 0.5


def test_search_modes_return_k_results(searcher):
    for mode in ["bm25", "dense", "hybrid", "adaptive"]:
        hits = searcher.search("How is a pituitary tumor treated?", k=3, mode=mode)
        assert len(hits) == 3
        assert all("section" in h and "source" in h for h in hits)


def test_bm25_finds_exact_medical_term(searcher):
    top = searcher.search("prolactinoma", k=1, mode="bm25")[0]
    assert top["doc_id"] == "pituitary_tumors"
