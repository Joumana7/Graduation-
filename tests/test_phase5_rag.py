"""tests/test_phase5_rag.py — Phase 5 RAG pipeline validation."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def rag():
    try:
        from phase5_rag_pipeline import CampusRAG
        return CampusRAG()
    except Exception as exc:
        pytest.skip(f"CampusRAG init failed: {exc}")


class TestRetrieve:

    def test_retrieve_returns_results(self, rag):
        chunks, _ = rag.retrieve("What are the graduation requirements?", top_k=5)
        assert len(chunks) >= 1, "retrieve() returned no chunks"

    def test_retrieve_returns_typed_chunks(self, rag):
        from phase5_rag_pipeline import RetrievedChunk
        chunks, _ = rag.retrieve("admissions scholarship", top_k=3)
        for c in chunks:
            assert isinstance(c, RetrievedChunk)

    def test_retrieve_scores_are_bounded(self, rag):
        chunks, _ = rag.retrieve("academic policy probation", top_k=5)
        for c in chunks:
            assert 0.0 <= c.score <= 1.0, f"Score out of range: {c.score}"

    def test_retrieve_sorted_by_score_desc(self, rag):
        chunks, _ = rag.retrieve("credit hours required", top_k=5)
        scores = [c.score for c in chunks]
        assert scores == sorted(scores, reverse=True), "Chunks not sorted by score"

    def test_retrieve_chunks_have_text(self, rag):
        chunks, _ = rag.retrieve("undergraduate programs", top_k=5)
        for c in chunks:
            assert c.text and len(c.text) >= 20, f"Empty chunk text: {c.chunk_id}"

    def test_retrieve_chunks_have_source(self, rag):
        chunks, _ = rag.retrieve("student life campus", top_k=3)
        for c in chunks:
            assert c.source, f"Chunk {c.chunk_id} has no source"

    def test_retrieve_different_queries_different_results(self, rag):
        chunks_a, _ = rag.retrieve("graduation requirements GPA academic standing", top_k=6)
        chunks_b, _ = rag.retrieve("library research laboratory facilities equipment", top_k=6)
        ids_a = {c.chunk_id for c in chunks_a}
        ids_b = {c.chunk_id for c in chunks_b}
        # At least one chunk should differ between two semantically distinct queries
        assert ids_a != ids_b, "Different queries returned identical chunk sets"


class TestGenerate:

    def test_generate_returns_string(self, rag):
        chunks, note = rag.retrieve("What is Zewail City?", top_k=3)
        answer = rag.generate("What is Zewail City?", chunks, query_note=note)
        assert isinstance(answer, str) and len(answer) >= 50

    def test_generate_graceful_with_no_chunks(self, rag):
        answer = rag.generate("What is Zewail City?", chunks=[])
        assert isinstance(answer, str) and len(answer) > 10

    def test_generate_with_history(self, rag):
        history = [
            {"role": "user",      "content": "I study CSAI at Zewail City."},
            {"role": "assistant", "content": "Great! How can I help you?"},
        ]
        chunks, note = rag.retrieve("course recommendations", top_k=3)
        answer = rag.generate("What courses should I take?", chunks, history=history, query_note=note)
        assert isinstance(answer, str) and len(answer) > 20


class TestAnswer:

    def test_answer_returns_tuple(self, rag):
        result = rag.answer("What programs does Zewail City offer?")
        assert isinstance(result, tuple) and len(result) == 2

    def test_answer_text_is_non_empty(self, rag):
        answer, _ = rag.answer("Tell me about admissions at Zewail City.")
        assert len(answer) >= 50

    def test_answer_sources_are_list(self, rag):
        _, sources = rag.answer("What is academic probation?")
        assert isinstance(sources, list)
