"""tests/test_phase4_embed.py — Phase 4 ChromaDB validation."""
from pathlib import Path
import pytest

ROOT       = Path(__file__).parent.parent
CHROMA_DIR = ROOT / "db" / "chroma_db"
COLLECTION = "zewail_campus"


class TestPhase4Output:

    def test_chroma_dir_exists(self):
        assert CHROMA_DIR.exists(), (
            f"{CHROMA_DIR} not found. Run: python phase4_chunk_and_embed.py"
        )

    def test_chroma_dir_not_empty(self):
        files = list(CHROMA_DIR.rglob("*"))
        assert len(files) > 0, "ChromaDB directory is empty"


class TestPhase4Collection:

    @pytest.fixture(scope="class")
    def collection(self):
        try:
            import chromadb
        except ImportError:
            pytest.skip("chromadb not installed")
        db  = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return db.get_collection(COLLECTION)

    def test_collection_exists(self, collection):
        assert collection is not None

    def test_collection_has_vectors(self, collection):
        count = collection.count()
        assert count >= 50, f"Only {count} vectors — expected at least 50"

    def test_basic_query_returns_results(self, collection):
        # Use get() — collection uses OpenAI 1536-dim embeddings so query_texts
        # (which invokes ChromaDB's built-in 384-dim embedder) would fail.
        results = collection.get(limit=3, include=["documents", "metadatas"])
        docs = results["documents"]
        assert len(docs) >= 1, "Collection returned no documents"

    def test_metadata_fields_present(self, collection):
        results = collection.get(limit=10, include=["metadatas"])
        required = {"source", "source_type", "category", "doc_id"}
        for meta in results["metadatas"]:
            missing = required - set(meta.keys())
            assert not missing, f"Metadata missing fields: {missing}"

    def test_categories_in_collection(self, collection):
        """Collection should contain chunks from multiple categories."""
        results = collection.get(limit=200, include=["metadatas"])
        cats = {m.get("category") for m in results["metadatas"]}
        assert len(cats) >= 3, f"Only {len(cats)} categories in collection: {cats}"
