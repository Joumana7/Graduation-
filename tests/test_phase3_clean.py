"""tests/test_phase3_clean.py — Phase 3 output validation."""
import json
from pathlib import Path
import pytest

ROOT       = Path(__file__).parent.parent
CLEAN_JSONL = ROOT / "data" / "clean" / "cleaned_documents.jsonl"

VALID_CATEGORIES = {
    "courses", "admissions", "policy", "deadlines",
    "facilities", "faculty", "research", "general",
}
VALID_SOURCE_TYPES = {"web", "pdf"}


class TestPhase3Output:

    def test_output_file_exists(self):
        assert CLEAN_JSONL.exists(), (
            f"{CLEAN_JSONL} not found. Run: python phase3_clean_data.py"
        )

    def test_output_file_not_empty(self):
        assert CLEAN_JSONL.stat().st_size > 0

    def test_at_least_20_records(self):
        count = sum(1 for l in CLEAN_JSONL.read_text(encoding="utf-8").splitlines() if l.strip())
        assert count >= 20, f"Only {count} clean records"


class TestPhase3Schema:

    @pytest.fixture(scope="class")
    def records(self):
        recs = []
        with open(CLEAN_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        return recs

    def test_required_fields(self, records):
        required = {"doc_id", "source_type", "source", "page",
                    "category", "text", "char_count", "cleaned_at"}
        for rec in records:
            missing = required - set(rec.keys())
            assert not missing, f"Missing fields {missing} in {rec.get('doc_id')}"

    def test_source_types_valid(self, records):
        for rec in records:
            assert rec["source_type"] in VALID_SOURCE_TYPES, \
                f"Invalid source_type: {rec['source_type']}"

    def test_categories_valid(self, records):
        for rec in records:
            assert rec["category"] in VALID_CATEGORIES, \
                f"Invalid category: {rec['category']}"

    def test_text_minimum_length(self, records):
        for rec in records:
            assert len(rec["text"]) >= 80, \
                f"Text too short ({len(rec['text'])} chars) for {rec['doc_id']}"

    def test_char_count_accurate(self, records):
        for rec in records:
            assert rec["char_count"] == len(rec["text"]), \
                f"char_count mismatch for {rec['doc_id']}"

    def test_no_duplicate_doc_ids(self, records):
        ids = [rec["doc_id"] for rec in records]
        assert len(ids) == len(set(ids)), "Duplicate doc_ids found"

    def test_no_boilerplate_phrases(self, records):
        boilerplate = ["apply now!", "loading...", "skip to content"]
        for rec in records:
            t = rec["text"].lower()
            for bp in boilerplate:
                assert bp not in t, \
                    f"Boilerplate '{bp}' found in {rec['doc_id']}"

    def test_text_does_not_contain_html_tags(self, records):
        import re
        html_re = re.compile(r"<[a-z][\s\S]*?>", re.I)
        for rec in records:
            assert not html_re.search(rec["text"]), \
                f"HTML tags found in cleaned text for {rec['doc_id']}"

    def test_categories_cover_multiple_types(self, records):
        """At least 3 different categories should be present."""
        cats = {rec["category"] for rec in records}
        assert len(cats) >= 3, f"Only {len(cats)} categories: {cats}"
