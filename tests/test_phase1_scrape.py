"""tests/test_phase1_scrape.py — Phase 1 output validation."""
import json
from pathlib import Path
import pytest

ROOT      = Path(__file__).parent.parent
WEB_JSONL = ROOT / "data" / "raw" / "web_raw.jsonl"

VALID_CATEGORIES = {
    "courses", "admissions", "policy", "deadlines",
    "facilities", "faculty", "research", "general",
}


# ── File existence ─────────────────────────────────────────────────────────────

class TestPhase1Output:

    def test_output_file_exists(self):
        assert WEB_JSONL.exists(), (
            f"{WEB_JSONL} not found. Run: python phase1_scrape_website.py"
        )

    def test_output_file_not_empty(self):
        assert WEB_JSONL.stat().st_size > 0, "web_raw.jsonl is empty"

    def test_at_least_10_pages_scraped(self):
        count = sum(1 for line in WEB_JSONL.read_text(encoding="utf-8").splitlines() if line.strip())
        assert count >= 10, f"Only {count} pages scraped — expected at least 10"


# ── Record structure ───────────────────────────────────────────────────────────

class TestPhase1Schema:

    @pytest.fixture(scope="class")
    def records(self):
        recs = []
        with open(WEB_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        return recs

    def test_all_records_have_required_fields(self, records):
        required = {"url", "title", "raw_text", "scraped_at", "char_count", "category_hint"}
        for rec in records:
            missing = required - set(rec.keys())
            assert not missing, f"Record missing fields {missing}: {rec.get('url')}"

    def test_urls_are_strings_and_non_empty(self, records):
        for rec in records:
            assert isinstance(rec["url"], str) and rec["url"].startswith("http"), \
                f"Bad URL: {rec['url']}"

    def test_raw_text_is_non_empty(self, records):
        for rec in records:
            assert isinstance(rec["raw_text"], str) and len(rec["raw_text"]) >= 50, \
                f"raw_text too short for {rec['url']}: {len(rec['raw_text'])} chars"

    def test_char_count_matches_text_length(self, records):
        for rec in records:
            assert rec["char_count"] == len(rec["raw_text"]), \
                f"char_count mismatch for {rec['url']}"

    def test_category_hints_are_valid(self, records):
        for rec in records:
            assert rec["category_hint"] in VALID_CATEGORIES, \
                f"Invalid category '{rec['category_hint']}' for {rec['url']}"

    def test_no_duplicate_urls(self, records):
        urls = [rec["url"] for rec in records]
        assert len(urls) == len(set(urls)), \
            f"{len(urls) - len(set(urls))} duplicate URLs in web_raw.jsonl"

    def test_scraped_at_is_iso_format(self, records):
        import re
        iso_re = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        for rec in records:
            assert iso_re.search(rec["scraped_at"]), \
                f"scraped_at not ISO format: {rec['scraped_at']}"

    def test_domain_is_zewailcity(self, records):
        for rec in records:
            assert "zewailcity.edu.eg" in rec["url"], \
                f"Foreign domain in output: {rec['url']}"
