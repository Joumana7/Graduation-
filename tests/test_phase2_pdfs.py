"""tests/test_phase2_pdfs.py — Phase 2 output validation."""
import json
from pathlib import Path
import pytest

ROOT      = Path(__file__).parent.parent
PDF_DIR   = ROOT / "data" / "raw" / "pdfs"
PDF_JSONL = ROOT / "data" / "raw" / "pdf_raw.jsonl"


class TestPhase2Output:

    def test_pdf_raw_jsonl_exists(self):
        assert PDF_JSONL.exists(), (
            f"{PDF_JSONL} not found. Run: python phase2_extract_pdfs.py"
        )

    def test_pdf_dir_exists(self):
        assert PDF_DIR.exists(), f"{PDF_DIR} not found."

    def test_pdf_raw_at_least_created(self):
        # It's okay if zero PDFs were found (site may serve them behind auth)
        # but the file must exist
        assert PDF_JSONL.exists()


class TestPhase2Schema:

    @pytest.fixture(scope="class")
    def records(self):
        if not PDF_JSONL.exists():
            return []
        recs = []
        with open(PDF_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
        return recs

    def test_required_fields_present(self, records):
        if not records:
            pytest.skip("No PDF records found")
        required = {"filename", "page_number", "raw_text", "school",
                    "doc_type", "year", "extracted_at"}
        for rec in records:
            missing = required - set(rec.keys())
            assert not missing, f"Record missing fields {missing}"

    def test_page_numbers_are_positive(self, records):
        if not records:
            pytest.skip("No PDF records")
        for rec in records:
            assert rec["page_number"] >= 1, \
                f"page_number must be >= 1, got {rec['page_number']}"

    def test_school_values_valid(self, records):
        if not records:
            pytest.skip("No PDF records")
        valid = {"CSAI", "SCI", "BUS", "general"}
        for rec in records:
            assert rec["school"] in valid, \
                f"Unknown school: {rec['school']} in {rec['filename']}"

    def test_doc_type_values_valid(self, records):
        if not records:
            pytest.skip("No PDF records")
        valid = {"curricula", "catalog", "policy", "handbook",
                 "course_table", "curricula_table", "catalog_table",
                 "policy_table", "handbook_table"}
        for rec in records:
            assert rec["doc_type"] in valid, \
                f"Unknown doc_type: {rec['doc_type']} in {rec['filename']}"

    def test_multiple_schools_present(self, records):
        if not records:
            pytest.skip("No PDF records")
        schools = {rec["school"] for rec in records}
        assert len(schools) >= 3, f"Expected at least 3 schools, got: {schools}"

    def test_raw_text_non_empty(self, records):
        if not records:
            pytest.skip("No PDF records")
        for rec in records:
            assert len(rec["raw_text"].strip()) >= 10, \
                f"Empty page {rec['page_number']} of {rec['filename']}"

    def test_filenames_are_strings(self, records):
        if not records:
            pytest.skip("No PDF records")
        for rec in records:
            assert isinstance(rec["filename"], str) and rec["filename"]

    def test_downloaded_pdfs_are_valid(self):
        pdfs = list(PDF_DIR.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDFs downloaded")
        for pdf in pdfs:
            assert pdf.stat().st_size > 1000, f"{pdf.name} appears corrupt (< 1 KB)"
