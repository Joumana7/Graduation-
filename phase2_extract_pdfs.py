#!/usr/bin/env python3
"""
phase2_extract_pdfs.py  -  Zewail Campus Digital Assistant
Phase 2: Extract text from official Zewail City PDF handbooks.

PDF source: d:/FINAL_PROJECT_v3/FINAL_PROJECT/data/pdfs/
  (17 PDFs covering CSAI, SCI, BUS schools - curricula, catalogs, policies)

Strategy:
- Extract raw text per page using PyMuPDF
- For curricula pages containing course tables, also emit a structured
  natural-language summary of each table row (improves RAG accuracy)
- Tag every record with school + doc_type so Phase 3 can classify correctly

Output: data/raw/pdf_raw.jsonl
  Fields: filename, page_number, raw_text, school, doc_type, year
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("PyMuPDF not installed. Run: pip install pymupdf")

# ── Paths ──────────────────────────────────────────────────────────────────────

PDF_SRC_DIR = Path(r"D:\FINAL_PROJECT_v3\FINAL_PROJECT\data\pdfs")
OUT_DIR     = Path(__file__).parent / "data" / "raw"
OUT_FILE    = OUT_DIR / "pdf_raw.jsonl"
PDF_DIR     = OUT_DIR / "pdfs"

MIN_CHARS = 60   # skip near-empty pages

# ── Boilerplate patterns to strip from PDF pages ───────────────────────────────

_BOILER_RE = [
    re.compile(r"^\s*\d{15,}.*$", re.M),          # long doc-ID strings
    re.compile(r"^\s*Page \d+ of \d+\s*$", re.M), # "Page X of Y"
    re.compile(r"^\s*-[IVX]+-\s*$", re.M),         # -III-, -IV- etc.
    re.compile(r"^\s*\(continued\)\s*$", re.M | re.I),
]

def _strip_boilerplate(text: str) -> str:
    for pat in _BOILER_RE:
        text = pat.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Filename metadata classifier ───────────────────────────────────────────────

def classify_filename(name: str) -> dict:
    """Parse school, doc_type, year from PDF filename."""
    n = name.lower()

    if "csai" in n:
        school = "CSAI"
    elif "bus" in n:
        school = "BUS"
    elif "sci" in n:
        school = "SCI"
    else:
        school = "general"

    if "curricul" in n or "study plan" in n:
        doc_type = "curricula"
    elif "catalog" in n:
        doc_type = "catalog"
    elif "polic" in n or "regulation" in n:
        doc_type = "policy"
    else:
        doc_type = "handbook"

    m = re.search(r"(20\d{2})", name)
    year = m.group(1) if m else "unknown"

    return {"school": school, "doc_type": doc_type, "year": year}


# ── Course-row parser (for curricula tables) ───────────────────────────────────

# Matches course rows like: "CSAI 201   Data Structures   3  2  3  CSAI 151, ..."
_COURSE_ROW_RE = re.compile(
    r"^([A-Z]{2,6}\s+\d{3,4}[A-Z]?)\s{2,}(.+?)\s{2,}(\d)\s+(\d)\s+(\d|-)\s*(.*)?$"
)
# Matches semester headers like "Year 2 / Semester 1" or "Foundation Year Semester 2"
_SEMESTER_HDR_RE = re.compile(
    r"(Year\s+\d+\s*/\s*Semester\s+\d+|Foundation\s+Year\s+Semester\s+\d+|Semester\s+\d+)",
    re.I,
)


def _table_to_structured(table) -> str:
    """Convert a fitz table object to clean pipe-delimited rows."""
    rows = []
    for row in table.extract():
        cells = []
        for c in row:
            if c is not None:
                val = str(c).replace("\n", " ").strip()
                if val:
                    cells.append(val)
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _build_course_summary(raw_text: str, school: str, year: str) -> str:
    """
    Parse course rows from a curricula page and format them as natural language.
    Returns "" if no course rows found.
    """
    lines = raw_text.splitlines()
    courses = []
    current_section = ""

    for line in lines:
        line = line.strip()
        hdr_m = _SEMESTER_HDR_RE.search(line)
        if hdr_m:
            current_section = hdr_m.group(1).strip()
            continue

        m = _COURSE_ROW_RE.match(line)
        if m:
            code   = m.group(1).strip()
            title  = m.group(2).strip()
            cr     = m.group(3)
            prereq = m.group(6).strip() if m.group(6) else ""
            entry  = f"{code}: {title} [{cr} credits"
            if prereq:
                entry += f", Prereq: {prereq}"
            entry += "]"
            if current_section:
                entry = f"({current_section}) {entry}"
            courses.append(entry)

    if not courses:
        return ""

    header = f"{school} Curriculum {year} - Course Listing:\n"
    return header + "\n".join(courses)


# ── Per-PDF extractor ──────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract all usable records from one PDF file."""
    meta     = classify_filename(pdf_path.name)
    school   = meta["school"]
    doc_type = meta["doc_type"]
    year     = meta["year"]

    records = []
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"  ERROR opening {pdf_path.name}: {exc}")
        return []

    for page_num, page in enumerate(doc, 1):
        raw  = page.get_text()
        text = _strip_boilerplate(raw)

        if len(text) < MIN_CHARS:
            continue

        # 1. Base record: raw page text
        records.append({
            "filename":     pdf_path.name,
            "page_number":  page_num,
            "raw_text":     text,
            "school":       school,
            "doc_type":     doc_type,
            "year":         year,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        })

        # 2. For curricula: also emit a structured course-summary record
        if doc_type == "curricula":
            summary = _build_course_summary(text, school, year)
            if summary:
                records.append({
                    "filename":     pdf_path.name,
                    "page_number":  page_num,
                    "raw_text":     summary,
                    "school":       school,
                    "doc_type":     "course_table",
                    "year":         year,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                })

        # 3. For any page with detectable tables: emit a clean table record
        try:
            tabs = page.find_tables()
            for t in tabs.tables:
                structured = _table_to_structured(t)
                if structured and len(structured) >= MIN_CHARS:
                    prefix = (
                        f"{school} {doc_type.title()} {year} "
                        f"(page {page_num}):\n"
                    )
                    records.append({
                        "filename":     pdf_path.name,
                        "page_number":  page_num,
                        "raw_text":     prefix + structured,
                        "school":       school,
                        "doc_type":     doc_type + "_table",
                        "year":         year,
                        "extracted_at": datetime.now(timezone.utc).isoformat(),
                    })
        except Exception:
            pass

    doc.close()
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 2 - PDF Extraction")
    print("=" * 62)

    pdf_files = sorted(PDF_SRC_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  No PDFs found in {PDF_SRC_DIR}")
        OUT_FILE.write_text("", encoding="utf-8")
        return

    print(f"  Source : {PDF_SRC_DIR}")
    print(f"  PDFs   : {len(pdf_files)} files")
    print()

    all_records: list[dict] = []
    school_counts: dict[str, int] = {}

    for pdf_path in pdf_files:
        meta = classify_filename(pdf_path.name)
        recs = extract_pdf(pdf_path)
        all_records.extend(recs)
        s = meta["school"]
        school_counts[s] = school_counts.get(s, 0) + len(recs)
        print(f"  [{s:7s} / {meta['doc_type']:10s} / {meta['year']}]  "
              f"{len(recs):3d} records  {pdf_path.name}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print("=" * 62)
    print("  Phase 2 - SUMMARY")
    print("=" * 62)
    print(f"  Total records : {len(all_records)}")
    print(f"  Output        : {OUT_FILE}")
    print()
    print("  Records by school:")
    for school, cnt in sorted(school_counts.items()):
        print(f"    {school:8s}: {cnt}")
    print()
    print("  Phase 2 complete. Run phase3_clean_data.py next.")


if __name__ == "__main__":
    run()
