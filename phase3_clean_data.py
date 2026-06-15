#!/usr/bin/env python3
"""
phase3_clean_data.py  —  Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Phase 3: Load raw web + PDF data, clean it, deduplicate, and categorise.

Inputs : data/raw/web_raw.jsonl  (Phase 1)
         data/raw/pdf_raw.jsonl  (Phase 2)
Output : data/clean/cleaned_documents.jsonl

Each output record:
  doc_id       – unique document identifier
  source_type  – "web" or "pdf"
  source       – URL or PDF filename
  page         – page number (PDFs) or "" (web)
  category     – courses / admissions / policy / deadlines /
                 facilities / faculty / research / general
  text         – cleaned, normalised text
  char_count   – len(text)
  cleaned_at   – ISO-8601 UTC timestamp

Usage:
  python phase3_clean_data.py
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
WEB_JSONL    = PROJECT_ROOT / "data" / "raw" / "web_raw.jsonl"
PDF_JSONL    = PROJECT_ROOT / "data" / "raw" / "pdf_raw.jsonl"
CLEAN_DIR    = PROJECT_ROOT / "data" / "clean"
OUT_JSONL    = CLEAN_DIR / "cleaned_documents.jsonl"

MIN_CHARS         = 80    # discard records shorter than this after cleaning
NEAR_DUP_RATIO    = 0.85  # Jaccard similarity threshold for near-duplicate removal


# ── Category keywords ──────────────────────────────────────────────────────────

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("policy",     ["regulation", "policy", "academic integrity", "probation",
                    "dismissal", "attendance", "grading", "withdrawal", "conduct",
                    "credit transfer", "academic standing", "honour"]),
    ("courses",    ["course", "curriculum", "prerequisite", "credit hour",
                    "study plan", "syllabus", "elective", "core requirement",
                    "course catalog", "lecture", "lab session", "course code"]),
    ("admissions", ["admission", "apply", "application", "eligibility", "enroll",
                    "tuition", "fees", "scholarship", "financial aid",
                    "high school", "sat score", "gpa requirement"]),
    ("deadlines",  ["academic calendar", "registration", "add/drop", "deadline",
                    "spring semester", "fall semester", "summer term",
                    "exam period", "withdrawal date"]),
    ("research",   ["research", "publication", "journal", "conference", "grant",
                    "nanotechnology", "biomedical", "innovation", "phd",
                    "dissertation", "thesis"]),
    ("faculty",    ["faculty", "professor", "instructor", "dean", "staff",
                    "office hours", "phd from", "research interest", "biography"]),
    ("facilities", ["library", "laboratory", "dormitory", "cafeteria", "campus",
                    "shuttle", "clinic", "sports", "gym", "makerspace"]),
]


def classify(url_or_src: str, text: str) -> str:
    t = text.lower()
    s = url_or_src.lower()

    # URL/filename fast-path
    url_hints = {
        "policy":     ["polic", "regulat", "conduct", "integr"],
        "admissions": ["admiss", "apply", "enroll", "tuition", "scholar"],
        "courses":    ["cours", "curricul", "program", "degree", "catalog",
                       "minor", "handbook"],
        "deadlines":  ["calendar", "deadline", "registr", "schedule"],
        "research":   ["research", "publicat", "innovat"],
        "faculty":    ["faculty", "staff", "professor", "people"],
        "facilities": ["campus", "facilit", "librar", "housing"],
    }
    for cat, keys in url_hints.items():
        if any(k in s for k in keys):
            return cat

    # Content scoring
    scores: dict[str, int] = {c: 0 for c, _ in CATEGORY_RULES}
    for cat, kws in CATEGORY_RULES:
        for kw in kws:
            scores[cat] += t.count(kw)
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "general"


# ── Cleaning helpers ───────────────────────────────────────────────────────────

# Boilerplate phrases common on Zewail City pages
BOILERPLATE_PHRASES: list[str] = [
    "unlock your potential at zewail city",
    "apply now",
    "schedule a visit",
    "loading...",
    "back to top",
    "all rights reserved",
    "privacy policy",
    "terms of use",
    "follow us",
    "contact us",
    "subscribe",
    "newsletter",
    "copyright",
    "cookie",
    "accept all cookies",
    "we use cookies",
    "quick links",
    "social media",
    "share this",
    "read more",
    "learn more",
    "click here",
    "see all",
    "view all",
    "show more",
    "skip to content",
    "main menu",
    "navigation",
    "breadcrumb",
    "search results",
    "no results found",
    "page not found",
    "404",
    "403 forbidden",
]


def clean_text(raw: str, source_type: str) -> str:
    # Normalise unicode (NFKC: compose + compatibility decomposition)
    text = unicodedata.normalize("NFKC", raw)

    # Fix common encoding artefacts
    text = text.replace("\x00", "").replace("�", "").replace(" ", " ")

    # PDF-specific: remove repeated hyphenation at line ends
    if source_type == "pdf":
        text = re.sub(r"-\n([a-z])", r"\1", text)  # rejoin hyphenated words

    # Remove URLs embedded in text
    text = re.sub(r"https?://\S+", "", text)

    # Remove excessive punctuation repetition
    text = re.sub(r"[_\-=]{4,}", " ", text)
    text = re.sub(r"\.{4,}", "...", text)

    # Normalise whitespace
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        # Skip boilerplate lines
        line_low = line.lower()
        if any(bp in line_low for bp in BOILERPLATE_PHRASES):
            continue
        # Skip lines that look like navigation (very short, capitalised)
        if len(line) < 4:
            continue
        # Skip lines that are purely numeric (page numbers, dates in headers)
        if re.fullmatch(r"[\d\s/\-|]+", line) and len(line) < 12:
            continue
        lines.append(line)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def shingle_hash(text: str, k: int = 5) -> set[str]:
    """k-shingle set for Jaccard near-duplicate detection."""
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i: i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_web(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append({
                "source_type": "web",
                "source":      rec["url"],
                "page":        "",
                "raw_text":    rec["raw_text"],
                "hint":        rec.get("category_hint", ""),
            })
    return records


def load_pdf(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            # Build a category hint from school + doc_type for accurate classification
            school   = rec.get("school", "")
            doc_type = rec.get("doc_type", "")
            if "policy" in doc_type or "regulation" in doc_type:
                hint = "policy"
            elif doc_type in ("curricula", "catalog", "course_table",
                              "curricula_table", "catalog_table"):
                hint = "courses"
            else:
                hint = ""
            records.append({
                "source_type": "pdf",
                "source":      rec["filename"],
                "page":        str(rec["page_number"]),
                "raw_text":    rec["raw_text"],
                "hint":        hint,
                "school":      school,
                "doc_type":    doc_type,
            })
    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3 - Data Cleaning")
    print("=" * 62)

    raw_web  = load_web(WEB_JSONL)
    raw_pdf  = load_pdf(PDF_JSONL)
    total_in = len(raw_web) + len(raw_pdf)
    print(f"  Loaded : {len(raw_web)} web records, {len(raw_pdf)} PDF records ({total_in} total)")

    cleaned_records: list[dict] = []
    seen_hashes:     set[str]   = set()
    shingles_list:   list[set]  = []

    stats = {
        "short":     0,
        "exact_dup": 0,
        "near_dup":  0,
        "kept":      0,
    }
    cat_counts: dict[str, int] = {}

    for idx, raw in enumerate(raw_web + raw_pdf):
        text = clean_text(raw["raw_text"], raw["source_type"])

        if len(text) < MIN_CHARS:
            stats["short"] += 1
            continue

        # Exact duplicate check
        ch = content_hash(text)
        if ch in seen_hashes:
            stats["exact_dup"] += 1
            continue
        seen_hashes.add(ch)

        # Near-duplicate check (Jaccard on 5-shingles)
        shingle = shingle_hash(text)
        near_dup = False
        for existing in shingles_list[-200:]:  # sliding window for speed
            if jaccard(shingle, existing) >= NEAR_DUP_RATIO:
                near_dup = True
                break
        if near_dup:
            stats["near_dup"] += 1
            continue
        shingles_list.append(shingle)

        # PDF records carry a pre-computed hint from school/doc_type;
        # use it directly to avoid mis-classification of policy PDFs as "research".
        hint = raw.get("hint", "")
        if hint in ("policy", "courses", "admissions", "deadlines",
                    "research", "faculty", "facilities", "general"):
            cat = hint
        else:
            cat = classify(raw["source"], text)
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

        doc_id = f"{raw['source_type']}_{idx:05d}"
        cleaned_records.append({
            "doc_id":      doc_id,
            "source_type": raw["source_type"],
            "source":      raw["source"],
            "page":        raw["page"],
            "category":    cat,
            "text":        text,
            "char_count":  len(text),
            "cleaned_at":  datetime.now(timezone.utc).isoformat(),
        })
        stats["kept"] += 1

    # Write output
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for rec in cleaned_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total_chars = sum(r["char_count"] for r in cleaned_records)

    print()
    print("=" * 62)
    print("  Phase 3 - SUMMARY")
    print("=" * 62)
    print(f"  Records in          : {total_in}")
    print(f"  Too short (< {MIN_CHARS}c)  : {stats['short']}")
    print(f"  Exact duplicates    : {stats['exact_dup']}")
    print(f"  Near-duplicates     : {stats['near_dup']}")
    print(f"  Records kept        : {stats['kept']}")
    print(f"  Total characters    : {total_chars:,}")
    print(f"  Output              : {OUT_JSONL}")
    print()
    print("  Category breakdown:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:<14s}  {cnt:>3}  {'|' * min(cnt, 40)}")
    print()
    print(f"  Phase 3 complete. {stats['kept']} clean documents ready for Phase 4.")


if __name__ == "__main__":
    run()
