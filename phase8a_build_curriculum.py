#!/usr/bin/env python3
"""
phase8a_build_curriculum.py — Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
One-time extractor: queries all ChromaDB chunks, parses Zewail City course
tables (CODE | Name | Cr | L | P | Prerequisites), and saves a structured
JSON to data/curriculum/courses.json for the Academic Advisor Engine.

Run once after phase 4 completes:
    python phase8a_build_curriculum.py

The advisor engine (phase8) loads this file at startup.  If the file does not
exist the advisor still works — it falls back to pure RAG reasoning.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT   = Path(__file__).parent
CURRICULUM_DIR = PROJECT_ROOT / "data" / "curriculum"
OUTPUT_FILE    = CURRICULUM_DIR / "courses.json"

# ── Valid department prefixes in Zewail City course codes ──────────────────────
_VALID_DEPTS = {
    "CSAI", "MATH", "PHYS", "CHEM", "BIO",  "ENG",  "ENGR",
    "BUS",  "SCI",  "DSAI", "HCI",  "ELEC", "MECH", "CIE",
    "AERO", "NANO", "ENVE", "RENE", "GEN",  "STAT", "ECON",
    "MGMT", "FIN",  "ACTU", "OM",   "ENT",  "BIOL", "CENG",
}

# ── Regex patterns ─────────────────────────────────────────────────────────────

# Full table row: CODE | Name | Cr | L | P | Prerequisite(s)
# e.g.  CSAI 201 | Data Structures | 3 | 2 | 2 | CSAI 101
_FULL_ROW = re.compile(
    r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)'   # dept + number
    r'\s*\|\s*'
    r'([^|\n]{3,80}?)'                       # course name (non-greedy)
    r'\s*\|\s*'
    r'(\d+)'                                 # credit hours
    r'\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*'      # L | P | (lecture / practical)
    r'([^\n]*)',                             # prerequisites (rest of line)
    re.MULTILINE,
)

# Short table row: CODE | Name | Cr | Prereqs  (no L/P columns)
_SHORT_ROW = re.compile(
    r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)'
    r'\s*\|\s*'
    r'([^|\n]{3,80}?)'
    r'\s*\|\s*'
    r'(\d+)'
    r'\s*\|\s*'
    r'([^\n|]{0,120})',
    re.MULTILINE,
)

# Course code embedded anywhere (for parsing prerequisite strings)
_CODE_ANYWHERE = re.compile(r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)\b')

# Strings that mean "no prerequisite"
_NO_PREREQ = re.compile(
    r'^\s*(-|–|—|none|n/a|nil|no prereq|no prerequisite|—|-)\s*$', re.I
)


# ── Helper functions ───────────────────────────────────────────────────────────

def norm_code(dept: str, num: str) -> str:
    return f"{dept.strip().upper()} {num.strip().upper()}"


def parse_prereqs(raw: str) -> list[str]:
    """
    Extract prerequisite course codes from a raw prerequisite string.
    Handles: "CSAI 101", "CSAI 101, CSAI 102", "CSAI 101 & MATH 201",
             "Co: CSAI 101" (co-requisite — treated as prerequisite),
             "-", "None", empty string.
    """
    if not raw or _NO_PREREQ.match(raw):
        return []
    codes: list[str] = []
    # Skip "Co:" prefix but still treat them as prerequisites
    text = re.sub(r'^Co[-:\s]+', '', raw, flags=re.I)
    for m in _CODE_ANYWHERE.finditer(text.upper()):
        dept, num = m.group(1), m.group(2)
        if dept in _VALID_DEPTS:
            c = norm_code(dept, num)
            if c not in codes:
                codes.append(c)
    return codes


def clean_name(raw: str) -> str:
    name = raw.strip()
    # Strip trailing footnote markers
    name = re.sub(r'[\*#†‡\^\s]+$', '', name).strip()
    return name[:120]


# ── Main build function ────────────────────────────────────────────────────────

def build() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from dotenv import load_dotenv
    load_dotenv()

    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    CHROMA_DIR = PROJECT_ROOT / "db" / "chroma_db"
    if not CHROMA_DIR.exists():
        print(f"ERROR: ChromaDB not found at {CHROMA_DIR}")
        print("Run phase4_chunk_and_embed.py first.")
        sys.exit(1)

    print(f"Connecting to ChromaDB at {CHROMA_DIR} …")
    db  = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = db.get_collection("zewail_campus")
    total = col.count()
    print(f"Total chunks in collection: {total:,}")
    print()

    # courses[code] = {name, credits, prerequisites, _conf}
    # _conf tracks source confidence (curriculum/catalog > web)
    courses: dict[str, dict] = {}

    batch_size = 500
    processed  = 0

    for offset in range(0, total, batch_size):
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )

        for doc, meta in zip(result["documents"], result["metadatas"]):
            source = meta.get("source", "").lower()
            # Curriculum / catalog PDFs are the most authoritative source
            is_curriculum = any(
                kw in source for kw in ("curricul", "catalog", "handbook", "syllabi")
            )
            conf = 2 if is_curriculum else 1

            # ── Try full-format rows first ────────────────────────────────────
            for m in _FULL_ROW.finditer(doc):
                dept = m.group(1).upper()
                num  = m.group(2).upper()
                if dept not in _VALID_DEPTS:
                    continue

                credits = int(m.group(4))
                if not (1 <= credits <= 6):
                    continue

                name = clean_name(m.group(3))
                if len(name) < 4:
                    continue

                code    = norm_code(dept, num)
                prereqs = parse_prereqs(m.group(5))

                existing = courses.get(code)
                if existing is None:
                    courses[code] = {
                        "name": name,
                        "credits": credits,
                        "prerequisites": prereqs,
                        "_conf": conf,
                    }
                elif conf > existing["_conf"]:
                    # Better source → overwrite name and prereqs
                    courses[code].update(
                        name=name,
                        credits=credits,
                        prerequisites=prereqs,
                        _conf=conf,
                    )

            # ── Try short-format rows for codes not yet captured ─────────────
            for m in _SHORT_ROW.finditer(doc):
                dept = m.group(1).upper()
                num  = m.group(2).upper()
                if dept not in _VALID_DEPTS:
                    continue

                code = norm_code(dept, num)
                if code in courses:
                    continue  # already have better data

                credits = int(m.group(4))
                if not (1 <= credits <= 6):
                    continue

                name = clean_name(m.group(3))
                if len(name) < 4:
                    continue

                prereqs = parse_prereqs(m.group(5))
                courses[code] = {
                    "name": name,
                    "credits": credits,
                    "prerequisites": prereqs,
                    "_conf": 0,
                }

        processed += len(result["documents"])
        n = len(courses)
        print(f"  [{processed:>5}/{total}] {n:>4} courses extracted …", end="\r")

    print()
    print()

    # ── Post-process: validate prerequisite references ─────────────────────────
    known = set(courses.keys())
    bad_refs = 0
    for code, info in courses.items():
        original = info["prerequisites"]
        valid = [p for p in original if p in known and p != code]
        if len(valid) != len(original):
            bad_refs += len(original) - len(valid)
        info["prerequisites"] = valid

    if bad_refs:
        print(f"  Removed {bad_refs} unresolved prerequisite references.")

    # ── Strip internal tracking field ──────────────────────────────────────────
    for info in courses.values():
        info.pop("_conf", None)

    # ── Statistics ─────────────────────────────────────────────────────────────
    dept_counts: dict[str, int] = defaultdict(int)
    with_prereqs = sum(1 for info in courses.values() if info["prerequisites"])
    for code in courses:
        dept_counts[code.split()[0]] += 1

    print(f"Extracted {len(courses)} courses total.")
    print(f"  With prerequisites : {with_prereqs}")
    print(f"  Without prereqs    : {len(courses) - with_prereqs}")
    print()
    print("By department:")
    for dept, cnt in sorted(dept_counts.items(), key=lambda x: -x[1]):
        print(f"  {dept:<10} {cnt}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "version":       "1.0",
        "built_at":      datetime.now(timezone.utc).isoformat(),
        "total_courses": len(courses),
        "courses":       dict(sorted(courses.items())),
    }
    OUTPUT_FILE.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved → {OUTPUT_FILE}")
    print()
    print("You can now run phase7_streamlit_app.py — the advisor engine will")
    print("use this curriculum data for prerequisite checks and planning.")


if __name__ == "__main__":
    build()
