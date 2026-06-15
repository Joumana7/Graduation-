#!/usr/bin/env python3
"""
phase4b_add_overview.py  —  Zewail Campus Digital Assistant
Injects curated school/program overview documents into ChromaDB so that
broad questions ("what majors?", "how many schools?") get direct answers.
Run once after phase4_chunk_and_embed.py.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT    = Path(__file__).parent
CHROMA_DIR      = PROJECT_ROOT / "db" / "chroma_db"
COLLECTION_NAME = "zewail_campus"
EMBED_MODEL     = "text-embedding-3-small"

# ── Overview documents ─────────────────────────────────────────────────────────

OVERVIEW_DOCS = [
    {
        "id":       "overview_schools_001",
        "category": "general",
        "source":   "Zewail City Overview (curated)",
        "page":     "",
        "source_type": "overview",
        "text": """\
Zewail City of Science and Technology (UST) — Schools and Programs Overview

Zewail City has 3 undergraduate schools (also called colleges or faculties):

1. School of Computational Sciences and Artificial Intelligence (CSAI)
   Undergraduate programs (majors) offered:
   • Computer Science (CS)
   • Data Science and Artificial Intelligence (DSAI)
   • Human-Computer Interaction (HCI / SWHCI)
   • Computer Engineering (CE)
   Total graduation requirement: 132 credit hours

2. School of Science (SCI)
   Undergraduate programs (majors) offered:
   • Biomedical Sciences
   • Nanoscience
   • Physics of the Universe (Physics)
   Total graduation requirement: varies by major (~150-160 credit hours)

3. School of Business (BUS)
   Undergraduate programs (majors) offered:
   • Finance
   • Business Analytics
   • Actuarial Analysis and Risk Management
   • Operations Management
   • Entrepreneurship and Innovation Management
   Total graduation requirement: 114 credit hours

All three schools are part of the University of Science and Technology (UST)
at Zewail City of Science, Technology and Innovation, located in 6th of October City, Egypt.
""",
    },
    {
        "id":       "overview_schools_002",
        "category": "general",
        "source":   "Zewail City Overview (curated)",
        "page":     "",
        "source_type": "overview",
        "text": """\
Zewail City Undergraduate Majors — Quick Reference

How many schools does Zewail City have?
Zewail City (UST) has 3 schools: CSAI, SCI, and BUS.

What majors / programs are offered at Zewail City?

CSAI School majors:
• Computer Science
• Data Science and Artificial Intelligence (DSAI)
• Human-Computer Interaction (HCI)
• Computer Engineering

SCI School (School of Science) majors:
• Biomedical Sciences — study of microbiology, biochemistry, pharmacology, genetics
• Nanoscience — materials, nanotechnology, nanofabrication
• Physics of the Universe — theoretical and experimental physics

BUS School (School of Business) majors:
• Finance
• Business Analytics
• Actuarial Analysis and Risk Management
• Operations Management
• Entrepreneurship and Innovation Management

Graduate programs are also available in fields such as Nanoscience, Biomedical Sciences,
Renewable Energy Engineering, Communications Engineering, and Business Administration (MBA).
""",
    },
    {
        "id":       "overview_csai_programs",
        "category": "courses",
        "source":   "CSAI Programs Overview (curated)",
        "page":     "",
        "source_type": "overview",
        "text": """\
CSAI School Programs — Detailed Overview

The School of Computational Sciences and Artificial Intelligence (CSAI) at Zewail City
offers 4 undergraduate concentrations (majors/tracks):

1. Computer Science (CS)
   Focus: algorithms, software engineering, databases, cybersecurity, AI.
   Courses include: Data Structures (CSAI 201), Algorithms (CSAI 251),
   Machine Learning (CSAI 253), Advanced Databases (CSAI 302), AI (CSAI 301).

2. Data Science and Artificial Intelligence (DSAI)
   Focus: data science, machine learning, deep learning, statistics.
   Courses include: Data Governance (DSAI 202), Deep Learning (DSAI 308),
   Linear Programming (MATH 303).

3. Human-Computer Interaction (HCI / SWHCI)
   Focus: user experience, UI development, cognitive psychology, prototyping.
   Courses include: User Interface Development (SW 302), Cognitive Psychology (SCH 273),
   Prototyping Algorithmic Experiences (SWHCI 301).

4. Computer Engineering (CE)
   Focus: embedded systems, hardware-software integration, communications.
   Courses include: Embedded Systems (SW 252), Computer Architecture, Digital Systems.

All CSAI tracks share a common first year and must complete 132 credit hours for graduation.
""",
    },
    {
        "id":       "overview_sci_programs",
        "category": "courses",
        "source":   "SCI Programs Overview (curated)",
        "page":     "",
        "source_type": "overview",
        "text": """\
SCI School (School of Science) Programs — Detailed Overview

The School of Science (SCI) at Zewail City offers 3 undergraduate programs (majors):

1. Biomedical Sciences
   Focus: biology, microbiology, biochemistry, pharmacology, genetics, cell biology.
   Key courses: General Microbiology (BMS 201), Cell Biology (BMS 202),
   Biochemistry (BMS 203), Principles of Pharmacology (BMS 205), Genetics (BMS 339).
   Electives include: Advanced Microbiology (BMS 335), Drug Targets (BMS 409),
   Protein Structure and Function (BMS 425).

2. Nanoscience
   Focus: nanotechnology, materials science, nanofabrication, quantum physics.
   Covers both theoretical foundations and practical lab-based nanoscience research.

3. Physics of the Universe (Physics)
   Focus: classical mechanics, electromagnetism, quantum mechanics, astrophysics.
   Provides strong theoretical and experimental physics foundation.

SCI students must complete school requirements plus major-specific requirements.
Total credit hours vary by program.
""",
    },
    {
        "id":       "overview_bus_programs",
        "category": "courses",
        "source":   "BUS Programs Overview (curated)",
        "page":     "",
        "source_type": "overview",
        "text": """\
BUS School (School of Business) Programs — Detailed Overview

The School of Business (BUS) at Zewail City offers 5 undergraduate majors:

1. Finance
   Focus: financial markets, investment analysis, corporate finance, risk management.

2. Business Analytics
   Focus: data-driven decision making, statistical analysis, business intelligence.

3. Actuarial Analysis and Risk Management
   Focus: probability, statistics, risk assessment, insurance mathematics.

4. Operations Management
   Focus: supply chain, process optimization, project management, logistics.

5. Entrepreneurship and Innovation Management
   Focus: startup creation, business models, innovation strategy, venture capital.

BUS graduation requirements:
• 48 Credit Hours of School Requirements
• 66 Credit Hours of Program/Major Requirements
• Total: 114 Credit Hours minimum

BUS students follow a common first-year curriculum before specializing in their major.
""",
    },
]


def run() -> None:
    import chromadb
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    oai = OpenAI(api_key=api_key)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_collection(COLLECTION_NAME)

    print(f"Phase 4b — Adding overview documents")
    print(f"  Collection before: {col.count()} chunks")

    added = 0
    skipped = 0
    for doc in OVERVIEW_DOCS:
        # Skip if already present
        existing = col.get(ids=[doc["id"]])
        if existing["ids"]:
            skipped += 1
            continue

        # Embed
        resp = oai.embeddings.create(model=EMBED_MODEL, input=[doc["text"]])
        emb = resp.data[0].embedding

        col.add(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[emb],
            metadatas=[{
                "source":      doc["source"],
                "source_type": doc["source_type"],
                "category":    doc["category"],
                "page":        doc["page"],
                "added_at":    datetime.now(timezone.utc).isoformat(),
            }],
        )
        print(f"  + Added: {doc['id']}")
        added += 1

    print(f"\n  Added  : {added} documents")
    print(f"  Skipped: {skipped} (already present)")
    print(f"  Collection after: {col.count()} chunks")
    print("\n  Phase 4b complete.")


if __name__ == "__main__":
    run()
