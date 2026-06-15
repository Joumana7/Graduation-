#!/usr/bin/env python3
"""
phase4c_fix_overviews.py  —  Zewail Campus Digital Assistant
Fixes the overview documents to include 4 schools (was 3 — Engineering was missing).
Adds Engineering school overview + comprehensive faculty directory docs.
Run once after phase4b_add_overview.py.
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

# ── Documents to DELETE (they had wrong 3-school info) ────────────────────────
IDS_TO_DELETE = [
    "overview_schools_001",
    "overview_schools_002",
]

# ── Replacement + new documents ───────────────────────────────────────────────
NEW_DOCS = [
    {
        "id":          "overview_schools_001",
        "category":    "general",
        "source":      "Zewail City Overview (curated)",
        "source_type": "overview",
        "text": """\
Zewail City of Science and Technology (UST) — Schools and Programs Overview

Zewail City has 4 undergraduate schools (also called colleges or faculties):

1. School of Engineering (ENGR)
   Undergraduate programs offered:
   • Aerospace Engineering
   • Nanotechnology and Nanoelectronics Engineering
   • Environmental Engineering (from Fall 2026: Chemical and Environmental Engineering)
   • Communications and Information Engineering (CIE)
   • Renewable Energy Engineering

2. School of Computational Sciences and Artificial Intelligence (CSAI)
   Undergraduate programs (majors/concentrations/tracks) offered:
   • Computer Science (CS)
   • Data Science and Artificial Intelligence (DSAI)
   • Human-Computer Interaction (HCI / SWHCI)
   • Computer Engineering (CE)
   Total graduation requirement: 132 credit hours

3. School of Science (SCI)
   Undergraduate programs (majors) offered:
   • Biomedical Sciences
   • Nanoscience
   • Physics of the Universe (Physics)

4. School of Business (BUS)
   Undergraduate programs (majors) offered:
   • Finance
   • Business Analytics
   • Actuarial Analysis and Risk Management
   • Operations Management
   • Entrepreneurship and Innovation Management
   Total graduation requirement: 114 credit hours

All four schools are part of the University of Science and Technology (UST)
at Zewail City of Science, Technology and Innovation, 6th of October City, Egypt.
""",
    },
    {
        "id":          "overview_schools_002",
        "category":    "general",
        "source":      "Zewail City Overview (curated)",
        "source_type": "overview",
        "text": """\
Zewail City Undergraduate Majors — Quick Reference

How many schools does Zewail City have?
Zewail City (UST) has 4 schools: Engineering (ENGR), CSAI, SCI, and BUS.

What majors / programs are offered at Zewail City?

ENGR School (School of Engineering) programs:
• Aerospace Engineering
• Nanotechnology and Nanoelectronics Engineering
• Environmental Engineering (→ Chemical and Environmental Engineering from Fall 2026)
• Communications and Information Engineering (CIE)
• Renewable Energy Engineering

CSAI School majors:
• Computer Science (CS)
• Data Science and Artificial Intelligence (DSAI)
• Human-Computer Interaction (HCI)
• Computer Engineering (CE)

SCI School (School of Science) majors:
• Biomedical Sciences — microbiology, biochemistry, pharmacology, genetics
• Nanoscience — materials, nanotechnology, nanofabrication
• Physics of the Universe — theoretical and experimental physics

BUS School (School of Business) majors:
• Finance
• Business Analytics
• Actuarial Analysis and Risk Management
• Operations Management
• Entrepreneurship and Innovation Management

Graduate programs are also available in Nanoscience, Biomedical Sciences,
Renewable Energy Engineering, Communications Engineering, and Business Administration (MBA).
""",
    },
    {
        "id":          "overview_eng_programs",
        "category":    "courses",
        "source":      "Engineering School Overview (curated)",
        "source_type": "overview",
        "text": """\
School of Engineering (ENGR) — Programs and Details

The School of Engineering (ENGR) at Zewail City offers 5 undergraduate programs:

1. Aerospace Engineering
   Focus: fluid mechanics, aerodynamics, control engineering, robotics, propulsion,
   thermal management. Hands-on and experimental design.
   Program Director: Dr. Ahmed Eltaweel (Acting Program Director)

2. Nanotechnology and Nanoelectronics Engineering
   Focus: nanomaterials, nanodevices, nanoelectronics, photonics, quantum physics.
   Director: Dr. Mohamed Farhat O. Hameed (also Director of Center for Nanotechnology)

3. Environmental Engineering
   Focus: environmental science, chemical engineering, waste management, sustainability.
   Note: Starting Fall 2026, new admissions are under Chemical and Environmental Engineering.
   Director: Dr. Tamer Samir Ahmed (Acting Dean of Academic Affairs)

4. Communications and Information Engineering (CIE)
   Focus: electronics, communications, embedded systems, control, machine learning,
   IoT, wireless communications, cybersecurity, information theory.
   Director: Dr. Samy Soliman

5. Renewable Energy Engineering
   Focus: solar, wind, energy storage, smart grids, sustainable power systems.
   Director: Dr. Amgad A. El-Deib (also Director of CREEE Center)

All Engineering programs emphasize hands-on experience, multidisciplinary education,
and culminate in a two-semester senior design (capstone) project.
Graduate programs include MSc in Nanotechnology and Nanoelectronics Engineering.
""",
    },
    {
        "id":          "overview_faculty_engr",
        "category":    "faculty",
        "source":      "Engineering School Faculty Directory (curated)",
        "source_type": "overview",
        "text": """\
School of Engineering (ENGR) — Faculty and Directors

Program Directors / Heads:
• Dr. Tamer Samir Ahmed — Acting Dean of Academic Affairs; Director of Environmental Engineering
  Email: tsamir@zewailcity.edu.eg
• Dr. Ahmed Eltaweel — Acting Program Director, Aerospace Engineering (Visiting Asst. Professor)
  Email: aeltaweel@zewailcity.edu.eg
• Dr. Amgad A. El-Deib — Director of Renewable Energy Engineering; Director of CREEE
  Associate Professor, Renewable Energy Engineering
  Email: aeldeib@zewailcity.edu.eg
• Dr. Mohamed Farhat O. Hameed — Director of Nanotechnology & Nanoelectronics Engineering;
  Director of Center for Nanotechnology; Professor
  Email: mfarahat@zewailcity.edu.eg
• Dr. Samy Soliman — Director of CIE Program; Associate Professor, CIE
  Email: ssoliman@zewailcity.edu.eg
• Dr. Tamer Ashour Ali — Acting Dean of Strategic Enrollment Management;
  Director of CIAU; Professor, CIE Program
  Email: tali@zewailcity.edu.eg

Faculty Members:
• Dr. Moustafa Elshafei — Professor, Communications and Information Engineering
  Email: moelshafei@zewailcity.edu.eg
• Dr. Omar Fahmy — Professor, Communications and Information Engineering
  Email: oFahmy@zewailcity.edu.eg
• Dr. Ahmed Fahmy A. Youssef — Associate Professor, Environmental Engineering
  Email: ahyoussef@zewailcity.edu.eg
• Dr. Mohamed L. Shaltout — Associate Professor, Renewable Energy Engineering
• Dr. Hatem Fayed — Director of Applied Mathematics; Associate Professor, Math Department
  Email: hfayed@zewailcity.edu.eg
• Dr. Mahmoud Abdelaziz — Assistant Professor, CIE
  Email: mhabdelaziz@zewailcity.edu.eg
• Dr. Mohamad Samir A. Eid — Assistant Professor, CIE (Cybersecurity)
  Email: mseid@zewailcity.edu.eg
• Dr. Mohannad Draz — Assistant Professor, Aerospace Engineering
• Dr. Asmaa Harraz — Assistant Professor, Environmental Engineering
• Dr. Ahmed S. Abd-Rabou — Assistant Professor, Nanotechnology and Nanoelectronics
  Email: ahmed.abdrabou@zewailcity.edu.eg
• Dr. Noha Gaber — Assistant Professor, Center for Nanotechnology
• Dr. Shaimaa Ali Mohamed — Assistant Professor, Center for Nanotechnology
  Email: smohamed@zewailcity.edu.eg
""",
    },
    {
        "id":          "overview_faculty_csai",
        "category":    "faculty",
        "source":      "CSAI School Faculty Directory (curated)",
        "source_type": "overview",
        "text": """\
School of Computational Sciences and Artificial Intelligence (CSAI) — Faculty and Directors

Program Directors:
• Dr. Doaa Shawky — Director of Software Development Program; Professor, CSAI School
  Email: dshawky@zewailcity.edu.eg
• Dr. Khaled Mostafa El Sayed — Director of Data Science Program; Professor, CSAI School
• Dr. Ahmed Sayed Abdelsamea — Director of Academic Advising Unit; Associate Professor, Math Dept.
  Email: aabdelsamea@zewailcity.edu.eg

Faculty Members:
• Dr. Mayada Hadhoud — Associate Professor, CSAI School
• Dr. Mohamed Maher Ata — Associate Professor, CSAI School (Data Science and AI)
  Email: momaher@zewailcity.edu.eg
• Dr. Abdallah Aboutahoun — Professor, Math Department
  Email: atahoun@zewailcity.edu.eg

Note: The CSAI school includes programs in Computer Science, DSAI, HCI, and Computer Engineering.
""",
    },
    {
        "id":          "overview_faculty_sci",
        "category":    "faculty",
        "source":      "SCI School Faculty Directory (curated)",
        "source_type": "overview",
        "text": """\
School of Science (SCI) — Faculty and Directors

Program Directors:
• Dr. Nagwa El-Badri — Director of Biomedical Sciences Program;
  Director of Center of Excellence for Stem Cells and Regenerative Medicine (CECSRM); Professor
• Dr. Tarek Ibrahim — Director of Physics of Universe Program; Professor
• Dr. Abdelrahman S. Mayhoub — Director of CRM Center; Professor, Nanoscience Program
  Email: amayhoub@zewailcity.edu.eg
• Dr. Ayman El-Shibiny — Director of Center for Microbiology and Phage Therapy (CMP);
  Director of Center of Scientific Excellence for Food Research (cSEFRA);
  Founding Director of Food Safety and Quality Diploma

Faculty Members:
• Dr. Amr Mohamed — Associate Professor, Physics of Universe Program
  Email: amr@zewailcity.edu.eg (Physics PhD from MIT)
• Dr. Eman Badr — Associate Professor, Biomedical Sciences Program
• Dr. Ali Nassar — Assistant Professor, Physics of Universe Program
• Dr. Reem K. Arafa — Professor, Biomedical Sciences Program
• Dr. Ali Wagdy Mohamed — Professor, School of Business (also listed cross-school)
""",
    },
    {
        "id":          "overview_faculty_all",
        "category":    "faculty",
        "source":      "Zewail City Faculty Directory (curated)",
        "source_type": "overview",
        "text": """\
Zewail City Faculty and Program Directors — Full Listing

ENGINEERING SCHOOL (ENGR) Directors:
• Dr. Tamer Samir Ahmed — Dean of Academic Affairs; Director, Environmental Engineering
• Dr. Ahmed Eltaweel — Acting Director, Aerospace Engineering
• Dr. Amgad A. El-Deib — Director, Renewable Energy Engineering
• Dr. Mohamed Farhat O. Hameed — Director, Nanotechnology & Nanoelectronics Engineering
• Dr. Samy Soliman — Director, Communications & Information Engineering (CIE)
• Dr. Tamer Ashour Ali — Dean of Strategic Enrollment; Director, CIAU

CSAI SCHOOL Directors:
• Dr. Doaa Shawky — Director, Software Development Program
• Dr. Khaled El Sayed — Director, Data Science and AI Program
• Dr. Ahmed Abdelsamea — Director, Academic Advising Unit

SCI SCHOOL Directors:
• Dr. Nagwa El-Badri — Director, Biomedical Sciences Program
• Dr. Tarek Ibrahim — Director, Physics of Universe Program
• Dr. Abdelrahman Mayhoub — Director, CRM Center / Nanoscience
• Dr. Ayman El-Shibiny — Director, Center for Microbiology; Director, Food Research Center

MATH DEPARTMENT:
• Dr. Hatem Fayed — Director, Applied Mathematics
• Dr. Ahmed Abdelsamea — Director, Academic Advising Unit
• Dr. Abdallah Aboutahoun — Professor, Math

OTHER / CROSS-SCHOOL:
• Dr. Mostafa Samir Moussa Badawy — Director, Valley of Science and Technology
• Dr. Abdelrahman Mayhoub — Director, CRM Center
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

    print(f"Phase 4c — Fixing overview documents + adding faculty docs")
    print(f"  Collection before: {col.count()} chunks")

    # 1. Delete stale overview docs
    for doc_id in IDS_TO_DELETE:
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            col.delete(ids=[doc_id])
            print(f"  - Deleted: {doc_id}")
        else:
            print(f"  - Not found (skip delete): {doc_id}")

    # 2. Add/update documents
    added = skipped = 0
    for doc in NEW_DOCS:
        existing = col.get(ids=[doc["id"]])
        if existing["ids"]:
            # Delete and re-add to update
            col.delete(ids=[doc["id"]])
            print(f"  ~ Replacing: {doc['id']}")

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
                "page":        "",
                "added_at":    datetime.now(timezone.utc).isoformat(),
            }],
        )
        print(f"  + Added: {doc['id']}")
        added += 1

    print(f"\n  Added/replaced: {added} documents")
    print(f"  Collection after: {col.count()} chunks")
    print("\n  Phase 4c complete.")


if __name__ == "__main__":
    run()
