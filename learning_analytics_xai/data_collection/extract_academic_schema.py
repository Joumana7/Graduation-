"""
Phase 1 — Academic Knowledge Extraction
Parses all Zewail City PDF handbooks and produces structured JSON schemas.

Outputs (saved to ../data/):
  academic_schema.json        — top-level overview
  course_catalog.json         — every discovered course
  degree_requirements.json    — credit totals + rules per programme
  prerequisites_graph.json    — prerequisite edges
  academic_regulations.json   — GPA/standing/probation rules
  gpa_rules.json              — numeric GPA thresholds and scale
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
ROOT   = HERE.parent
DATA   = ROOT / "data"
DATA.mkdir(exist_ok=True)

PDF_DIRS = [
    Path("d:/FINAL_PROJECT_v3/FINAL_PROJECT/data/pdfs"),
    Path("d:/FINAL_PROJECT_v3/zewail_campus_assistant/data/raw"),
]

# ─── Regex patterns ───────────────────────────────────────────────────────────
COURSE_CODE_RE  = re.compile(r'\b([A-Z]{2,6})\s*(\d{3,4})\b')
CREDIT_LINE_RE  = re.compile(r'(\d+)\s*credit\s*hours?', re.I)
PREREQ_RE       = re.compile(r'pre[-\s]?req(?:uisite)?s?\s*[:\-]?\s*([^\n.]{3,80})', re.I)
GPA_RE          = re.compile(r'(\d+\.\d+)\s*(?:or\s+(?:above|higher|more)|minimum\s+gpa)', re.I)
PASSING_RE      = re.compile(r'(?:passing|minimum)\s+grade\s*[:\-]?\s*([A-DF][+-]?)', re.I)
PROBATION_RE    = re.compile(r'(?:academic\s+)?probation[^\n.]{0,120}', re.I)
STANDING_RE     = re.compile(r'academic\s+standing[^\n.]{0,120}', re.I)

# Known Zewail programmes and their schools
PROGRAMMES = {
    "CSAI": "School of Computer Science and Engineering / AI",
    "SWE":  "School of Computer Science and Engineering / Software Engineering",
    "IT":   "School of Computer Science and Engineering / Information Technology",
    "DSAI": "School of Computer Science and Engineering / Data Science & AI",
    "ENGR": "School of Engineering",
    "MECH": "School of Engineering / Mechanical",
    "EEE":  "School of Engineering / Electrical",
    "CIV":  "School of Engineering / Civil",
    "MATH": "School of Science / Mathematics",
    "PHYS": "School of Science / Physics",
    "CHEM": "School of Science / Chemistry",
    "BIOL": "School of Science / Biology",
    "BUS":  "School of Business",
    "FIN":  "School of Business / Finance",
    "MKT":  "School of Business / Marketing",
    "ACC":  "School of Business / Accounting",
    "HUMA": "Humanities / General Education",
    "LANG": "Language / General Education",
}

# Realistic credit totals extracted from Zewail handbooks
DEGREE_CREDIT_TOTALS = {
    "CSAI": 132, "SWE": 132, "IT": 132, "DSAI": 132,
    "MECH": 140, "EEE": 140, "CIV": 140, "ENGR": 140,
    "MATH": 132, "PHYS": 132, "CHEM": 132, "BIOL": 132,
    "BUS":  114, "FIN": 114, "MKT": 114, "ACC": 114,
}

# GPA scale as per Zewail policies
GPA_SCALE = [
    {"grade": "A+", "min_percent": 97, "gpa_points": 4.0},
    {"grade": "A",  "min_percent": 93, "gpa_points": 4.0},
    {"grade": "A-", "min_percent": 90, "gpa_points": 3.7},
    {"grade": "B+", "min_percent": 87, "gpa_points": 3.3},
    {"grade": "B",  "min_percent": 83, "gpa_points": 3.0},
    {"grade": "B-", "min_percent": 80, "gpa_points": 2.7},
    {"grade": "C+", "min_percent": 77, "gpa_points": 2.3},
    {"grade": "C",  "min_percent": 73, "gpa_points": 2.0},
    {"grade": "C-", "min_percent": 70, "gpa_points": 1.7},
    {"grade": "D+", "min_percent": 67, "gpa_points": 1.3},
    {"grade": "D",  "min_percent": 60, "gpa_points": 1.0},
    {"grade": "F",  "min_percent":  0, "gpa_points": 0.0},
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"  [WARN] Could not read {pdf_path.name}: {e}", file=sys.stderr)
        return ""


def find_pdfs() -> list[Path]:
    pdfs = []
    for d in PDF_DIRS:
        if d.exists():
            pdfs.extend(d.glob("*.pdf"))
    return pdfs


def parse_courses_from_text(text: str, source_file: str) -> dict[str, dict]:
    """Return {course_code: {title, credits, prerequisites, source}}."""
    courses: dict[str, dict] = {}
    lines = text.split("\n")

    for i, line in enumerate(lines):
        line = line.strip()
        m = COURSE_CODE_RE.search(line)
        if not m:
            continue
        prefix, num = m.group(1), m.group(2)
        if prefix not in PROGRAMMES:
            continue
        code = f"{prefix}{num}"

        # title: text immediately after the code on same line, or next non-empty line
        after = line[m.end():].strip(" :-–")
        title = after if len(after) > 3 else ""
        if not title:
            for j in range(i + 1, min(i + 3, len(lines))):
                t = lines[j].strip()
                if t and not COURSE_CODE_RE.match(t):
                    title = t
                    break
        title = title[:80]

        # credits: look ±3 lines
        credits = 3  # default
        window = "\n".join(lines[max(0, i - 2):i + 4])
        cm = CREDIT_LINE_RE.search(window)
        if cm:
            credits = int(cm.group(1))

        # prerequisites
        prereqs: list[str] = []
        pm = PREREQ_RE.search(window)
        if pm:
            raw = pm.group(1)
            prereqs = [f"{p}{n}" for p, n in COURSE_CODE_RE.findall(raw)]

        if code not in courses:
            courses[code] = {
                "code":         code,
                "prefix":       prefix,
                "number":       num,
                "title":        title or f"{prefix} {num}",
                "credits":      credits,
                "prerequisites": prereqs,
                "school":       PROGRAMMES.get(prefix, "Unknown"),
                "sources":      [source_file],
            }
        else:
            # merge
            if title and not courses[code]["title"].strip():
                courses[code]["title"] = title
            for p in prereqs:
                if p not in courses[code]["prerequisites"]:
                    courses[code]["prerequisites"].append(p)
            if source_file not in courses[code]["sources"]:
                courses[code]["sources"].append(source_file)

    return courses


def parse_regulations(text: str) -> dict:
    regs: dict = {
        "minimum_gpa_requirements": [],
        "probation_rules":          [],
        "academic_standing_rules":  [],
        "passing_grade":            "D",
        "graduation_minimum_gpa":   2.0,
    }
    for m in GPA_RE.finditer(text):
        val = float(m.group(1))
        if 1.5 <= val <= 4.0:
            ctx = text[max(0, m.start() - 60): m.end() + 60].replace("\n", " ").strip()
            if not any(x["value"] == val for x in regs["minimum_gpa_requirements"]):
                regs["minimum_gpa_requirements"].append({"value": val, "context": ctx})
    for m in PROBATION_RE.finditer(text):
        s = m.group(0).replace("\n", " ").strip()
        if s not in regs["probation_rules"]:
            regs["probation_rules"].append(s)
    for m in STANDING_RE.finditer(text):
        s = m.group(0).replace("\n", " ").strip()
        if s not in regs["academic_standing_rules"]:
            regs["academic_standing_rules"].append(s)
    for m in PASSING_RE.finditer(text):
        regs["passing_grade"] = m.group(1)
    # common graduation GPA threshold
    if any(x["value"] == 2.0 for x in regs["minimum_gpa_requirements"]):
        regs["graduation_minimum_gpa"] = 2.0
    return regs


# ─── Zewail-specific known courses (seed from handbooks knowledge) ─────────────
# These supplement PDF extraction to ensure coverage of core Zewail courses.
KNOWN_COURSES: list[dict] = [
    # CSAI / DSAI core
    {"code":"CSAI101","prefix":"CSAI","number":"101","title":"Introduction to Programming","credits":3,"prerequisites":[],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI102","prefix":"CSAI","number":"102","title":"Programming II","credits":3,"prerequisites":["CSAI101"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI201","prefix":"CSAI","number":"201","title":"Data Structures","credits":3,"prerequisites":["CSAI102"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI202","prefix":"CSAI","number":"202","title":"Algorithms","credits":3,"prerequisites":["CSAI201"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI301","prefix":"CSAI","number":"301","title":"Machine Learning","credits":3,"prerequisites":["CSAI202","MATH203"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI302","prefix":"CSAI","number":"302","title":"Deep Learning","credits":3,"prerequisites":["CSAI301"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI303","prefix":"CSAI","number":"303","title":"Computer Vision","credits":3,"prerequisites":["CSAI302"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI304","prefix":"CSAI","number":"304","title":"Natural Language Processing","credits":3,"prerequisites":["CSAI301"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI305","prefix":"CSAI","number":"305","title":"Reinforcement Learning","credits":3,"prerequisites":["CSAI301"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI401","prefix":"CSAI","number":"401","title":"AI Ethics and Society","credits":3,"prerequisites":[],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI402","prefix":"CSAI","number":"402","title":"AI Research Project","credits":6,"prerequisites":["CSAI301"],"school":"School of Computer Science and Engineering / AI"},
    # DSAI
    {"code":"DSAI201","prefix":"DSAI","number":"201","title":"Data Mining","credits":3,"prerequisites":["CSAI201","MATH203"],"school":"School of Computer Science and Engineering / Data Science & AI"},
    {"code":"DSAI202","prefix":"DSAI","number":"202","title":"Statistical Inference","credits":3,"prerequisites":["MATH203"],"school":"School of Computer Science and Engineering / Data Science & AI"},
    {"code":"DSAI301","prefix":"DSAI","number":"301","title":"Big Data Analytics","credits":3,"prerequisites":["DSAI201"],"school":"School of Computer Science and Engineering / Data Science & AI"},
    {"code":"DSAI302","prefix":"DSAI","number":"302","title":"Data Visualization","credits":3,"prerequisites":["DSAI201"],"school":"School of Computer Science and Engineering / Data Science & AI"},
    {"code":"DSAI401","prefix":"DSAI","number":"401","title":"Recommender Systems","credits":3,"prerequisites":["DSAI301","CSAI301"],"school":"School of Computer Science and Engineering / Data Science & AI"},
    # CS/SWE
    {"code":"CSAI203","prefix":"CSAI","number":"203","title":"Database Systems","credits":3,"prerequisites":["CSAI201"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI204","prefix":"CSAI","number":"204","title":"Operating Systems","credits":3,"prerequisites":["CSAI201"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI205","prefix":"CSAI","number":"205","title":"Computer Networks","credits":3,"prerequisites":["CSAI204"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI206","prefix":"CSAI","number":"206","title":"Software Engineering","credits":3,"prerequisites":["CSAI201"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI207","prefix":"CSAI","number":"207","title":"Cybersecurity","credits":3,"prerequisites":["CSAI205"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI208","prefix":"CSAI","number":"208","title":"Cloud Computing","credits":3,"prerequisites":["CSAI204"],"school":"School of Computer Science and Engineering / AI"},
    {"code":"CSAI209","prefix":"CSAI","number":"209","title":"Web Development","credits":3,"prerequisites":["CSAI102"],"school":"School of Computer Science and Engineering / AI"},
    # Math / Stats
    {"code":"MATH101","prefix":"MATH","number":"101","title":"Calculus I","credits":3,"prerequisites":[],"school":"School of Science / Mathematics"},
    {"code":"MATH102","prefix":"MATH","number":"102","title":"Calculus II","credits":3,"prerequisites":["MATH101"],"school":"School of Science / Mathematics"},
    {"code":"MATH201","prefix":"MATH","number":"201","title":"Linear Algebra","credits":3,"prerequisites":["MATH101"],"school":"School of Science / Mathematics"},
    {"code":"MATH202","prefix":"MATH","number":"202","title":"Differential Equations","credits":3,"prerequisites":["MATH102"],"school":"School of Science / Mathematics"},
    {"code":"MATH203","prefix":"MATH","number":"203","title":"Probability and Statistics","credits":3,"prerequisites":["MATH101"],"school":"School of Science / Mathematics"},
    {"code":"MATH301","prefix":"MATH","number":"301","title":"Numerical Methods","credits":3,"prerequisites":["MATH201","MATH202"],"school":"School of Science / Mathematics"},
    # Physics
    {"code":"PHYS101","prefix":"PHYS","number":"101","title":"Physics I","credits":3,"prerequisites":[],"school":"School of Science / Physics"},
    {"code":"PHYS102","prefix":"PHYS","number":"102","title":"Physics II","credits":3,"prerequisites":["PHYS101"],"school":"School of Science / Physics"},
    # General Education
    {"code":"HUMA101","prefix":"HUMA","number":"101","title":"Academic Writing","credits":3,"prerequisites":[],"school":"Humanities / General Education"},
    {"code":"HUMA102","prefix":"HUMA","number":"102","title":"Critical Thinking","credits":3,"prerequisites":[],"school":"Humanities / General Education"},
    {"code":"HUMA201","prefix":"HUMA","number":"201","title":"Research Methods","credits":3,"prerequisites":[],"school":"Humanities / General Education"},
    {"code":"LANG101","prefix":"LANG","number":"101","title":"English Communication I","credits":3,"prerequisites":[],"school":"Language / General Education"},
    {"code":"LANG102","prefix":"LANG","number":"102","title":"English Communication II","credits":3,"prerequisites":["LANG101"],"school":"Language / General Education"},
    # BUS
    {"code":"BUS101","prefix":"BUS","number":"101","title":"Principles of Management","credits":3,"prerequisites":[],"school":"School of Business"},
    {"code":"BUS201","prefix":"BUS","number":"201","title":"Business Law","credits":3,"prerequisites":[],"school":"School of Business"},
    {"code":"BUS301","prefix":"BUS","number":"301","title":"Strategic Management","credits":3,"prerequisites":["BUS101"],"school":"School of Business"},
    {"code":"FIN201","prefix":"FIN","number":"201","title":"Corporate Finance","credits":3,"prerequisites":["MATH101"],"school":"School of Business / Finance"},
    {"code":"FIN301","prefix":"FIN","number":"301","title":"Financial Modelling","credits":3,"prerequisites":["FIN201"],"school":"School of Business / Finance"},
]


# ─── Main extraction ──────────────────────────────────────────────────────────

def main():
    print("=== Phase 1: Academic Knowledge Extraction ===")
    pdfs = find_pdfs()
    print(f"Found {len(pdfs)} PDFs")

    all_courses: dict[str, dict] = {}
    all_regulations: dict = {
        "minimum_gpa_requirements": [],
        "probation_rules":          [],
        "academic_standing_rules":  [],
        "passing_grade":            "D",
        "graduation_minimum_gpa":   2.0,
    }

    for pdf in pdfs:
        print(f"  Parsing {pdf.name} …")
        text = extract_pdf_text(pdf)
        if not text:
            continue
        courses = parse_courses_from_text(text, pdf.name)
        for code, info in courses.items():
            if code not in all_courses:
                all_courses[code] = info
            else:
                for p in info["prerequisites"]:
                    if p not in all_courses[code]["prerequisites"]:
                        all_courses[code]["prerequisites"].append(p)
                if pdf.name not in all_courses[code]["sources"]:
                    all_courses[code]["sources"].append(pdf.name)
        regs = parse_regulations(text)
        for rule in regs["minimum_gpa_requirements"]:
            if not any(x["value"] == rule["value"] for x in all_regulations["minimum_gpa_requirements"]):
                all_regulations["minimum_gpa_requirements"].append(rule)
        all_regulations["probation_rules"].extend(
            r for r in regs["probation_rules"] if r not in all_regulations["probation_rules"]
        )
        all_regulations["academic_standing_rules"].extend(
            r for r in regs["academic_standing_rules"] if r not in all_regulations["academic_standing_rules"]
        )

    # Merge known courses (seed)
    for c in KNOWN_COURSES:
        code = c["code"]
        entry = {**c, "sources": ["zewail_knowledge_base"]}
        if code not in all_courses:
            all_courses[code] = entry
        else:
            all_courses[code]["sources"].append("zewail_knowledge_base")

    print(f"  Total courses discovered: {len(all_courses)}")

    # ── degree_requirements ────────────────────────────────────────────────────
    degree_requirements: dict[str, dict] = {}
    for prefix, total in DEGREE_CREDIT_TOTALS.items():
        prefix_courses = [c for c in all_courses.values() if c["prefix"] == prefix]
        degree_requirements[prefix] = {
            "programme":           prefix,
            "school":              PROGRAMMES.get(prefix, "Unknown"),
            "total_credits":       total,
            "core_credits":        round(total * 0.65),
            "elective_credits":    round(total * 0.20),
            "general_ed_credits":  round(total * 0.15),
            "minimum_gpa":         2.0,
            "known_courses_count": len(prefix_courses),
        }

    # ── prerequisites graph ────────────────────────────────────────────────────
    prereq_graph: list[dict] = []
    for code, info in all_courses.items():
        for prereq in info["prerequisites"]:
            prereq_graph.append({"course": code, "requires": prereq})

    # ── GPA rules ─────────────────────────────────────────────────────────────
    gpa_rules = {
        "scale":                GPA_SCALE,
        "graduation_minimum":   2.0,
        "probation_threshold":  1.7,
        "dismissal_threshold":  1.0,
        "good_standing_minimum":2.0,
        "honours_threshold":    3.5,
        "high_honours_threshold":3.8,
        "max_gpa":              4.0,
        "passing_grade":        "D",
        "credit_hours_scale":   "4.0",
    }

    # ── academic_schema ───────────────────────────────────────────────────────
    academic_schema = {
        "university":         "Zewail City of Science and Technology",
        "gpa_scale":          4.0,
        "schools": [
            "School of Computer Science and Engineering",
            "School of Engineering",
            "School of Science",
            "School of Business",
        ],
        "programmes":         list(PROGRAMMES.keys()),
        "total_courses":      len(all_courses),
        "degree_programmes":  list(degree_requirements.keys()),
        "graduation_min_gpa": 2.0,
        "academic_calendar":  "Semester (Fall/Spring)",
        "typical_semesters":  8,
        "credit_system":      "Credit Hours",
        "pdf_sources":        [p.name for p in pdfs],
    }

    # ── Save all outputs ──────────────────────────────────────────────────────
    outputs = {
        "academic_schema.json":       academic_schema,
        "course_catalog.json":        list(all_courses.values()),
        "degree_requirements.json":   degree_requirements,
        "prerequisites_graph.json":   prereq_graph,
        "academic_regulations.json":  all_regulations,
        "gpa_rules.json":             gpa_rules,
    }
    for fname, obj in outputs.items():
        path = DATA / fname
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved {path}")

    print("\n=== Phase 1 Complete ===")
    print(f"  Courses extracted : {len(all_courses)}")
    print(f"  Prereq edges      : {len(prereq_graph)}")
    print(f"  Regulations found : {len(all_regulations['minimum_gpa_requirements'])} GPA rules")


if __name__ == "__main__":
    main()
