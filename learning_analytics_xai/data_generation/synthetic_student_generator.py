"""
Phase 2 — Synthetic Student Data Generator (Kaggle/Zenodo-calibrated)

Strategy (as per project spec, Part A + B + C):
  Part A: Download public Zenodo student performance dataset (14,003 records).
          Extract realistic distributions and correlations.
  Part B: Map those distributions to actual Zewail City courses and credit system.
  Part C: Generate 7,500 synthetic students using the calibrated distributions
          with real correlation structure preserved.

Real distributions extracted from Zenodo merged_dataset.csv:
  ExamScore        : mean=70.35 std=17.69 range=[40,100] (p25=55, p50=70, p75=86)
  Attendance       : mean=80.19 std=11.47 range=[60,100] (p25=70, p50=80, p75=90)
  AssignmentCompl  : mean=74.50 std=14.63 range=[50,100] (p25=62, p50=74, p75=87)
  StudyHours/week  : mean=19.99 std=5.89  range=[5,44]

Realistic educational correlations added (from research literature):
  Attendance -> ExamScore         r ≈ 0.50
  AssignmentCompl -> ExamScore    r ≈ 0.45
  StudyHours -> ExamScore         r ≈ 0.35
  Attendance -> AssignmentCompl   r ≈ 0.40

GPA computed via Zewail 4.0 scale.
Risk: Low/Medium/High based on cum_gpa, failed courses, attendance.

Outputs:
  ../data/kaggle_reference_stats.json  — distribution stats from real data
  ../data/synthetic_students.csv       — per-course records
  ../data/students_summary.csv         — per-student aggregated
"""
from __future__ import annotations

import io
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)

# ── Zenodo dataset URL (CC-BY 4.0) ───────────────────────────────────────────
ZENODO_URL = "https://zenodo.org/records/16459132/files/merged_dataset.csv"

# ── Programme metadata ────────────────────────────────────────────────────────
PROGRAMMES: dict[str, dict] = {
    "CSAI": {"school": "CS&AI",  "total_credits": 132, "hard_frac": 0.55},
    "DSAI": {"school": "CS&AI",  "total_credits": 132, "hard_frac": 0.62},
    "SWE":  {"school": "CS&AI",  "total_credits": 132, "hard_frac": 0.40},
    "MECH": {"school": "ENGR",   "total_credits": 140, "hard_frac": 0.73},
    "EEE":  {"school": "ENGR",   "total_credits": 140, "hard_frac": 0.80},
    "CIV":  {"school": "ENGR",   "total_credits": 140, "hard_frac": 0.75},
    "MATH": {"school": "SCI",    "total_credits": 132, "hard_frac": 0.82},
    "PHYS": {"school": "SCI",    "total_credits": 132, "hard_frac": 0.80},
    "CHEM": {"school": "SCI",    "total_credits": 132, "hard_frac": 0.71},
    "BUS":  {"school": "BUS",    "total_credits": 114, "hard_frac": 0.18},
    "FIN":  {"school": "BUS",    "total_credits": 114, "hard_frac": 0.22},
}

PROG_WEIGHTS = [0.22, 0.18, 0.10, 0.08, 0.07, 0.05, 0.07, 0.05, 0.04, 0.09, 0.05]

# (code, title, credits, is_hard)
PROG_COURSES: dict[str, list] = {
    "CSAI": [
        ("CSAI101","Intro to Programming",3,False),
        ("MATH101","Calculus I",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("PHYS101","Physics I",3,True),
        ("LANG101","English I",3,False),
        ("CSAI102","Programming II",3,False),
        ("MATH102","Calculus II",3,True),
        ("MATH201","Linear Algebra",3,True),
        ("CSAI201","Data Structures",3,True),
        ("LANG102","English II",3,False),
        ("CSAI202","Algorithms",3,True),
        ("CSAI203","Database Systems",3,False),
        ("CSAI204","Operating Systems",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("HUMA102","Critical Thinking",3,False),
        ("CSAI205","Computer Networks",3,False),
        ("CSAI206","Software Engineering",3,False),
        ("CSAI301","Machine Learning",3,True),
        ("DSAI201","Data Mining",3,True),
        ("MATH202","Differential Equations",3,True),
        ("CSAI302","Deep Learning",3,True),
        ("CSAI303","Computer Vision",3,True),
        ("CSAI304","NLP",3,True),
        ("CSAI207","Cybersecurity",3,False),
        ("CSAI208","Cloud Computing",3,False),
        ("DSAI302","Data Visualization",3,False),
        ("DSAI202","Statistical Inference",3,True),
        ("CSAI305","Reinforcement Learning",3,True),
        ("CSAI209","Web Development",3,False),
        ("DSAI301","Big Data Analytics",3,True),
        ("DSAI401","Recommender Systems",3,True),
        ("HUMA201","Research Methods",3,False),
        ("MATH301","Numerical Methods",3,True),
        ("CSAI401","AI Ethics",3,False),
        ("BUS101","Principles of Management",3,False),
        ("CSAI402","AI Research Project",6,True),
        ("PHYS102","Physics II",3,True),
        ("FIN201","Corporate Finance",3,False),
    ],
    "DSAI": [
        ("CSAI101","Intro to Programming",3,False),
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH201","Linear Algebra",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("LANG101","English I",3,False),
        ("HUMA101","Academic Writing",3,False),
        ("DSAI202","Statistical Inference",3,True),
        ("CSAI201","Data Structures",3,True),
        ("CSAI203","Database Systems",3,False),
        ("DSAI201","Data Mining",3,True),
        ("CSAI301","Machine Learning",3,True),
        ("DSAI301","Big Data Analytics",3,True),
        ("DSAI302","Data Visualization",3,False),
        ("CSAI302","Deep Learning",3,True),
        ("CSAI304","NLP",3,True),
        ("DSAI401","Recommender Systems",3,True),
        ("MATH301","Numerical Methods",3,True),
        ("CSAI401","AI Ethics",3,False),
        ("BUS101","Principles of Management",3,False),
        ("LANG102","English II",3,False),
    ],
    "SWE": [
        ("CSAI101","Intro to Programming",3,False),
        ("CSAI102","Programming II",3,False),
        ("CSAI201","Data Structures",3,True),
        ("CSAI202","Algorithms",3,True),
        ("CSAI203","Database Systems",3,False),
        ("CSAI206","Software Engineering",3,False),
        ("CSAI205","Computer Networks",3,False),
        ("CSAI209","Web Development",3,False),
        ("CSAI207","Cybersecurity",3,False),
        ("CSAI208","Cloud Computing",3,False),
        ("MATH101","Calculus I",3,True),
        ("MATH201","Linear Algebra",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
        ("CSAI301","Machine Learning",3,True),
    ],
    "MECH": [
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH202","Differential Equations",3,True),
        ("PHYS101","Physics I",3,True),
        ("PHYS102","Physics II",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("MATH201","Linear Algebra",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
        ("BUS101","Principles of Management",3,False),
    ],
    "EEE": [
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH202","Differential Equations",3,True),
        ("PHYS101","Physics I",3,True),
        ("PHYS102","Physics II",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("MATH201","Linear Algebra",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
    ],
    "CIV": [
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH202","Differential Equations",3,True),
        ("PHYS101","Physics I",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("MATH201","Linear Algebra",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
    ],
    "MATH": [
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH201","Linear Algebra",3,True),
        ("MATH202","Differential Equations",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("MATH301","Numerical Methods",3,True),
        ("PHYS101","Physics I",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
        ("DSAI202","Statistical Inference",3,True),
    ],
    "PHYS": [
        ("PHYS101","Physics I",3,True),
        ("PHYS102","Physics II",3,True),
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("MATH201","Linear Algebra",3,True),
        ("MATH202","Differential Equations",3,True),
        ("MATH203","Probability & Statistics",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
    ],
    "CHEM": [
        ("MATH101","Calculus I",3,True),
        ("MATH102","Calculus II",3,True),
        ("PHYS101","Physics I",3,True),
        ("CSAI101","Intro to Programming",3,False),
        ("MATH203","Probability & Statistics",3,True),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
    ],
    "BUS": [
        ("BUS101","Principles of Management",3,False),
        ("BUS201","Business Law",3,False),
        ("BUS301","Strategic Management",3,False),
        ("FIN201","Corporate Finance",3,False),
        ("FIN301","Financial Modelling",3,True),
        ("MATH101","Calculus I",3,True),
        ("MATH203","Probability & Statistics",3,False),
        ("CSAI101","Intro to Programming",3,False),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
        ("LANG102","English II",3,False),
    ],
    "FIN": [
        ("FIN201","Corporate Finance",3,False),
        ("FIN301","Financial Modelling",3,True),
        ("BUS101","Principles of Management",3,False),
        ("BUS201","Business Law",3,False),
        ("MATH101","Calculus I",3,True),
        ("MATH203","Probability & Statistics",3,False),
        ("CSAI101","Intro to Programming",3,False),
        ("HUMA101","Academic Writing",3,False),
        ("LANG101","English I",3,False),
    ],
}

# Prerequisite chain boosts (title -> (prereq_title, strength))
CHAIN_MAP: dict[str, tuple[str, float]] = {
    "Deep Learning":          ("Machine Learning",      0.14),
    "Computer Vision":        ("Deep Learning",         0.16),
    "NLP":                    ("Machine Learning",      0.12),
    "Reinforcement Learning": ("Machine Learning",      0.10),
    "Recommender Systems":    ("Machine Learning",      0.09),
    "Statistical Inference":  ("Probability & Statistics", 0.14),
    "Big Data Analytics":     ("Data Mining",           0.10),
    "Calculus II":            ("Calculus I",            0.18),
    "Linear Algebra":         ("Calculus I",            0.12),
    "Differential Equations": ("Calculus II",           0.16),
    "Numerical Methods":      ("Differential Equations",0.13),
    "Physics II":             ("Physics I",             0.18),
    "Programming II":         ("Intro to Programming",  0.16),
    "Data Structures":        ("Programming II",        0.18),
    "Algorithms":             ("Data Structures",       0.18),
    "Financial Modelling":    ("Corporate Finance",     0.15),
    "Calculus II":            ("Calculus I",            0.18),
}


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(x, lo, hi))


def score_to_gpa(score: float) -> float:
    """Zewail City 4.0 GPA scale."""
    if score >= 93: return 4.0
    if score >= 90: return 3.7
    if score >= 87: return 3.3
    if score >= 83: return 3.0
    if score >= 80: return 2.7
    if score >= 77: return 2.3
    if score >= 73: return 2.0
    if score >= 70: return 1.7
    if score >= 67: return 1.3
    if score >= 60: return 1.0
    return 0.0


# ── Part A: Download and extract real distributions ───────────────────────────

def fetch_real_stats() -> dict:
    """Download Zenodo dataset and extract feature distributions."""
    print("  Downloading real dataset from Zenodo (CC-BY 4.0) ...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode   = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(ZENODO_URL, context=ctx, timeout=60) as r:
            raw = r.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
        print(f"  Downloaded {len(df):,} real student records.")
    except Exception as e:
        print(f"  [WARN] Could not fetch real data ({e}). Using empirical defaults.")
        # Empirical defaults from prior download analysis
        return _empirical_defaults()

    # Use Zenodo for distribution SHAPE (std, percentile ratios) only.
    # Zenodo correlations are near-zero (artificially generated dataset) —
    # override with research-backed values (Credé & Kuncel 2008; Rakes et al.).
    # Shift exam mean to 87 to reflect Zewail's selective admission (top-tier STEM).
    exam_s   = _dist_stats(df["ExamScore"])
    exam_s["mean"] = 87.0   # selective university shift (+16.65 pts from Zenodo mean)

    att_s    = _dist_stats(df["Attendance"])
    asg_s    = _dist_stats(df["AssignmentCompletion"])
    sh_s     = _dist_stats(df["StudyHours"])

    stats = {
        "n_real_records": int(len(df)),
        "exam_score":          exam_s,
        "attendance":          att_s,
        "assignment_complete": asg_s,
        "study_hours":         sh_s,
        # Research-backed educational correlations (Zenodo's are near zero)
        "correlations": {
            "att_exam":    0.44,   # Credé & Kuncel (2008) meta-analysis
            "assign_exam": 0.45,   # Rakes et al. — assignment completion vs grade
            "study_exam":  0.26,   # Swanberg & Martinsen (2010)
            "att_assign":  0.38,   # moderate co-occurrence
        },
        "final_grade_dist": df["FinalGrade"].value_counts(normalize=True).sort_index().to_dict(),
    }

    # Save for reference
    out = DATA / "kaggle_reference_stats.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  Saved reference stats -> {out}")
    return stats


def _dist_stats(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std":  float(series.std()),
        "min":  float(series.min()),
        "p10":  float(series.quantile(0.10)),
        "p25":  float(series.quantile(0.25)),
        "p50":  float(series.quantile(0.50)),
        "p75":  float(series.quantile(0.75)),
        "p90":  float(series.quantile(0.90)),
        "max":  float(series.max()),
    }


def _empirical_defaults() -> dict:
    """Hard-coded stats from the Zenodo dataset (in case download fails)."""
    return {
        "n_real_records": 14003,
        "exam_score":          {"mean":70.35,"std":17.69,"min":40,"p25":55,"p50":70,"p75":86,"p90":94,"max":100},
        "attendance":          {"mean":80.19,"std":11.47,"min":60,"p25":70,"p50":80,"p75":90,"p90":96,"max":100},
        "assignment_complete": {"mean":74.50,"std":14.63,"min":50,"p25":62,"p50":74,"p75":87,"p90":95,"max":100},
        "study_hours":         {"mean":19.99,"std":5.89, "min":5, "p25":16,"p50":20,"p75":24,"p90":28,"max":44},
        "correlations": {
            # Zenodo correlations are near-zero (artificially generated dataset).
            # We use RESEARCH-BACKED correlations for realistic data:
            # Credé & Kuncel (2008): att-GPA r=0.44
            # Rakes et al.: assignment-GPA r=0.45
            # Swanberg & Martinsen (2010): study time-GPA r=0.26
            "att_exam":    0.44,
            "assign_exam": 0.45,
            "study_exam":  0.26,
            "att_assign":  0.38,
        },
        "final_grade_dist": {0: 0.274, 1: 0.236, 2: 0.258, 3: 0.232},
    }


# ── Part B+C: Generate correlated synthetic Zewail students ──────────────────

def make_correlated_student(stats: dict) -> dict:
    """
    Generate ONE student's latent traits using the real distribution stats.
    Uses Cholesky decomposition to produce correlated (attendance, assignments,
    study_hours, exam_base) matching Zenodo distributions + research correlations.
    """
    # ── Build correlation matrix ──────────────────────────────────────────────
    # Variables: [attendance, assignments, study_hours, exam_base]
    r_ae = stats["correlations"]["att_exam"]       # 0.44
    r_asg_e = stats["correlations"]["assign_exam"] # 0.45
    r_sh_e = stats["correlations"]["study_exam"]   # 0.26
    r_a_asg = stats["correlations"]["att_assign"]  # 0.38
    # estimate remaining correlations
    r_a_sh   = 0.30   # attendance correlates with study hours
    r_asg_sh = 0.32   # assignments correlate with study hours

    corr = np.array([
        [1.00,  r_a_asg, r_a_sh,   r_ae    ],
        [r_a_asg, 1.00,  r_asg_sh, r_asg_e ],
        [r_a_sh,  r_asg_sh, 1.00,  r_sh_e  ],
        [r_ae,    r_asg_e,  r_sh_e, 1.00   ],
    ])
    # Cholesky of the correlation matrix
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # fallback if not positive definite (numerical issue)
        corr = np.eye(4)
        L    = np.eye(4)

    # ── Draw correlated standard normals ─────────────────────────────────────
    z = RNG.standard_normal(4)
    corr_z = L @ z  # shape (4,)

    # ── Map to original scales using real distribution stats ─────────────────
    att_s   = stats["attendance"]
    asg_s   = stats["assignment_complete"]
    sh_s    = stats["study_hours"]
    ex_s    = stats["exam_score"]

    attendance    = clamp(att_s["mean"]  + att_s["std"]  * corr_z[0], att_s["min"],  100)
    assignments   = clamp(asg_s["mean"] + asg_s["std"]  * corr_z[1], asg_s["min"],  100)
    study_hours   = clamp(sh_s["mean"]  + sh_s["std"]   * corr_z[2], sh_s["min"],   sh_s["max"])
    base_exam     = clamp(ex_s["mean"]  + ex_s["std"]   * corr_z[3], ex_s["min"],   100)

    return {
        "attendance":   attendance,
        "assignments":  assignments,
        "study_hours":  study_hours,
        "base_exam":    base_exam,
    }


def generate_student(student_id: int, prog: str, stats: dict) -> list[dict]:
    """Generate one student's full per-course records."""
    prog_info  = PROGRAMMES[prog]
    courses    = PROG_COURSES.get(prog, PROG_COURSES["CSAI"])
    hard_frac  = prog_info["hard_frac"]

    # ── Draw correlated student traits from real distributions ────────────────
    traits = make_correlated_student(stats)
    attendance_base = traits["attendance"]
    assign_base     = traits["assignments"]
    study_hours     = traits["study_hours"]
    base_exam       = traits["base_exam"]

    # ── Derived traits ────────────────────────────────────────────────────────
    # quiz score correlates with base_exam (same student ability)
    quiz_base = clamp(base_exam + float(RNG.normal(0, 5)))
    # lab score: intermediate between assignments and exams
    lab_base  = clamp(0.55 * assign_base + 0.35 * base_exam + 0.10 * quiz_base + float(RNG.normal(0, 4)))
    # anxiety: inversely related to study_hours and attendance (mild)
    anxiety   = clamp((1 - study_hours / 44.0) * 0.6 + float(RNG.normal(0, 0.12)), 0, 1)

    # ── Semester ──────────────────────────────────────────────────────────────
    total_creds    = prog_info["total_credits"]
    n_semesters    = 8 if total_creds <= 132 else 9
    semester       = int(RNG.integers(1, n_semesters + 1))
    courses_per_sem = 5
    courses_taken  = min(semester * courses_per_sem, len(courses))
    course_list    = courses[:courses_taken]

    records = []
    failed_so_far = 0
    prev_scores: dict[str, float] = {}

    for sem_idx, (code, title, cred, is_hard) in enumerate(course_list):
        sem_num = (sem_idx // courses_per_sem) + 1

        # ── Per-course attendance (varies ±8% around student base) ───────────
        att = clamp(attendance_base + float(RNG.normal(0, 8)), 40, 100)
        att_multiplier = att / 100.0

        # ── Prerequisite chain boost ──────────────────────────────────────────
        chain_boost = 0.0
        if title in CHAIN_MAP:
            prereq_title, strength = CHAIN_MAP[title]
            prior = prev_scores.get(prereq_title, base_exam)
            # If student did well in prereq, they do better here
            chain_boost = strength * max(0, prior - base_exam) / 15.0

        # ── Hard course penalty ───────────────────────────────────────────────
        # Hard courses reduce scores by ~5 pts on average
        hard_pen = 5 if is_hard else 0

        # ── Semester difficulty drift ─────────────────────────────────────────
        # Courses get harder each semester; 0.8 pts per semester
        sem_drift = 0.8 * (sem_num - 1)

        # ── Effective base score for this course ──────────────────────────────
        eff = base_exam - hard_pen - sem_drift + 5 * chain_boost
        eff = clamp(eff, 30, 100)

        # ── Score components ──────────────────────────────────────────────────
        # Assignments: closely tied to student's base assignment ability
        asgn = clamp(assign_base - 0.5 * hard_pen + float(RNG.normal(0, 6)))
        # Quizzes: tied to in-class attention (attendance effect)
        quiz = clamp(quiz_base * att_multiplier - 0.4 * hard_pen + float(RNG.normal(0, 7)))
        # Labs: practical, less affected by hard penalty for motivated students
        labs = clamp(lab_base - 0.3 * hard_pen + float(RNG.normal(0, 6)))
        # Midterm: affected by anxiety under exam pressure
        mid  = clamp(eff - 6 * anxiety + float(RNG.normal(0, 9)))
        # Final: harder exam; anxiety matters more; attendance helps
        fin  = clamp(
            eff
            + 0.08 * (mid - eff)
            - 9 * anxiety
            + 3 * (att_multiplier - 0.75)
            + float(RNG.normal(0, 10))
        )

        # ── Weighted overall score ────────────────────────────────────────────
        overall = clamp(
            0.10 * quiz
            + 0.15 * asgn
            + 0.15 * labs
            + 0.30 * mid
            + 0.30 * fin
        )

        passed = overall >= 60.0
        if not passed:
            failed_so_far += 1

        prev_scores[title] = overall

        records.append({
            "student_id":          f"STU{student_id:05d}",
            "programme":           prog,
            "school":              prog_info["school"],
            "semester":            sem_num,
            "course_code":         code,
            "course_name":         title,
            "credits":             cred,
            "attendance":          round(att,     1),
            "assignments":         round(asgn,    1),
            "quizzes":             round(quiz,    1),
            "labs":                round(labs,    1),
            "midterm":             round(mid,     1),
            "final_exam":          round(fin,     1),
            "overall_score":       round(overall, 1),
            "passed":              int(passed),
            "course_gpa_points":   score_to_gpa(overall),
            "failed_count_so_far": failed_so_far,
            # Student-level traits (for EDA / XAI validation)
            "attendance_base":     round(attendance_base, 1),
            "study_hours":         round(study_hours,     1),
            "base_exam":           round(base_exam,       1),
            "anxiety":             round(anxiety,         3),
        })

    return records


def compute_summary(records: list[dict]) -> dict:
    if not records:
        return {}
    s = records[0]
    passed = [r for r in records if r["passed"]]
    failed = [r for r in records if not r["passed"]]

    cr = sum(r["credits"] for r in records)
    cp = sum(r["credits"] for r in passed)
    cum_gpa = sum(r["course_gpa_points"] * r["credits"] for r in records) / max(cr, 1)
    cum_gpa = round(cum_gpa, 3)

    if   cum_gpa >= 3.5: gpa_band = "High (>=3.5)"
    elif cum_gpa >= 3.0: gpa_band = "Good (3.0-3.5)"
    elif cum_gpa >= 2.5: gpa_band = "Average (2.5-3.0)"
    elif cum_gpa >= 2.0: gpa_band = "Below Average (2.0-2.5)"
    else:                gpa_band = "At Risk (<2.0)"

    avg_att  = float(np.mean([r["attendance"] for r in records]))
    n_failed = len(failed)

    if   cum_gpa < 2.0 or n_failed >= 4 or avg_att < 55: risk = "High Risk"
    elif cum_gpa < 2.5 or n_failed >= 2 or avg_att < 68: risk = "Medium Risk"
    else:                                                  risk = "Low Risk"

    return {
        "student_id":         s["student_id"],
        "programme":          s["programme"],
        "school":             s["school"],
        "semester":           max(r["semester"] for r in records),
        "credits_registered": cr,
        "credits_passed":     cp,
        "failed_courses":     n_failed,
        "failed_ratio":       round(n_failed / max(len(records), 1), 4),
        "cumulative_gpa":     cum_gpa,
        "gpa_band":           gpa_band,
        "risk_level":         risk,
        "avg_attendance":     round(avg_att, 1),
        "avg_assignments":    round(float(np.mean([r["assignments"]   for r in records])), 1),
        "avg_quizzes":        round(float(np.mean([r["quizzes"]       for r in records])), 1),
        "avg_labs":           round(float(np.mean([r["labs"]          for r in records])), 1),
        "avg_midterm":        round(float(np.mean([r["midterm"]       for r in records])), 1),
        "avg_final":          round(float(np.mean([r["final_exam"]    for r in records])), 1),
        "avg_overall":        round(float(np.mean([r["overall_score"] for r in records])), 1),
        "study_hours":        s["study_hours"],
        "base_exam":          s["base_exam"],
        "anxiety":            s["anxiety"],
        "attendance_base":    s["attendance_base"],
    }


def main():
    print("=== Phase 2: Kaggle-Calibrated Synthetic Student Data Generation ===")
    print()

    # Part A: get real distributions
    stats = fetch_real_stats()

    print()
    print("  Reference distribution (real data):")
    for feat in ("exam_score", "attendance", "assignment_complete"):
        d = stats[feat]
        print(f"    {feat:20s}: mean={d['mean']:.1f}  std={d['std']:.1f}  "
              f"p25={d['p25']:.0f}  p50={d['p50']:.0f}  p75={d['p75']:.0f}")

    corr = stats["correlations"]
    print(f"\n  Correlations used:")
    print(f"    Attendance -> Exam Score  : {corr['att_exam']:.2f}")
    print(f"    Assignments -> Exam Score : {corr['assign_exam']:.2f}")
    print(f"    Study Hours -> Exam Score : {corr['study_exam']:.2f}")

    # Part B + C: generate synthetic Zewail students
    print()
    N = 7_500
    progs   = list(PROGRAMMES.keys())
    weights = [0.22, 0.18, 0.10, 0.08, 0.07, 0.05, 0.07, 0.05, 0.04, 0.09, 0.05]
    selected = RNG.choice(progs, size=N, p=weights)

    all_records:   list[dict] = []
    all_summaries: list[dict] = []

    for i, prog in enumerate(selected):
        recs    = generate_student(i + 1, str(prog), stats)
        summary = compute_summary(recs)
        all_records.extend(recs)
        if summary:
            all_summaries.append(summary)
        if (i + 1) % 1500 == 0:
            print(f"  {i+1:,} / {N:,} students generated ...")

    df_raw = pd.DataFrame(all_records)
    df_sum = pd.DataFrame(all_summaries)

    df_raw.to_csv(DATA / "synthetic_students.csv",  index=False)
    df_sum.to_csv(DATA / "students_summary.csv",    index=False)

    print(f"\n  Raw course records : {len(df_raw):,}")
    print(f"  Student summaries  : {len(df_sum):,}")
    print("\n  Risk distribution:")
    print(df_sum["risk_level"].value_counts().to_string())
    print("\n  GPA band distribution:")
    print(df_sum["gpa_band"].value_counts().to_string())
    print("\n  Cumulative GPA stats:")
    print(df_sum["cumulative_gpa"].describe().round(3).to_string())
    print("\n  Avg Attendance stats:")
    print(df_sum["avg_attendance"].describe().round(1).to_string())
    print("\n  Avg Overall Score stats:")
    print(df_sum["avg_overall"].describe().round(1).to_string())

    # Validate key correlations in generated data
    print("\n  Correlation validation (generated data):")
    corr_gen = df_sum[["avg_attendance","avg_overall","cumulative_gpa"]].corr()
    print(f"    avg_attendance vs avg_overall : {corr_gen.loc['avg_attendance','avg_overall']:.3f}")
    print(f"    avg_attendance vs cum_gpa     : {corr_gen.loc['avg_attendance','cumulative_gpa']:.3f}")
    print(f"    avg_overall    vs cum_gpa     : {corr_gen.loc['avg_overall','cumulative_gpa']:.3f}")

    print("\n=== Phase 2 Complete ===")


if __name__ == "__main__":
    main()
