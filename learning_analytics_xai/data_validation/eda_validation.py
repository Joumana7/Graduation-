"""
Phase 3 — EDA and Data Validation
Validates the synthetic student dataset for realism and correctness.

Checks:
  1. GPA distribution (shape, mean, std, range)
  2. Attendance distribution vs Zenodo reference
  3. Risk level split
  4. Correlation matrix (attendance -> GPA, assignments -> GPA, etc.)
  5. Programme distribution
  6. Failure rate analysis
  7. Score component distributions
  8. Prerequisite chain correlations (ML->DL->CV)

Outputs:
  ../reports/data_validation_report.md
  ../reports/eda_figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

HERE    = Path(__file__).parent
DATA    = HERE.parent / "data"
REPORTS = HERE.parent / "reports"
FIGS    = REPORTS / "eda_figures"
REPORTS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

RAW_PATH = DATA / "synthetic_students.csv"
SUM_PATH = DATA / "students_summary.csv"
REF_PATH = DATA / "kaggle_reference_stats.json"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df_raw = pd.read_csv(RAW_PATH)
    df_sum = pd.read_csv(SUM_PATH)
    ref    = json.loads(REF_PATH.read_text(encoding="utf-8")) if REF_PATH.exists() else {}
    return df_raw, df_sum, ref


def check_distributions(df_sum: pd.DataFrame, ref: dict) -> list[str]:
    """Run statistical checks and return a list of findings."""
    findings = []

    # ── GPA ──────────────────────────────────────────────────────────────────
    gpa = df_sum["cumulative_gpa"]
    findings.append(f"GPA mean={gpa.mean():.3f}  std={gpa.std():.3f}  "
                    f"min={gpa.min():.2f}  max={gpa.max():.2f}")
    if 1.5 <= gpa.mean() <= 3.5:
        findings.append("PASS: GPA mean is within realistic university range (1.5-3.5)")
    else:
        findings.append(f"WARN: GPA mean {gpa.mean():.2f} outside typical range")

    # Normality test on GPA
    if len(df_sum) >= 50:
        _, p = scipy_stats.normaltest(gpa.dropna())
        findings.append(f"GPA normality test p-value: {p:.4f} "
                        f"({'approximately normal' if p > 0.05 else 'non-normal - expected for mixed population'})")

    # ── Attendance ────────────────────────────────────────────────────────────
    att = df_sum["avg_attendance"]
    ref_att = ref.get("attendance", {})
    findings.append(f"Attendance mean={att.mean():.1f}%  std={att.std():.1f}%")
    if ref_att:
        findings.append(f"Zenodo reference attendance mean={ref_att['mean']:.1f}%  "
                        f"std={ref_att['std']:.1f}%")
        diff = abs(att.mean() - ref_att["mean"])
        if diff < 5:
            findings.append(f"PASS: Attendance mean within 5% of reference ({diff:.1f}%)")
        else:
            findings.append(f"WARN: Attendance mean differs by {diff:.1f}% from reference")

    # ── Correlations ──────────────────────────────────────────────────────────
    corr_att_gpa  = df_sum["avg_attendance"].corr(df_sum["cumulative_gpa"])
    corr_asg_gpa  = df_sum["avg_assignments"].corr(df_sum["cumulative_gpa"])
    corr_mid_gpa  = df_sum["avg_midterm"].corr(df_sum["cumulative_gpa"])
    corr_fail_gpa = df_sum["failed_courses"].corr(df_sum["cumulative_gpa"])

    findings.append(f"Correlation: Attendance  vs GPA = {corr_att_gpa:.3f}  "
                    f"(research target ~0.44)")
    findings.append(f"Correlation: Assignments vs GPA = {corr_asg_gpa:.3f}  "
                    f"(research target ~0.45)")
    findings.append(f"Correlation: Midterm     vs GPA = {corr_mid_gpa:.3f}  "
                    f"(expected 0.60-0.90)")
    findings.append(f"Correlation: Failed#     vs GPA = {corr_fail_gpa:.3f}  "
                    f"(expected negative -0.50 to -0.80)")

    for label, r, lo, hi in [
        ("Attendance-GPA",   corr_att_gpa,  0.30, 0.70),
        ("Assignments-GPA",  corr_asg_gpa,  0.30, 0.70),
        ("Midterm-GPA",      corr_mid_gpa,  0.50, 0.99),
        ("FailedCourses-GPA",corr_fail_gpa,-0.90,-0.30),
    ]:
        if lo <= r <= hi:
            findings.append(f"PASS: {label} correlation {r:.3f} in expected range [{lo},{hi}]")
        else:
            findings.append(f"WARN: {label} correlation {r:.3f} outside expected range [{lo},{hi}]")

    # ── Failure rate ──────────────────────────────────────────────────────────
    overall_fail_rate = 1 - df_sum["credits_passed"].sum() / df_sum["credits_registered"].sum()
    findings.append(f"Course failure rate: {overall_fail_rate*100:.1f}% of credit-weighted courses")
    if 0.05 <= overall_fail_rate <= 0.30:
        findings.append("PASS: Course failure rate within realistic 5-30% range")
    else:
        findings.append(f"WARN: Failure rate {overall_fail_rate*100:.1f}% outside expected range")

    # ── Risk distribution ─────────────────────────────────────────────────────
    risk_dist = df_sum["risk_level"].value_counts(normalize=True)
    for k, v in risk_dist.items():
        findings.append(f"Risk level '{k}': {v*100:.1f}%")

    # ── Programme distribution ────────────────────────────────────────────────
    prog_dist = df_sum["programme"].value_counts()
    findings.append(f"Programme distribution: {prog_dist.to_dict()}")

    return findings


def plot_distributions(df_raw: pd.DataFrame, df_sum: pd.DataFrame, ref: dict):
    """Generate EDA figures."""
    # ── Figure 1: Score distributions ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Synthetic Student Data — Score Distributions", fontsize=14, fontweight="bold")

    cols = ["attendance", "assignments", "quizzes", "midterm", "final_exam", "overall_score"]
    labels = ["Attendance (%)", "Assignments (0-100)", "Quizzes (0-100)",
              "Midterm (0-100)", "Final Exam (0-100)", "Overall Score (0-100)"]
    colors = ["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8"]

    for ax, col, label, color in zip(axes.flat, cols, labels, colors):
        ax.hist(df_raw[col], bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(label, fontweight="bold", fontsize=10)
        ax.set_xlabel("Score", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        mean_val = df_raw[col].mean()
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_val:.1f}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "01_score_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 01_score_distributions.png")

    # ── Figure 2: GPA and Risk distribution ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Cumulative GPA and Risk Level Distribution", fontsize=14, fontweight="bold")

    # GPA histogram
    axes[0].hist(df_sum["cumulative_gpa"], bins=40, color="#7C3AED", edgecolor="white", alpha=0.85)
    axes[0].set_title("Cumulative GPA Distribution", fontweight="bold")
    axes[0].set_xlabel("Cumulative GPA (0-4.0)")
    axes[0].set_ylabel("Student Count")
    axes[0].axvline(df_sum["cumulative_gpa"].mean(), color="red", linestyle="--",
                    linewidth=2, label=f"Mean={df_sum['cumulative_gpa'].mean():.2f}")
    axes[0].axvline(2.0, color="orange", linestyle=":", linewidth=1.5, label="Min (2.0)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Risk pie chart
    risk_counts = df_sum["risk_level"].value_counts()
    colors_risk = ["#E74C3C", "#F39C12", "#27AE60"]
    axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct="%1.1f%%",
                colors=colors_risk, startangle=90)
    axes[1].set_title("Risk Level Distribution", fontweight="bold")

    # GPA band bar chart
    gpa_bands = df_sum["gpa_band"].value_counts().sort_index()
    bar_colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71", "#3498DB"]
    axes[2].barh(range(len(gpa_bands)), gpa_bands.values,
                 color=bar_colors[:len(gpa_bands)], edgecolor="white")
    axes[2].set_yticks(range(len(gpa_bands)))
    axes[2].set_yticklabels(gpa_bands.index, fontsize=9)
    axes[2].set_title("GPA Band Distribution", fontweight="bold")
    axes[2].set_xlabel("Student Count")
    axes[2].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "02_gpa_risk_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_gpa_risk_distribution.png")

    # ── Figure 3: Correlation heatmap ─────────────────────────────────────────
    corr_cols = ["avg_attendance", "avg_assignments", "avg_quizzes",
                 "avg_labs", "avg_midterm", "avg_final", "cumulative_gpa", "failed_courses"]
    labels_map = {
        "avg_attendance": "Attendance",
        "avg_assignments": "Assignments",
        "avg_quizzes": "Quizzes",
        "avg_labs": "Labs",
        "avg_midterm": "Midterm",
        "avg_final": "Final",
        "cumulative_gpa": "Cum. GPA",
        "failed_courses": "Failed #",
    }
    corr_df = df_sum[corr_cols].corr()
    corr_df.columns = [labels_map[c] for c in corr_df.columns]
    corr_df.index   = [labels_map[c] for c in corr_df.index]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(corr_df.index, fontsize=10)
    for i in range(len(corr_df)):
        for j in range(len(corr_df.columns)):
            val = corr_df.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(val) < 0.5 else "white")
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(FIGS / "03_correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 03_correlation_heatmap.png")

    # ── Figure 4: Programme and semester distributions ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Programme and Semester Distribution", fontsize=13, fontweight="bold")

    prog_gpa = df_sum.groupby("programme")["cumulative_gpa"].mean().sort_values(ascending=False)
    prog_colors = plt.cm.viridis(np.linspace(0, 1, len(prog_gpa)))
    axes[0].barh(prog_gpa.index, prog_gpa.values, color=prog_colors, edgecolor="white")
    axes[0].set_title("Mean GPA by Programme", fontweight="bold")
    axes[0].set_xlabel("Mean Cumulative GPA")
    axes[0].axvline(2.0, color="red", linestyle="--", alpha=0.7, label="Min. Standing (2.0)")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="x", alpha=0.3)
    for i, (prog, gpa_val) in enumerate(prog_gpa.items()):
        axes[0].text(gpa_val + 0.02, i, f"{gpa_val:.2f}", va="center", fontsize=9)

    sem_dist = df_sum["semester"].value_counts().sort_index()
    axes[1].bar(sem_dist.index, sem_dist.values, color="#A855F7", edgecolor="white", alpha=0.85)
    axes[1].set_title("Student Distribution by Semester", fontweight="bold")
    axes[1].set_xlabel("Current Semester")
    axes[1].set_ylabel("Student Count")
    axes[1].set_xticks(range(1, max(sem_dist.index) + 1))
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / "04_programme_semester_dist.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_programme_semester_dist.png")

    # ── Figure 5: Prerequisite chain validation (CSAI students) ─────────────
    csai_raw = df_raw[df_raw["programme"] == "CSAI"]

    chain_pairs = [
        ("Machine Learning", "Deep Learning"),
        ("Deep Learning", "Computer Vision"),
        ("Machine Learning", "NLP"),
        ("Probability & Statistics", "Statistical Inference"),
        ("Calculus I", "Calculus II"),
        ("Data Structures", "Algorithms"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Prerequisite Chain Correlations (CSAI Programme)", fontsize=13, fontweight="bold")

    for ax, (course_a, course_b) in zip(axes.flat, chain_pairs):
        data_a = csai_raw[csai_raw["course_name"] == course_a][["student_id","overall_score"]].rename(columns={"overall_score": course_a})
        data_b = csai_raw[csai_raw["course_name"] == course_b][["student_id","overall_score"]].rename(columns={"overall_score": course_b})
        merged = data_a.merge(data_b, on="student_id")
        if len(merged) > 10:
            r, _ = scipy_stats.pearsonr(merged[course_a], merged[course_b])
            ax.scatter(merged[course_a], merged[course_b], alpha=0.3, s=10, color="#7C3AED")
            z = np.polyfit(merged[course_a], merged[course_b], 1)
            p = np.poly1d(z)
            x_line = np.linspace(merged[course_a].min(), merged[course_a].max(), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=1.5)
            ax.set_title(f"{course_a}\nvs {course_b}\nr={r:.2f}", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"{course_a} Score", fontsize=8)
            ax.set_ylabel(f"{course_b} Score", fontsize=8)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "Insufficient\ndata", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title(f"{course_a} vs {course_b}", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGS / "05_prerequisite_chains.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 05_prerequisite_chains.png")


def generate_report(findings: list[str], df_sum: pd.DataFrame, df_raw: pd.DataFrame) -> str:
    """Generate markdown validation report."""
    n_pass = sum(1 for f in findings if f.startswith("PASS"))
    n_warn = sum(1 for f in findings if f.startswith("WARN"))
    n_total = n_pass + n_warn

    report = f"""# Data Validation Report
## Explainable AI Learning Analytics Platform — Zewail City

**Generated:** Phase 3 — EDA and Data Validation
**Dataset:** Synthetic (Zenodo-calibrated, research-backed correlations)

---

## Summary

| Metric | Value |
|--------|-------|
| Students | {len(df_sum):,} |
| Course Records | {len(df_raw):,} |
| Programmes | {df_sum["programme"].nunique()} |
| Validation Checks | {n_total} ({n_pass} passed, {n_warn} warnings) |

---

## Key Statistics

### Cumulative GPA
| Stat | Value |
|------|-------|
| Mean | {df_sum["cumulative_gpa"].mean():.3f} |
| Std  | {df_sum["cumulative_gpa"].std():.3f} |
| Min  | {df_sum["cumulative_gpa"].min():.3f} |
| Max  | {df_sum["cumulative_gpa"].max():.3f} |
| 25th pct | {df_sum["cumulative_gpa"].quantile(0.25):.3f} |
| Median   | {df_sum["cumulative_gpa"].median():.3f} |
| 75th pct | {df_sum["cumulative_gpa"].quantile(0.75):.3f} |

### Risk Distribution
{df_sum["risk_level"].value_counts().to_frame("Count").assign(Pct=lambda x: (x["Count"]/len(df_sum)*100).round(1)).to_markdown()}

### GPA Band Distribution
{df_sum["gpa_band"].value_counts().to_frame("Count").assign(Pct=lambda x: (x["Count"]/len(df_sum)*100).round(1)).to_markdown()}

### Score Component Statistics (per course)
{df_raw[["attendance","assignments","quizzes","labs","midterm","final_exam","overall_score"]].describe().round(2).to_markdown()}

### Correlation Matrix (student summaries)
"""
    corr_cols = ["avg_attendance", "avg_assignments", "avg_midterm", "cumulative_gpa", "failed_courses"]
    corr_labels = ["Attendance", "Assignments", "Midterm", "Cum GPA", "Failed #"]
    corr_matrix = df_sum[corr_cols].corr()
    corr_matrix.columns = corr_labels
    corr_matrix.index   = corr_labels
    report += corr_matrix.round(3).to_markdown()

    report += "\n\n---\n\n## Validation Findings\n\n"
    for f in findings:
        if f.startswith("PASS"):
            report += f"- ✅ {f}\n"
        elif f.startswith("WARN"):
            report += f"- ⚠️ {f}\n"
        else:
            report += f"- {f}\n"

    report += "\n\n---\n\n## EDA Figures\n\n"
    report += "- `eda_figures/01_score_distributions.png` — All score component histograms\n"
    report += "- `eda_figures/02_gpa_risk_distribution.png` — GPA histogram and risk pie chart\n"
    report += "- `eda_figures/03_correlation_heatmap.png` — Feature correlation heatmap\n"
    report += "- `eda_figures/04_programme_semester_dist.png` — Programme and semester distributions\n"
    report += "- `eda_figures/05_prerequisite_chains.png` — Prerequisite chain correlations\n"
    report += "\n\n---\n\n## Conclusion\n\n"

    if n_warn == 0:
        report += "All validation checks passed. The synthetic dataset is realistic and ready for model training.\n"
    elif n_warn <= 2:
        report += f"{n_pass}/{n_total} checks passed. The dataset is realistic with minor deviations, acceptable for model training.\n"
    else:
        report += f"WARNING: {n_warn} checks failed. Dataset may need regeneration.\n"

    return report


def main():
    print("=== Phase 3: EDA and Data Validation ===")
    df_raw, df_sum, ref = load_data()
    print(f"  Loaded {len(df_raw):,} course records, {len(df_sum):,} student summaries")

    print("  Running validation checks ...")
    findings = check_distributions(df_sum, ref)

    print("  Generating EDA figures ...")
    plot_distributions(df_raw, df_sum, ref)

    print("  Generating validation report ...")
    report = generate_report(findings, df_sum, df_raw)
    report_path = REPORTS / "data_validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved report -> {report_path}")

    print("\n  Validation Summary:")
    for f in findings:
        prefix = "  [PASS]" if f.startswith("PASS") else ("  [WARN]" if f.startswith("WARN") else "  ")
        safe = f.encode("ascii", errors="replace").decode("ascii")
        print(f"{prefix} {safe}")

    print("\n=== Phase 3 Complete ===")


if __name__ == "__main__":
    main()
