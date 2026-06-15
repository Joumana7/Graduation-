# Zewail City Campus Digital Assistant

> **Dual-Product Intelligent Academic Advisor for Zewail City University of Science and Technology**
>
> Combining Retrieval-Augmented Generation (RAG) with Explainable AI (XAI) and Curriculum Intelligence into one unified platform.

---

## Overview

The Zewail City Campus Digital Assistant is an end-to-end AI system built across two complementary products that work together under a single Streamlit interface:

| Product | What it does |
|---------|-------------|
| **Product A — RAG Academic Advisor** | Answers student questions about policies, regulations, degree plans, prerequisites, and campus life by retrieving from live Zewail City content via GPT-4o |
| **Product B — Learning Analytics & XAI Advisor** | Predicts student GPA and academic risk, explains predictions with SHAP, maps the student's position against the official Zewail curriculum, and generates curriculum-aware recommendations and what-if scenarios |

Both products share the same Streamlit app and student profile, but are architecturally independent — Product A can run without Product B's models, and vice versa.

---

## System Architecture

### Product A — RAG + Conversational Memory (Phases 1–7)

```
+-------------------------------------------------------------+
|                  DATA ACQUISITION LAYER                     |
|  Phase 1: Playwright -> zewailcity.edu.eg -> web_raw.jsonl  |
|  Phase 2: Playwright -> PDF links -> pdf_raw.jsonl          |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  PROCESSING LAYER                           |
|  Phase 3: Clean + deduplicate -> cleaned_documents.jsonl   |
|  Phase 4: Chunk + OpenAI Embed -> ChromaDB (zewail_campus) |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  INTELLIGENCE LAYER                         |
|  Phase 5: CampusRAG  -- retrieve() + generate() (GPT-4o)  |
|  Phase 6: ConversationalAssistant -- short & long-term mem |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  PRESENTATION LAYER                         |
|  Phase 7: Streamlit app -- chat UI + citations + memory    |
+-------------------------------------------------------------+
```

### Product B — Learning Analytics & Explainable AI (XAI Module)

```
+-------------------------------------------------------------+
|                  DATA & CURRICULUM LAYER                    |
|  Synthetic student dataset (5,000 records)                  |
|  course_catalog.json        -- all Zewail courses           |
|  degree_requirements.json   -- credit requirements per prog |
|  prerequisites_graph.json   -- full prerequisite DAG        |
|  academic_regulations.json  -- probation, GPA rules         |
|  gpa_rules.json             -- grade-to-GPA conversion      |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  FEATURE ENGINEERING LAYER                  |
|  Phase 4: 21 academic features (attendance, scores, etc.)  |
|  Phase 4b: +8 curriculum-aware features                     |
|    graduation_progress_ratio  core_completion_ratio         |
|    prereq_completion_proxy    blocked_progress_ratio        |
|    curriculum_alignment_proxy curriculum_readiness_score    |
|    graduation_delay_semesters expected_progress_ratio       |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  PREDICTION LAYER (XGBoost)                 |
|  GPA Regression Model   -- predicts cumulative GPA (0-4)   |
|  Risk Classification    -- Low / Medium / High risk         |
|  StandardScaler         -- fitted on 21 original features   |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  EXPLAINABILITY LAYER (SHAP + XAI)         |
|  SHAP TreeExplainer     -- per-feature GPA contribution     |
|  Curriculum SHAP        -- rule-based curriculum impact     |
|    "Graduation Pace", "Core Completion", "Prereq Integrity" |
|  Waterfall charts, global summary, PDP, PCA clusters        |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  CURRICULUM INTELLIGENCE LAYER              |
|  CurriculumEngine       -- singleton, loads JSONs once      |
|    match_courses()      -- fuzzy code/name -> catalogue     |
|    get_graduation_status()  -- credits, delay, readiness    |
|    get_blocked_courses()    -- failed -> blocked chain      |
|    simulate_course_scenario()  -- pass/fail/retake impact   |
|    get_curriculum_shap_values()  -- GPA-impact per factor   |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  RECOMMENDATION & SIMULATION LAYER          |
|  RecommendationEngine   -- curriculum-first priority order  |
|    Curriculum Advisories  (prerequisite gaps, grad risk)    |
|    Academic Performance   (SHAP-driven score advice)        |
|  WhatIfEngine           -- slider + curriculum scenarios    |
|    Academic sliders     -- attendance, midterm, finals      |
|    Smart Scenarios      -- auto retake/fail/pass scenarios  |
+---------------------+---------------------------------------+
                      |
+---------------------v---------------------------------------+
|                  PRESENTATION LAYER (Streamlit — 5 tabs)   |
|  Tab 1: AI Analysis Chat    -- conversational profile intake|
|  Tab 2: Academic Dashboard  -- GPA, risk, curriculum charts |
|  Tab 3: Recommendations     -- grouped curriculum + academic|
|  Tab 4: What-If Simulator   -- sliders + smart scenarios    |
|  Tab 5: Advanced Analytics  -- SHAP, curriculum XAI, PCA   |
+-------------------------------------------------------------+
```

---

## Project Structure

```
zewail_campus_assistant/
|
|-- phase1_scrape_website.py            <- Product A: Playwright web crawler
|-- phase2_extract_pdfs.py              <- Product A: PDF discovery + extraction
|-- phase3_clean_data.py                <- Product A: Text cleaning + categorisation
|-- phase4_chunk_and_embed.py           <- Product A: Chunking + embeddings -> ChromaDB
|-- phase5_rag_pipeline.py              <- Product A: CampusRAG class (retrieve + generate)
|-- phase6_conversational_memory.py     <- Product A: ConversationalAssistant with memory
|-- phase7_streamlit_app.py             <- Product A: Streamlit UI (RAG chat)
|
|-- learning_analytics_xai/             <- Product B: XAI Academic Advisor
|   |
|   |-- data/                           <- Curriculum knowledge base (JSON)
|   |   |-- course_catalog.json         <- All Zewail courses with credits + prerequisites
|   |   |-- degree_requirements.json    <- Per-programme credit requirements
|   |   |-- prerequisites_graph.json    <- Full prerequisite dependency graph
|   |   |-- academic_regulations.json   <- Probation, GPA thresholds, policies
|   |   |-- gpa_rules.json              <- Grade-to-GPA conversion table
|   |   |-- synthetic_students.csv      <- 5,000 synthetic student records
|   |   |-- students_summary.csv        <- Per-student aggregate summary
|   |   +-- feature_names.json          <- 21 ML feature column names
|   |
|   |-- models/                         <- Trained XGBoost models (pre-built)
|   |   |-- gpa_model_xgb.pkl           <- GPA regression model
|   |   |-- risk_model_xgb.pkl          <- Risk classification model
|   |   |-- scaler.pkl                  <- StandardScaler fitted on 21 features
|   |   +-- model_metrics.json          <- R², MAE, RMSE, CV scores
|   |
|   |-- curriculum_intelligence/
|   |   +-- curriculum_engine.py        <- CurriculumEngine singleton (core logic)
|   |
|   |-- feature_engineering/
|   |   +-- feature_engineer.py         <- 21 + 8 curriculum features, build_features()
|   |
|   |-- xai/
|   |   +-- explainability.py           <- SHAP waterfall, global summary, PDP, PCA
|   |
|   |-- recommendation_engine/
|   |   +-- recommender.py              <- Priority-ordered curriculum + academic recs
|   |
|   |-- what_if_analysis/
|   |   +-- what_if_engine.py           <- Slider simulation + curriculum preset scenarios
|   |
|   +-- dashboard/
|       +-- analytics_page.py           <- Full 5-tab Streamlit dashboard (Product B UI)
|
|-- data/
|   |-- raw/
|   |   |-- web_raw.jsonl               <- Phase 1 output (71 pages)
|   |   |-- pdf_raw.jsonl               <- Phase 2 output
|   |   +-- pdfs/                       <- Downloaded PDF files
|   |-- clean/
|   |   +-- cleaned_documents.jsonl     <- Phase 3 output (68 docs)
|   +-- sessions/                       <- Phase 6 session JSON files
|
|-- db/
|   +-- chroma_db/                      <- ChromaDB vector store (387 chunks)
|
|-- tests/
|   |-- test_phase1_scrape.py
|   |-- test_phase2_pdfs.py
|   |-- test_phase3_clean.py
|   |-- test_phase4_embed.py
|   |-- test_phase5_rag.py
|   +-- test_phase6_memory.py
|
|-- .env.example                        <- Copy to .env and add API key
|-- requirements.txt
+-- README.md
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
playwright install chromium    # For Phase 1 + 2 scraping
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

### Product A — RAG Pipeline (run in order)

```bash
# Phase 1 -- Scrape the Zewail City website (~5-10 min, JS-rendered)
python phase1_scrape_website.py

# Phase 2 -- Discover and download PDF handbooks
python phase2_extract_pdfs.py

# Phase 3 -- Clean and categorise documents
python phase3_clean_data.py

# Phase 4 -- Chunk and embed into ChromaDB (requires OPENAI_API_KEY)
python phase4_chunk_and_embed.py

# Phase 5 -- Verify RAG pipeline with 5 sample queries
python phase5_rag_pipeline.py

# Phase 6 -- Run 3-turn memory demo
python phase6_conversational_memory.py
```

### Product B — XAI Models (pre-trained, no retraining needed)

The XGBoost models in `learning_analytics_xai/models/` are pre-trained and ready to use.
The curriculum JSON files in `learning_analytics_xai/data/` are the primary knowledge source.
No additional pipeline steps are required for Product B.

### Start the Unified Streamlit App

```bash
# Product A — RAG Academic Advisor chat
streamlit run phase7_streamlit_app.py
# -> http://localhost:8501

# Product B — Learning Analytics & XAI Advisor (5-tab dashboard)
streamlit run learning_analytics_xai/dashboard/analytics_page.py
# -> http://localhost:8502
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

- **Phases 1-4 tests**: File/structure validation — no API key required.
- **Phases 5-6 tests**: Make real OpenAI API calls — requires `OPENAI_API_KEY`.
- **Phase 2 tests**: 5 schema tests are auto-skipped when `pdf_raw.jsonl` is empty (the Zewail City website serves PDFs via authenticated portals — this is expected).

Expected result: **64 passed, 5 skipped**.

---

## Key Design Decisions

### Product A — RAG

| Decision | Choice | Reason |
|----------|--------|--------|
| Web scraper | Playwright (headless Chromium) | zewailcity.edu.eg is fully JS-rendered |
| PDF extraction | PyMuPDF (fitz) | Fast, accurate, no Java dependency |
| Chunking | Paragraph-aware + sliding window | Preserves semantic units |
| Embeddings | text-embedding-3-small | Cost-effective, strong quality |
| Vector store | ChromaDB (local) | No server needed, fast, persistent |
| Generation | GPT-4o | Best quality for factual advising |
| Memory | In-session dict + JSON file | Simple, transparent, survives reloads |

### Product B — Learning Analytics & XAI

| Decision | Choice | Reason |
|----------|--------|--------|
| Prediction model | XGBoost (regression + classification) | High accuracy on tabular academic data, native SHAP support |
| Explainability | SHAP TreeExplainer | Exact Shapley values for tree models, not approximations |
| Curriculum SHAP | Rule-based GPA-impact scores | Models trained on 21 features; curriculum layer runs in parallel without retraining |
| Curriculum knowledge | JSON files (catalog, requirements, prereq graph) | Official PDFs are the ground truth; JSON enables fast in-memory graph traversal |
| Prerequisite graph | Directed Acyclic Graph (DAG) | Enables `get_all_dependents()` to propagate blockage through the full chain |
| Recommendation order | Curriculum critical-first, then SHAP-driven | Programme obligations take precedence over score optimisation |
| What-If scenarios | Slider (live recompute) + Smart Scenarios (auto-generated) | Covers both ad-hoc exploration and curated curriculum-relevant scenarios |
| Feature count | 21 original (model) + 8 curriculum-approximate (display) | Keeps existing trained models intact while adding curriculum intelligence |

---

## Curriculum Intelligence Design

The `CurriculumEngine` is the core of Product B's academic awareness. It loads the official Zewail curriculum JSON files once at startup (singleton pattern) and exposes:

- **Course matching** — fuzzy lookup by course code or name across all programmes
- **Graduation status** — actual credits vs. required, delay in semesters, graduation readiness score
- **Prerequisite blocking** — given a set of failed course codes, returns the full transitive closure of blocked courses, blocked credit hours, and chain depth
- **Scenario simulation** — simulates passing, failing, retaking, or postponing any course and returns the projected impact on graduation timeline and blocked/unblocked courses
- **Curriculum SHAP values** — returns per-factor GPA-impact scores (e.g., "Graduation Pace: −0.20 GPA") grounded in official degree regulations, displayed as a waterfall chart alongside the standard SHAP waterfall

This layer is purely rule-based and additive — it does not modify the trained ML models, ensuring predictions remain stable while adding curriculum-aware context on top.

---

## Memory Design (Product A)

**Short-term memory:** The last 10 user+assistant turns are injected into every GPT-4o call as conversation history, enabling natural follow-up questions without repeating context.

**Long-term/user memory:** Student profile information (program, semester, GPA, completed courses, failed courses) is extracted from each message using regex patterns and stored in the session object. This profile is prepended to every generation call so answers are personalised from the first mention onward.

**Persistence:** Every session is saved as `data/sessions/<session_id>.json` and survives page reloads. The session ID is stored in Streamlit's `st.session_state` so the same session is recovered automatically.

---

## Example Interactions

### Product A — RAG Chat

```
Student: I'm in Semester 5 of CSAI. My GPA is 2.9. I failed Signals and Electronics.
         I completed Calculus I & II, Physics, Programming I & II. What should I take?

Assistant (with profile context):
  Based on your profile — CSAI, Semester 5, GPA 2.9, failed Signals and Electronics —
  I recommend focusing on courses that do not have Signals as a prerequisite and that
  can raise your GPA. Consider lighter electives this semester to improve your standing
  before attempting heavier core courses. Please confirm with your academic advisor.

Student: Do I need to retake Signals before taking Digital Systems?

Assistant:
  Yes — Signals and Electronics (CSAI203) is a prerequisite for Digital Systems (CSAI301).
  Failing it blocks CSAI301 and 3 other downstream courses (9 blocked credits total).
  Retaking it next semester is the highest-priority action for your degree path.
```

### Product B — XAI Dashboard

```
Input (via AI Analysis Chat):
  Name: Sara  Programme: CSAI  Semester 5  Attendance 68%  Midterm 61
  Failed: CSAI203 (Signals)  Passed: CSAI101, CSAI102, MATH101, PHYS101

Dashboard shows:
  Predicted GPA: 2.41  (Medium Risk)
  Graduation delay: 1.2 semesters behind expected pace
  Blocked by CSAI203 failure: 9 credits across 3 downstream courses

SHAP waterfall (academic factors):
  avg_overall      -0.42   <- largest negative driver
  avg_attendance   -0.18
  failed_courses   -0.11

Curriculum XAI waterfall (curriculum factors):
  Graduation Readiness   -0.28   <- 1.2 semesters behind schedule
  Failed Prerequisite    -0.16   <- CSAI203 blocks 3 courses
  Graduation Pace        -0.10

Top Recommendation (Curriculum — shown first):
  [CRITICAL] Retake CSAI203 (Signals and Electronics)
  Failing this course blocks CSAI301, CSAI310, and CSAI405 —
  9 credits and 3 core courses you cannot register until it is passed.
  This is your single highest-impact academic action this semester.

What-If Smart Scenario: "Retake CSAI203"
  Graduation delay: 0 semesters  Courses unblocked: CSAI301, CSAI310, CSAI405
```

---

## Known Limitations

- **No PDF handbooks scraped**: The Zewail City website serves programme handbooks and
  policies through authenticated or dynamically loaded links not accessible by a public
  crawler. Answers about specific credit-hour counts, exact prerequisite chains, and
  probation thresholds rely on the handbook PDFs — without them the system falls back
  to directing users to official contacts.
- **Website snapshot**: Web content was scraped at pipeline run time. Re-run Phase 1
  and Phase 4 whenever the website is updated.
- **OpenAI dependency**: Phases 4, 5, 6, 7, and all integration tests require a valid
  `OPENAI_API_KEY`. Costs are minimal (< $0.10 for a full pipeline run).
- **XGBoost/SHAP models are pre-trained**: The models were trained on 21 features and
  cannot be retrained in the current environment without installing xgboost and shap.
  Curriculum intelligence runs as a parallel rule-based layer and does not require retraining.

---

## License

This project is for academic/educational purposes only.