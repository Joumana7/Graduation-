# Zewail City Campus Digital Assistant

> **RAG + Conversational Memory academic advisor for Zewail City University of Science and Technology**

---

## Overview

An end-to-end intelligent campus assistant that answers student queries about:
- University regulations and academic policies
- Degree requirements, course prerequisites, and study plans
- Graduation requirements and academic planning
- Admissions, scholarships, and registration
- Campus facilities, research opportunities, and student life

**Core technology:** Retrieval-Augmented Generation (RAG) + Conversational Memory
**Data source:** Live Zewail City website scraped fresh by the pipeline
**LLM:** GPT-4o via OpenAI API
**Vector store:** ChromaDB (local, persistent)
**UI:** Streamlit

---

## Architecture

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
|  Phase 7: Streamlit app -- chat UI + citations + analytics |
+-------------------------------------------------------------+
```

---

## Project Structure

```
zewail_campus_assistant/
|-- phase1_scrape_website.py        <- Playwright web crawler
|-- phase2_extract_pdfs.py          <- PDF discovery + download + extraction
|-- phase3_clean_data.py            <- Text cleaning + categorisation
|-- phase4_chunk_and_embed.py       <- Chunking + OpenAI embeddings -> ChromaDB
|-- phase5_rag_pipeline.py          <- CampusRAG class (retrieve + generate)
|-- phase6_conversational_memory.py <- ConversationalAssistant with memory
|-- phase7_streamlit_app.py         <- Streamlit UI
|
|-- data/
|   |-- raw/
|   |   |-- web_raw.jsonl           <- Phase 1 output (71 pages)
|   |   |-- pdf_raw.jsonl           <- Phase 2 output (0 PDFs - site uses auth)
|   |   +-- pdfs/                   <- Downloaded PDF files
|   |-- clean/
|   |   +-- cleaned_documents.jsonl <- Phase 3 output (68 docs)
|   +-- sessions/                   <- Phase 6 session JSON files
|
|-- db/
|   +-- chroma_db/                  <- Phase 4 ChromaDB vector store (387 chunks)
|
|-- tests/
|   |-- test_phase1_scrape.py
|   |-- test_phase2_pdfs.py
|   |-- test_phase3_clean.py
|   |-- test_phase4_embed.py
|   |-- test_phase5_rag.py
|   +-- test_phase6_memory.py
|
|-- .env.example                    <- Copy to .env and add API key
|-- requirements.txt
+-- README.md
```

---

## Setup

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
playwright install chromium    # Download browser for Phase 1 + 2
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

---

## Running the Pipeline

Run each phase in order. Each prints a summary and exits.

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

### Start the Streamlit app

```bash
streamlit run phase7_streamlit_app.py
# -> http://localhost:8501
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

- **Phases 1-4 tests**: File/structure validation -- no API key required.
- **Phases 5-6 tests**: Make real OpenAI API calls -- requires `OPENAI_API_KEY`.
- **Phase 2 tests**: 5 schema tests are auto-skipped when `pdf_raw.jsonl` is empty (the Zewail City website serves PDFs via authenticated portals -- this is expected).

Expected result: **64 passed, 5 skipped**.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Web scraper | Playwright (headless Chromium) | zewailcity.edu.eg is fully JS-rendered |
| PDF extraction | PyMuPDF (fitz) | Fast, accurate, no Java dependency |
| Chunking | Paragraph-aware + sliding window | Preserves semantic units |
| Embeddings | text-embedding-3-small | Cost-effective, strong quality |
| Vector store | ChromaDB (local) | No server needed, fast, persistent |
| Generation | GPT-4o | Best quality for factual advising |
| Memory | In-session dict + JSON file | Simple, transparent, survives reloads |
| Architecture | Single-process (phases -> app) | No microservices overhead needed |

---

## Memory Design

**Short-term memory:** The last 10 user+assistant turns are injected into every GPT-4o call as conversation history, enabling natural follow-up questions without repeating context.

**Long-term/user memory:** Student profile information (program, semester, GPA, completed courses, failed courses) is extracted from each message using regex patterns and stored in the session object. This profile is prepended to every generation call so answers are personalised from the first mention onward.

**Persistence:** Every session is saved as `data/sessions/<session_id>.json` and survives page reloads. The session ID is stored in Streamlit's `st.session_state` so the same session is recovered automatically.

---

## Example Interaction

```
Student: I'm in Semester 5 of CSAI. My GPA is 2.9. I failed Signals and Electronics.
         I completed Calculus I & II, Physics, Programming I & II. What should I take?
Assistant (with profile context):
  Based on your profile — CSAI, Semester 5, GPA 2.9, failed Signals and Electronics —
  I recommend focusing on courses that do not have Signals as a prerequisite and that
  can raise your GPA. Consider lighter electives this semester to improve your standing
  before attempting heavier core courses. Please confirm with your academic advisor for
  the official approved plan.

Student: Do I need to retake Signals before taking Digital Systems?

Assistant:
  I don't have the specific prerequisite chain for Digital Systems in my knowledge base.
  Please check the official course catalog at https://www.zewailcity.edu.eg/course-catalog
  or contact the Registrar for the current prerequisite list.
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

---

## License

This project is for academic/educational purposes only.
