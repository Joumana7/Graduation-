#!/usr/bin/env python3
"""
phase1_scrape_website.py  —  Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Phase 1: Crawl the official Zewail City website and save raw page content.

The site is fully JavaScript-rendered, so this script uses Playwright
(Chromium, headless) with a single persistent browser session for speed.

Output : data/raw/web_raw.jsonl
Each record:
  url           – absolute page URL
  title         – page <title>
  raw_text      – visible text (boilerplate removed)
  scraped_at    – ISO-8601 UTC timestamp
  char_count    – len(raw_text)
  category_hint – coarse topic tag (courses/admissions/policy/…)

Usage:
  python phase1_scrape_website.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment

# ── Configuration ──────────────────────────────────────────────────────────────

DOMAIN        = "zewailcity.edu.eg"
MAX_PAGES     = 200          # hard ceiling on pages saved
PAGE_DELAY    = 0.8          # seconds between page loads (polite pacing)
NAV_TIMEOUT   = 20_000       # Playwright navigation timeout (ms)
MIN_CHARS     = 150          # discard pages with fewer content characters

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE  = OUTPUT_DIR / "web_raw.jsonl"

# ── Seed URLs — every major section ───────────────────────────────────────────
SEED_URLS: list[str] = [
    "https://www.zewailcity.edu.eg/",
    "https://www.zewailcity.edu.eg/about/",
    "https://www.zewailcity.edu.eg/academics/",
    "https://www.zewailcity.edu.eg/undergraduate-studies/",
    "https://www.zewailcity.edu.eg/graduate-studies/",
    "https://www.zewailcity.edu.eg/admissions/",
    "https://www.zewailcity.edu.eg/research/",
    "https://www.zewailcity.edu.eg/schools/",
    "https://www.zewailcity.edu.eg/programs/",
    "https://www.zewailcity.edu.eg/departments/",
    "https://www.zewailcity.edu.eg/faculty/",
    "https://www.zewailcity.edu.eg/campus/",
    "https://www.zewailcity.edu.eg/student-life/",
    "https://www.zewailcity.edu.eg/life-at-zewail/",
    "https://www.zewailcity.edu.eg/news/",
    "https://www.zewailcity.edu.eg/events/",
    "https://www.zewailcity.edu.eg/contact/",
    "https://www.zewailcity.edu.eg/faqs/",
    "https://www.zewailcity.edu.eg/scholarships/",
    "https://www.zewailcity.edu.eg/registration/",
    "https://www.zewailcity.edu.eg/calendar/",
    "https://www.zewailcity.edu.eg/student-affairs/",
    "https://www.zewailcity.edu.eg/facilities/",
    "https://www.zewailcity.edu.eg/library/",
    "https://www.zewailcity.edu.eg/internships/",
    "https://www.zewailcity.edu.eg/graduation/",
    "https://www.zewailcity.edu.eg/regulations/",
    "https://www.zewailcity.edu.eg/policies/",
    "https://www.zewailcity.edu.eg/school-of-science-and-engineering/",
    "https://www.zewailcity.edu.eg/school-of-business/",
    "https://www.zewailcity.edu.eg/csai/",
    "https://www.zewailcity.edu.eg/sci/",
    "https://www.zewailcity.edu.eg/bus/",
]

SKIP_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".bmp",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv", ".ogg", ".wav",
    ".zip", ".rar", ".tar", ".gz", ".7z",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".otf", ".css", ".js",
    ".xml", ".json", ".txt",
})

SKIP_PATTERNS: list[re.Pattern] = [re.compile(p, re.I) for p in [
    r"/wp-admin",
    r"/wp-login",
    r"/wp-content/uploads",
    r"/wp-json",
    r"/feed/?$",
    r"/tag/",
    r"/author/",
    r"/page/\d+/?$",
    r"\?p=\d+",
    r"[?&]lang=ar",
    r"/ar/",
    r"^#",
    r"^javascript:",
    r"^mailto:",
    r"^tel:",
    r"^whatsapp:",
]]

# ── Category tagging ───────────────────────────────────────────────────────────

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "courses": [
        "course", "curriculum", "credit hour", "prerequisite", "syllabus",
        "elective", "core requirement", "course catalog", "lecture", "lab course",
    ],
    "admissions": [
        "admission", "apply", "application", "eligibility", "enroll",
        "tuition", "fees", "scholarship", "financial aid", "sat", "interview",
    ],
    "policy": [
        "policy", "regulation", "academic rule", "code of conduct",
        "attendance", "grading", "probation", "dismissal",
        "graduation requirement", "withdrawal", "academic integrity", "plagiarism",
    ],
    "deadlines": [
        "deadline", "academic calendar", "registration period",
        "spring semester", "fall semester", "summer session",
        "add/drop", "exam schedule",
    ],
    "facilities": [
        "facility", "campus", "library", "laboratory", "sport", "clinic",
        "housing", "dormitory", "cafeteria", "shuttle", "building", "gym",
    ],
    "faculty": [
        "faculty", "professor", "instructor", "staff", "dr.", "phd",
        "department chair", "dean", "research interest", "office hours",
    ],
    "research": [
        "research", "publication", "project", "journal", "conference",
        "grant", "innovation", "nanotechnology", "biomedical", "energy",
    ],
}


def classify(url: str, text: str) -> str:
    u, t = url.lower(), text.lower()
    url_hints = {
        "admissions": ["admission", "apply", "enroll", "tuition", "scholarship"],
        "faculty":    ["faculty", "professor", "staff", "people", "team"],
        "research":   ["research", "publication", "innovation"],
        "courses":    ["course", "curriculum", "program", "degree",
                       "undergraduate", "graduate", "catalog"],
        "policy":     ["policy", "regulation", "rule", "conduct"],
        "deadlines":  ["calendar", "schedule", "deadline", "registration"],
        "facilities": ["campus", "facility", "life", "housing", "library"],
    }
    for cat, keys in url_hints.items():
        if any(k in u for k in keys):
            return cat
    scores: dict[str, int] = {c: 0 for c in CATEGORY_KEYWORDS}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            scores[cat] += t.count(kw)
    best = max(scores, key=lambda c: scores[c])
    return best if scores[best] > 0 else "general"


# ── URL helpers ────────────────────────────────────────────────────────────────

def normalise(url: str, base: str = "") -> Optional[str]:
    url = url.strip()
    if not url:
        return None
    if base:
        url = urljoin(base, url)
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return None
    if DOMAIN not in p.netloc.lower():
        return None
    path_low = p.path.lower().split("?")[0]
    if any(path_low.endswith(e) for e in SKIP_EXTENSIONS):
        return None
    for pat in SKIP_PATTERNS:
        if pat.search(url):
            return None
    canonical = f"{p.scheme}://{p.netloc}{p.path}"
    if p.query:
        canonical += f"?{p.query}"
    return canonical


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


# ── HTML → (title, clean_text, links) ─────────────────────────────────────────

NOISE_RE = re.compile(
    r"\b(menu|nav|navbar|breadcrumb|sidebar|widget|cookie|social|share|"
    r"comments?|ads?|advertisement|popup|modal|banner|ribbon|overlay|"
    r"topbar|back-to-top|scroll-to-top)\b",
    re.I,
)


def parse_html(html: str, base_url: str) -> tuple[str, str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # Title
    t = soup.find("title")
    title = t.get_text(strip=True) if t else base_url

    # Drop structural noise
    for tag in soup.find_all([
        "script", "style", "noscript", "iframe", "template",
        "nav", "header", "footer", "aside", "form",
        "button", "input", "select", "textarea", "label",
    ]):
        tag.decompose()
    for node in soup.find_all(string=lambda s: isinstance(s, Comment)):
        node.extract()
    for tag in soup.find_all(True):
        cls = " ".join(tag.get("class", []))
        id_ = tag.get("id", "")
        if NOISE_RE.search(cls) or NOISE_RE.search(id_):
            tag.decompose()

    # Collect outlinks
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        n = normalise(a["href"], base_url)
        if n:
            links.append(n)

    # Main content heuristic
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"\b(main|content|article|post|entry|body)\b", re.I)) or
        soup.find(class_=re.compile(r"\b(main|content|article|post|entry|body)\b", re.I)) or
        soup.find("body") or soup
    )

    raw = main.get_text(separator="\n")
    lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 4]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text, links


# ── Playwright crawler (single browser session) ────────────────────────────────

def crawl() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        print("ERROR: Playwright not installed.")
        print("  Run: pip install playwright && playwright install chromium")
        sys.exit(1)

    queue:   deque[str] = deque()
    visited: set[str]   = set()
    hashes:  set[str]   = set()
    records: list[dict] = []
    errors:  list[str]  = []

    for url in SEED_URLS:
        n = normalise(url)
        if n and n not in visited:
            visited.add(n)
            queue.append(n)

    print("Phase 1 - Zewail City Website Scraper")
    print("=" * 62)
    print(f"  Domain      : {DOMAIN}")
    print(f"  Seed URLs   : {len(queue)}")
    print(f"  Max pages   : {MAX_PAGES}")
    print(f"  Page delay  : {PAGE_DELAY}s")
    print(f"  Renderer    : Playwright / Chromium (headless)")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx     = browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = ctx.new_page()

        while queue and len(records) < MAX_PAGES:
            url = queue.popleft()
            idx = len(records) + 1

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)
                # Small wait to let React/Vue finish mounting
                page.wait_for_timeout(800)
                html = page.content()

                title, text, links = parse_html(html, url)

                if len(text) < MIN_CHARS:
                    print(f"  [{idx:>3}] SKIP  (sparse: {len(text)} chars)  {url}")
                    continue

                ch = content_hash(text)
                if ch in hashes:
                    print(f"  [{idx:>3}] SKIP  (duplicate content)  {url}")
                    continue
                hashes.add(ch)

                cat = classify(url, text)
                records.append({
                    "url":           url,
                    "title":         title,
                    "raw_text":      text,
                    "scraped_at":    datetime.now(timezone.utc).isoformat(),
                    "char_count":    len(text),
                    "category_hint": cat,
                })
                print(f"  [{idx:>3}] OK  [{cat:<12s}]  {len(text):>6,} chars  {url}")

                for link in links:
                    if link not in visited:
                        visited.add(link)
                        queue.append(link)

            except PWTimeout:
                errors.append(url)
                print(f"  [---] TIMEOUT  {url}")
            except Exception as exc:
                errors.append(url)
                print(f"  [---] {exc.__class__.__name__}: {str(exc)[:80]}  {url}")

            time.sleep(PAGE_DELAY)

        browser.close()

    # ── Write output ────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────────────
    total_chars = sum(r["char_count"] for r in records)
    cats: dict[str, int] = {}
    for r in records:
        cats[r["category_hint"]] = cats.get(r["category_hint"], 0) + 1

    print()
    print("=" * 62)
    print("  Phase 1 - SUMMARY")
    print("=" * 62)
    print(f"  Pages saved        : {len(records)}")
    print(f"  Total characters   : {total_chars:,}")
    print(f"  Unique URLs tried  : {len(visited)}")
    print(f"  Errors / timeouts  : {len(errors)}")
    print(f"  Output             : {OUTPUT_FILE}")
    print()
    print("  Category breakdown:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:<14s}  {count:>3}  {'|' * min(count, 40)}")
    print()
    if len(records) == 0:
        print("  WARNING: No pages saved. Check network access to zewailcity.edu.eg.")
        sys.exit(1)
    else:
        print(f"  Phase 1 complete. {len(records)} pages ready for Phase 2.")


if __name__ == "__main__":
    crawl()
