"""Diagnose retrieval for general school/program overview queries."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

f = open("D:/FINAL_PROJECT_v3/general_test.txt", "w", encoding="utf-8")

from phase5_rag_pipeline import CampusRAG
rag = CampusRAG()

queries = [
    "what are the majors offered in zewail city",
    "how many schools are in zewail city",
    "what are the programs of SCI school",
    "schools programs undergraduate zewail city",
]

for q in queries:
    chunks, note = rag.retrieve(q, top_k=6)
    f.write(f"\n=== Q: {q} ===\n")
    for c in chunks:
        f.write(f"  [{c.score:.3f}] {c.source} p.{c.page} [{c.category}]\n")
        f.write(f"  >> {c.text[:300]}\n\n")
    f.close()
    f = open("D:/FINAL_PROJECT_v3/general_test.txt", "a", encoding="utf-8")

# Also check what web pages are in the knowledge base
import json
from pathlib import Path
web_jsonl = Path("data/raw/web_raw.jsonl")
f.write("\n\n=== WEB PAGES SCRAPED ===\n")
if web_jsonl.exists():
    with open(web_jsonl, encoding="utf-8") as wf:
        for line in wf:
            rec = json.loads(line)
            url = rec.get("url","")
            f.write(f"  {url}\n")
else:
    f.write("  web_raw.jsonl not found!\n")

f.close()
print("Done")
