"""Test general overview queries after adding overview documents."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

f = open("D:/FINAL_PROJECT_v3/overview_test.txt", "w", encoding="utf-8")

from phase5_rag_pipeline import CampusRAG
rag = CampusRAG()

queries = [
    "what are the majors offered in zewail city?",
    "what are the majors of science school in zewail city?",
    "how many schools are in zewail city?",
    "what programs does CSAI offer?",
    "what majors can I study in BUS?",
    "what is the difference between CSAI tracks?",
]

for q in queries:
    ans, chunks = rag.answer(q)
    top_src = chunks[0].source if chunks else "none"
    top_score = chunks[0].score if chunks else 0
    f.write(f"\nQ: {q}\n")
    f.write(f"Top source: [{top_score:.3f}] {top_src}\n")
    f.write(f"A: {ans[:400]}\n")

f.close()
print("Done")
