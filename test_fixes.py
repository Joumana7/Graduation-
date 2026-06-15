"""Quick validation of the two new retrieval fixes."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

f = open("D:/FINAL_PROJECT_v3/fix_test.txt", "w", encoding="utf-8")

from phase5_rag_pipeline import CampusRAG
rag = CampusRAG()

# Test 1 – course code anchor search
f.write("=== Test: where_document $contains ===\n")
import chromadb
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
emb = oai.embeddings.create(model="text-embedding-3-small", input=["CSAI 201"]).data[0].embedding

client = chromadb.PersistentClient(path="db/chroma_db")
col = client.get_collection("zewail_campus")

try:
    r = col.query(
        query_embeddings=[emb],
        n_results=5,
        where_document={"$contains": "CSAI 201"},
        include=["documents", "distances"],
    )
    f.write(f"where_document OK — {len(r['documents'][0])} docs\n")
    for d, dist in zip(r["documents"][0], r["distances"][0]):
        f.write(f"  [{1-dist:.3f}] {d[:220]}\n")
except Exception as e:
    f.write(f"where_document FAILED: {e}\n")

f.write("\n=== Test: CSAI 201 full answer ===\n")
ans, chunks = rag.answer("what is CSAI 201 course name?")
f.write(f"Sources:\n")
for c in chunks[:4]:
    f.write(f"  [{c.score:.3f}] {c.source} p.{c.page} | {c.text[:150]}\n")
f.write(f"Answer: {ans}\n")

f.write("\n=== Test: MATH 105 course name ===\n")
ans_m, chunks_m = rag.answer("what is MATH 105 course name?")
f.write(f"Answer: {ans_m[:250]}\n")

f.write("\n=== Test: Dr el Reabey ===\n")
ans2, chunks2 = rag.answer("who is Dr el Reabey?")
f.write(f"Sources:\n")
for c in chunks2[:3]:
    f.write(f"  [{c.score:.3f}] {c.source} | {c.text[:120]}\n")
f.write(f"Answer: {ans2}\n")

f.write("\n=== Test: Dr Hadhoud ===\n")
ans3, chunks3 = rag.answer("who is Dr Hadhoud?")
f.write(f"Answer: {ans3[:300]}\n")

f.close()
print("Done")
