#!/usr/bin/env python3
"""
phase4_chunk_and_embed.py  —  Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Phase 4: Chunk cleaned documents and embed them into a ChromaDB vector store.

Input  : data/clean/cleaned_documents.jsonl  (Phase 3 output)
Output : db/chroma_db/                        (persistent ChromaDB collection)

Chunking strategy:
  - Paragraph-aware: split on double newlines first, then fall back to sliding
    window (chunk_size chars, chunk_overlap overlap) if a paragraph is too long.
  - Each chunk carries full metadata: doc_id, source, category, chunk_id, etc.

Embeddings:
  - Model : text-embedding-3-small  (OpenAI)
  - Batch : up to 100 chunks per API call
  - Key   : read from .env (OPENAI_API_KEY)

Usage:
  python phase4_chunk_and_embed.py
  python phase4_chunk_and_embed.py --force   # wipe and rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent
CLEAN_JSONL  = PROJECT_ROOT / "data" / "clean" / "cleaned_documents.jsonl"
CHROMA_DIR   = PROJECT_ROOT / "db" / "chroma_db"

COLLECTION_NAME = "zewail_campus"
EMBED_MODEL     = "text-embedding-3-small"
EMBED_BATCH     = 100       # chunks per embedding API call
CHUNK_SIZE      = 800       # target characters per chunk
CHUNK_OVERLAP   = 150       # overlap between consecutive chunks
MIN_CHUNK_LEN   = 60        # discard chunks shorter than this


# ── Custom chunker ─────────────────────────────────────────────────────────────

def sliding_window(text: str, size: int, overlap: int) -> Iterator[str]:
    """Character-level sliding window."""
    start = 0
    while start < len(text):
        end   = start + size
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = end - overlap


def chunk_document(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    1. Split on blank lines (paragraph boundaries).
    2. If a paragraph fits within `size` chars, keep it whole.
    3. If it's too long, apply sliding window on that paragraph.
    4. Merge very short consecutive chunks into the next one.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    raw_chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= size:
            raw_chunks.append(para)
        else:
            raw_chunks.extend(sliding_window(para, size, overlap))

    # Merge tiny trailing chunks into their predecessor
    merged: list[str] = []
    buf = ""
    for ch in raw_chunks:
        if buf:
            candidate = buf + "\n\n" + ch
            if len(candidate) <= size + overlap:
                buf = candidate
                continue
            else:
                merged.append(buf)
                buf = ch
        else:
            buf = ch
    if buf:
        merged.append(buf)

    return [c for c in merged if len(c) >= MIN_CHUNK_LEN]


# ── Embedding helper ───────────────────────────────────────────────────────────

def embed_batch(texts: list[str], client) -> list[list[float]]:
    """Call OpenAI Embeddings API for a batch of texts."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ── Main ───────────────────────────────────────────────────────────────────────

def run(force: bool = False) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        return

    print("Phase 4 - Chunk and Embed")
    print("=" * 62)
    print(f"  Collection  : {COLLECTION_NAME}")
    print(f"  Embed model : {EMBED_MODEL}")
    print(f"  Chunk size  : {CHUNK_SIZE} chars / {CHUNK_OVERLAP} overlap")

    if not CLEAN_JSONL.exists():
        print(f"  ERROR: {CLEAN_JSONL} not found. Run Phase 3 first.")
        return

    # ── Load documents ──────────────────────────────────────────────────────────
    docs: list[dict] = []
    with open(CLEAN_JSONL, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  Loaded {len(docs)} documents")

    # ── Chunk ───────────────────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_document(doc["text"])
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id":    f"{doc['doc_id']}_c{i:04d}",
                "text":        chunk_text,
                "doc_id":      doc["doc_id"],
                "source_type": doc["source_type"],
                "source":      doc["source"],
                "page":        doc.get("page", ""),
                "category":    doc["category"],
            })
    print(f"  Chunks created : {len(all_chunks)}")

    # ── Wipe and/or connect ChromaDB ────────────────────────────────────────────
    if force and CHROMA_DIR.exists():
        print("  Wiping existing vector store ...")
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    import chromadb
    client_db = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Get or create collection (wipe if force)
    try:
        col = client_db.get_collection(COLLECTION_NAME)
        existing = col.count()
        if existing > 0 and not force:
            print(f"  Collection already has {existing} vectors. Use --force to rebuild.")
            return
        client_db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client_db.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Embed and insert in batches ─────────────────────────────────────────────
    from openai import OpenAI
    oai = OpenAI(api_key=api_key)

    done   = 0
    errors = 0
    batch_size = EMBED_BATCH

    for batch_start in range(0, len(all_chunks), batch_size):
        batch  = all_chunks[batch_start: batch_start + batch_size]
        texts  = [c["text"] for c in batch]
        ids    = [c["chunk_id"] for c in batch]
        metas  = [{k: v for k, v in c.items() if k not in ("text", "chunk_id")}
                  for c in batch]

        try:
            embeddings = embed_batch(texts, oai)
            col.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metas,
            )
            done += len(batch)
            print(f"  Embedded {done}/{len(all_chunks)} chunks ...", end="\r")
            time.sleep(0.3)  # gentle rate limiting
        except Exception as exc:
            print(f"\n  ERROR embedding batch: {exc}")
            errors += len(batch)

    # ── Summary ─────────────────────────────────────────────────────────────────
    final_count = col.count()
    embed_dim   = len(oai.embeddings.create(model=EMBED_MODEL, input=["test"]).data[0].embedding)

    print()
    print()
    print("=" * 62)
    print("  Phase 4 - SUMMARY")
    print("=" * 62)
    print(f"  Documents processed   : {len(docs)}")
    print(f"  Chunks created        : {len(all_chunks)}")
    print(f"  Vectors stored        : {final_count}")
    print(f"  Embedding dimension   : {embed_dim}")
    print(f"  Embedding errors      : {errors}")
    print(f"  Collection            : {COLLECTION_NAME}")
    print(f"  Persist path          : {CHROMA_DIR}")
    print()
    print(f"  Phase 4 complete. Vector store ready for Phase 5.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Wipe and rebuild from scratch")
    args = ap.parse_args()
    run(force=args.force)
