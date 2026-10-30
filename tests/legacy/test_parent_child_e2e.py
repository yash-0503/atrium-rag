"""
test_parent_child_e2e.py

QUARANTINED — see README.md in this directory.

Single-fixture end-to-end validation for the parent-child retrieval path
(Task B, 2026-10-10).

This script targets `chunk_with_config` and `chunk_with_parent_child`,
neither of which was ever defined in any commit (`git log -S "def
chunk_with_config" --all` → zero hits). They belonged to the same
uncommitted token-based generation as the tests beside this one, so the
script cannot run against the shipped chunker.

Goal: prove that context_rich (parent-child chunking) actually returns
useful parent context for queries that target specific content in W-2.
Run against a live Qdrant + Ollama embedder.

Steps:
  1. Reset the 'documents' collection.
  2. Run precise (flat) mode on W-2; verify R@5, sample 3 results.
  3. Run context_rich (parent-child) mode on W-2; verify R@5,
     sample 3 results, confirm parent_text is populated and meaningful.
  4. Spot-check: for each top hit, does parent_text contain the
     expected evidence (e.g., "Copy A—For Social Security Administration"
     for the SSA query)?

Run: .venv-bench/bin/python3 tests/legacy/test_parent_child_e2e.py
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from atrium.adapters import OpenAICompatibleEmbedder, QdrantAdapter
from atrium.smart_chunker import PRESETS, chunk_with_config, chunk_with_parent_child
from atrium.universal_parser import UniversalParser


def get_embedder():
    import os
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/").replace("/v1", "")
    return OpenAICompatibleEmbedder(
        base_url=f"{base}/v1",
        model=os.getenv("EMBED_MODEL", "bge-large-en-v1.5"),
        dim=int(os.getenv("EMBED_DIM", "1024")),
    )


def main():
    fixture = REPO / "tests/benchmarks/fixtures/17_irs_w2/source.pdf"
    if not fixture.exists():
        print(f"FAIL: W-2 fixture not found at {fixture}")
        return 1

    print(f"Loading {fixture.name}...")
    file_bytes = fixture.read_bytes()
    parser = UniversalParser()
    elements = parser.parse(file_bytes, fixture.name, "application/pdf")
    print(f"  parsed {len(elements)} elements")

    embedder = get_embedder()
    print(f"  embedder: {embedder.model} (dim={embedder.dimension})")
    if not embedder.health():
        print("FAIL: embedder unhealthy")
        return 1

    db = QdrantAdapter(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
    db.ensure_collection("documents", embedder.dimension)

    queries = [
        "Which copy of Form W-2 is filed with the Social Security Administration?",
        "What does box 1 on Form W-2 report?",
        "How many years should an employee keep Copy C of Form W-2?",
    ]
    # Pre-embed the queries once (independent of chunking).
    query_vecs = embedder.embed(queries)

    # ----- MODE A: precise (flat) -----
    print("\n--- MODE A: precise (flat) ---")
    flat_chunks = chunk_with_config(
        elements, PRESETS["precise"],
        doc_id="w2-precise", file_name=fixture.name,
    )
    print(f"  flat chunks: {len(flat_chunks)}")
    flat_texts = [c.embed_text or c.text for c in flat_chunks]
    flat_vecs = embedder.embed(flat_texts)
    # Wipe + upsert
    from qdrant_client import models as qmodels
    from qdrant_client.models import PointStruct
    db.client.delete(
        collection_name="documents",
        points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=[])),
    )
    points = []
    for _i, (c, v) in enumerate(zip(flat_chunks, flat_vecs)):
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"w2-precise:{c.chunk_id}"))
        points.append(PointStruct(id=pid, vector=v, payload={
            "text": c.text,
            "doc_id": "w2-precise",
            "chunk_id": c.chunk_id,
            "page_number": c.source.page_number,
            "section_path": list(c.source.section_path),
            "element_type": c.element_type,
            "is_parent": False,
            "parent_id": None,
            "schema_version": 0,
        }))
    db.client.upsert(collection_name="documents", points=points)
    print(f"  upserted {len(points)} flat points")

    flat_hits = []
    for q, qv in zip(queries, query_vecs):
        hits = db.search(qv, top_k=5)
        flat_hits.append(hits)
        first = hits[0] if hits else None
        print(f"  Q: {q[:60]}...")
        print(f"     top score={first['score']:.3f} page={first.get('page_number')} text={first['text'][:80]!r}")

    # ----- MODE B: context_rich (parent-child) -----
    print("\n--- MODE B: context_rich (parent-child) ---")
    pc_chunks = chunk_with_parent_child(
        elements, doc_id="w2-context-rich", file_name=fixture.name,
        config=PRESETS["context_rich"],
    )
    n_parents = sum(1 for c in pc_chunks if c.child_ids)
    n_children = sum(1 for c in pc_chunks if c.parent_id)
    print(f"  parent-child chunks: {len(pc_chunks)} ({n_parents} parents, {n_children} children)")
    pc_texts = [c.embed_text or c.text for c in pc_chunks]
    pc_vecs = embedder.embed(pc_texts)
    parent_lookup = {}
    for c in pc_chunks:
        if c.child_ids:
            parent_lookup[c.chunk_id] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"w2-context-rich:{c.chunk_id}"))
    points = []
    for c, v in zip(pc_chunks, pc_vecs):
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"w2-context-rich:{c.chunk_id}"))
        parent_point_id = parent_lookup.get(c.parent_id)
        points.append(PointStruct(id=pid, vector=v, payload={
            "text": c.text,
            "doc_id": "w2-context-rich",
            "chunk_id": c.chunk_id,
            "page_number": c.source.page_number,
            "section_path": list(c.source.section_path),
            "element_type": c.element_type,
            "is_parent": bool(c.child_ids),
            "parent_id": parent_point_id,
            "schema_version": 1,
        }))
    db.client.delete(
        collection_name="documents",
        points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=[])),
    )
    db.client.upsert(collection_name="documents", points=points)
    print(f"  upserted {len(points)} parent-child points")

    from atrium.retrieval import retrieve_with_parent_resolution
    pc_hits = []
    # Debug: manually verify the parent-child round-trip
    print("  DEBUG: schema_version check")
    sv = db.collection_schema_version()
    print(f"    collection_schema_version = {sv}")
    print("  DEBUG: search_children first hit parent_id")
    sample_children = db.search_children(query_vecs[0], top_k=3)
    for sc in sample_children:
        print(f"    child_id={sc.get('id')!r} parent_id={sc.get('parent_id')!r}")
    if sample_children and sample_children[0].get("parent_id"):
        sample_pid = sample_children[0]["parent_id"]
        print(f"  DEBUG: retrieve_parents with id={sample_pid!r}")
        rm = db.retrieve_parents([sample_pid])
        print(f"    retrieved {len(rm)} parent(s): keys={list(rm.keys())}")
    for q, qv in zip(queries, query_vecs):
        hits = retrieve_with_parent_resolution(db, qv, top_k=5)
        pc_hits.append(hits)
        first = hits[0] if hits else None
        print(f"  Q: {q[:60]}...")
        print(f"     top score={first['score']:.3f} page={first.get('page_number')}")
        print(f"     child text={first['text'][:80]!r}")
        pt = first.get('parent_text')
        print(f"     parent_text={(pt or '<None>')[:120]!r}")

    # ----- EVIDENCE SPOT CHECK -----
    print("\n--- EVIDENCE SPOT CHECK ---")
    expected_phrases = {
        0: ["Social Security Administration", "Copy A"],
        1: ["Wages", "tips", "compensation", "Box 1"],
        2: ["3 years", "Keep Copy C"],
    }
    for i, (_q, hits) in enumerate(zip(queries, pc_hits)):
        if not hits:
            print(f"  Q{i}: NO HITS")
            continue
        # Check parent_text for expected evidence.
        parent_texts = [h.get("parent_text") or "" for h in hits]
        all_parent = "\n".join(parent_texts).lower()
        phrases = expected_phrases.get(i, [])
        matches = [p for p in phrases if p.lower() in all_parent]
        status = "PASS" if matches else "FAIL"
        print(f"  Q{i}: {status}  matched phrases={matches}")
        print(f"        evidence text (truncated): {(hits[0].get('parent_text') or '')[:200]!r}")

    # Cleanup
    db.client.delete(
        collection_name="documents",
        points_selector=qmodels.FilterSelector(filter=qmodels.Filter(must=[])),
    )
    return 0


import os

if __name__ == "__main__":
    sys.exit(main())