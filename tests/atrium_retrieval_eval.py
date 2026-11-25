"""Run eval_v2's 700 questions through ATRIUM'S OWN retrieval code.

eval_v2 ships its own retriever (`eval_v2/retrieval/pipeline.py`). That
means its R@5 measures what goes INTO the index, not how Atrium searches
it — and every retrieval bug (the `top_k` one, the two RRF
implementations, the score semantics) is invisible to it.

This harness closes that gap. It reuses eval_v2's corpus, questions,
canonical text and scoring verbatim, and swaps ONLY the retriever for
one that calls `atrium.retrieval.retrieve_and_rerank` — the function
`/search` calls in production.

Everything below is a scratch script. It modifies neither repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# `import atrium` must resolve to atrium-rag, and `import eval_v2` to
# rag-ingest. They cannot both be first on sys.path: rag-ingest contains
# a real `atrium/` package directory (a three-week-old copy), and a
# directory on sys.path beats a package one level up. So atrium-rag goes
# first, and rag-ingest is imported by its own package path rather than
# by being on sys.path.
sys.path.insert(0, "/Users/yashsolanki/Desktop/rag-ingest")
sys.path.insert(0, "/Users/yashsolanki/Desktop/atrium-rag")

EV = Path("/Users/yashsolanki/Desktop/rag-ingest/eval_v2")
ROOT = EV.parent

from eval_v2.adapters.base import get_adapter
from eval_v2.chunking import ChunkOptions
from eval_v2.common import build_embedder
from eval_v2.core.align import merge_intervals
from eval_v2.core.aligning import align_blocks
from eval_v2.core.types import Passage
from eval_v2.corpus.manifest import read_manifest
from eval_v2.questions.generate import read_questions
from eval_v2.runner import Index, _embed_in_batches, _resolve_path


class CachingEmbedder:
    """Wraps an embedder and remembers what it has embedded.

    The evaluation asks the same 700 questions of two retrievers, and
    every question needs one embedding. Without a cache that is 1400
    round trips to the embedding server, one at a time, and the run
    takes most of its wall-clock in HTTP.
    """

    def __init__(self, inner):
        self.inner = inner
        self._cache = {}

    def embed(self, texts):
        missing = [t for t in texts if t not in self._cache]
        if missing:
            for text, vector in zip(missing, self.inner.embed(missing)):
                self._cache[text] = vector
        return [self._cache[t] for t in texts]


class AtriumRetriever:
    """A retriever that calls `atrium.retrieval.retrieve_and_rerank`.

    It satisfies the same interface the harness's own Retriever does —
    `search(query, top_k) -> list[Passage]` — so eval_v2's scoring code
    is used unchanged. The only difference is where the ranking comes
    from.
    """

    def __init__(self, chunks, embedder, config, vectors=None, db=None, reranker=None):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.config = config
        self._vectors = list(vectors or [])
        self.db = db
        self.reranker = reranker
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._query_count = 0
        self._latencies = []

    def search(self, query: str, top_k=None):
        from atrium.retrieval import retrieve_and_rerank

        top_k = top_k or self.config.top_k
        started = time.perf_counter()
        query_vector = self.embedder.embed([query])[0]

        bm25 = None
        if self.config.mode in ("bm25", "hybrid") and self.config.bm25_index is not None:
            bm25 = self.config.bm25_index

        hits = retrieve_and_rerank(
            self.db,
            query_vector,
            query,
            top_k,
            reranker=self.reranker,
            candidate_k=self.config.candidate_k,
            bm25_index=bm25,
            bm25_candidate_k=self.config.candidate_k,
        )

        passages = []
        for rank, hit in enumerate(hits[:top_k], start=1):
            chunk = self._by_id.get(hit.get("chunk_id") or hit.get("id"))
            if chunk is None:
                continue
            passages.append(Passage(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=hit.get("text", ""),
                score=float(hit.get("score", 0.0)),
                rank=rank,
                page=hit.get("page_number"),
                section_path=list(hit.get("section_path") or []),
                canon_intervals=list(getattr(chunk, "canon_intervals", []) or []),
            ))
        self._query_count += 1
        self._latencies.append(time.perf_counter() - started)
        return passages


def build_atrium_index(doc_ids, embedder, chunker="native", mode="hybrid",
                       top_k=10, candidate_k=50, use_reranker=False):
    """The corpus, chunked and embedded, indexed in a real SqliteAdapter."""
    import tempfile

    from atrium.local_store import SqliteAdapter
    from atrium.models import ChunkRecord, SourceMetadata

    specs = {s.doc_id: s for s in read_manifest(EV / "corpus_manifest.csv")}
    adapter = get_adapter("ours")

    index = Index(system="ours", chunker=chunker)
    db = SqliteAdapter(os.path.join(tempfile.mkdtemp(), "atrium.db"))
    records = []

    for doc_id in doc_ids:
        spec = specs.get(doc_id)
        if spec is None:
            continue
        canonical_path = EV / "corpus/reference" / f"{doc_id}.txt"
        if not canonical_path.exists():
            index.failed[doc_id] = "no canonical text"
            continue
        from eval_v2.corpus.build_fixtures import _canonical_doc
        canonical = _canonical_doc(doc_id, canonical_path.read_text(encoding="utf-8"))
        index.canonical_by_doc[doc_id] = canonical

        path = _resolve_path(EV, ROOT, spec.filename)
        if not path.exists():
            index.failed[doc_id] = f"missing document: {path}"
            continue

        started = time.perf_counter()
        try:
            blocks = adapter.ingest(path, doc_id)
        except Exception as exc:
            index.failed[doc_id] = f"{type(exc).__name__}: {exc}"
            continue
        index.parse_seconds += time.perf_counter() - started
        index.blocks_by_doc[doc_id] = blocks
        index.block_counts[doc_id] = len(blocks)
        index.adapter_text_by_doc[doc_id] = "\n\n".join(b.text for b in blocks)

        # Our own chunker, through the same adapter the harness uses.
        started = time.perf_counter()
        chunks = adapter.chunk(blocks, ChunkOptions(strategy=chunker), doc_id=doc_id)
        index.chunk_seconds += time.perf_counter() - started

        alignment = align_blocks(blocks, canonical)
        for chunk in chunks:
            chunk.doc_id = doc_id
            intervals = []
            for bi in chunk.block_indices:
                if 0 <= bi < len(alignment.intervals):
                    intervals.extend(alignment.intervals[bi])
            chunk.canon_intervals = merge_intervals(intervals)
            index.chunks.append(chunk)

            # Write the chunk into the real store, exactly as the
            # production ingest path would.
            records.append(ChunkRecord(
                doc_id=doc_id,
                chunk_id=chunk.chunk_id,
                ingest_run_id="bench",
                text=chunk.text,
                embed_text=chunk.embed_text,
                embedding=[0.0],       # filled below
                element_type=chunk.element_type,
                source=SourceMetadata(
                    file_name="document.pdf",
                    page_number=chunk.page,
                    section_path=list(chunk.section_path or []),
                    page_numbers=[chunk.page] if chunk.page else [],
                ),
                char_span=(0, len(chunk.text or "")),
                content_hash=chunk.chunk_id,
                parent_id=getattr(chunk, "parent_id", None),
                is_parent=getattr(chunk, "is_parent", False),
                child_ids=getattr(chunk, "child_ids", []),
                schema_version=getattr(chunk, "schema_version", 0),
                chunking_strategy=getattr(chunk, "chunking_strategy", ""),
            ))

    # Embed with the harness's embedder, exactly as build_index does.
    started = time.perf_counter()
    vectors = _embed_in_batches(
        [(c.embed_text if c.embed_text else c.text) for c in index.chunks],
        embedder,
    )
    index.embed_seconds = time.perf_counter() - started

    dim = len(vectors[0]) if vectors else 768
    db.ensure_collection("documents", dim=dim)
    for rec, vec in zip(records, vectors):
        rec.embedding = list(vec)
    db.upsert(records)

    # A BM25 index over the same chunks, for the hybrid mode.
    bm25 = None
    from atrium.bm25 import BM25Index
    bm25 = BM25Index()
    for c in index.chunks:
        bm25.add_doc(c.chunk_id, c.text or "", meta={
            "id": c.chunk_id, "chunk_id": c.chunk_id, "doc_id": c.doc_id,
            "text": c.text or "", "page_number": getattr(c, "page", None),
            "section_path": list(c.section_path or []),
            "element_type": c.element_type,
        })

    reranker = None
    if use_reranker:
        from atrium.rerank_adapter import build_reranker
        reranker = build_reranker()
        if reranker is None:
            # A silently-missing reranker turns a reranked run into a
            # dense-only run that looks identical. Say so and stop.
            raise SystemExit(
                "--rerank was asked for but no reranker could be built. "
                "Install the backend in THIS python's environment "
                f"({sys.executable}) and set RERANK_BACKEND / RERANK_ENABLED."
            )
        print(f"  reranker: {type(reranker).__name__}", flush=True)

    config = type("C", (), {})()
    config.mode = mode
    config.top_k = top_k
    config.candidate_k = candidate_k
    config.rerank = use_reranker
    config.parent_expansion = False
    config.heading_prefix = True
    config.with_confidence = True
    config.min_score = 0.0
    config.bm25_index = bm25

    index.retriever = AtriumRetriever(index.chunks, embedder, config,
                                      vectors=vectors, db=db, reranker=reranker)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "heldout", "all"])
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "bm25_only", "embeddings_only"])
    ap.add_argument("--chunker", default="native")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/tmp/atrium_audit/atrium_retrieval.json")
    ap.add_argument("--compare", action="store_true",
                    help="also score the harness's own retriever on the same chunks")
    args = ap.parse_args()

    splits = json.loads((EV / "splits.json").read_text())
    doc_ids = splits["dev"] if args.split == "dev" else (
        splits["heldout"] if args.split == "heldout" else splits["dev"] + splits["heldout"])
    if args.limit:
        doc_ids = doc_ids[: args.limit]

    questions = read_questions(EV / "questions.jsonl")
    embedder = CachingEmbedder(build_embedder("ollama"))

    print(f"building the index with ATRIUM's own pipeline "
          f"({len(doc_ids)} docs, split={args.split})", flush=True)
    t0 = time.time()
    index = build_atrium_index(doc_ids, embedder, chunker=args.chunker,
                               mode=args.mode, use_reranker=args.rerank)
    print(f"  {len(index.chunks)} chunks, {len(index.failed)} failed, "
          f"{time.time()-t0:.0f}s", flush=True)

    from eval_v2.runner import evaluate
    records, metrics = evaluate(index, questions, mode=args.mode, top_k=10,
                                candidate_k=50)

    # Where does the retrieval actually lose? A single mean hides the
    # document that is costing the points, and one bad document is a
    # different fix from a broad ranking problem.
    if records:
        by_doc = {}
        for r in records:
            d = by_doc.setdefault(r.get("doc_id", "?"), {"n": 0, "labels": Counter()})
            d["n"] += 1
            d["labels"][r.get("failure_label", "?")] += 1
        print()
        print("=" * 92)
        print("PER DOCUMENT — where the losses are")
        print("=" * 92)
        print(f"  {'document':30} {'n':>4} {'R@5':>7} {'R@10':>7}  breakdown")
        rows = []
        for doc_id, d in sorted(by_doc.items()):
            rs = [r for r in records if r.get("doc_id") == doc_id
                  and r.get("is_answerable")]
            r5 = sum(1 for r in rs if r.get("recall_at_5")) / max(1, len(rs))
            r10 = sum(1 for r in rs if r.get("recall_at_10")) / max(1, len(rs))
            rows.append((d["n"] - d["labels"].get("NO_ANSWER_ERROR", 0), doc_id, d, r5, r10))
        for _loss, doc_id, d, r5, r10 in sorted(rows, reverse=True):
            labels = {k: v for k, v in d["labels"].items()
                      if k not in ("NO_ANSWER_ERROR",)}
            print(f"  {doc_id:30} {d['n']:4} {r5:7.2f} {r10:7.2f}  {labels}")

    if args.compare:
        # The harness's own retriever, on the SAME chunks and the SAME
        # questions. The difference between the two numbers is the
        # retrieval code and nothing else.
        from eval_v2.retrieval.pipeline import RetrievalConfig, Retriever
        from eval_v2.runner import _embed_in_batches
        cfg = RetrievalConfig(top_k=10, candidate_k=50, mode=args.mode)
        vectors = _embed_in_batches(
            [(c.embed_text if c.embed_text else c.text) for c in index.chunks],
            embedder)
        index.retriever = Retriever(index.chunks, embedder, cfg, vectors=vectors)
        _, hmetrics = evaluate(index, questions, mode=args.mode, top_k=10,
                               candidate_k=50)
        print()
        print("=" * 70)
        print("SIDE BY SIDE — same chunks, same questions, same embedder")
        print("=" * 70)
        print(f"  {'metric':16} {'harness':>12} {'atrium':>12} {'delta':>10}")
        for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
            if k in metrics and k in hmetrics:
                d = metrics[k] - hmetrics[k]
                print(f"  {k:16} {hmetrics[k]:12.4f} {metrics[k]:12.4f} {d:+10.4f}")
        print(f"  {'harness labels':16} {hmetrics.get('failure_counts')}")
        print(f"  {'atrium  labels':16} {metrics.get('failure_counts')}")
    print()
    print("=" * 70)
    print(f"ATRIUM'S OWN RETRIEVER — split={args.split} mode={args.mode} "
          f"chunker={args.chunker} rerank={args.rerank}")
    print("=" * 70)
    for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
        if k in metrics:
            print(f"  {k:16} {metrics[k]:.4f}")
    print(f"  failure labels  {metrics.get('failure_counts')}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"metrics": metrics, "n_records": len(records),
         "split": args.split, "mode": args.mode, "rerank": args.rerank},
        indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
