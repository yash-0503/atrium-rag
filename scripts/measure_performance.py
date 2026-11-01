#!/usr/bin/env python3
"""
measure_performance.py

Measure what the product costs, on the build that ships.

    python scripts/measure_performance.py \
        --corpus ~/atrium-benchmarks/tests/benchmarks/fixtures \
        --limit 19 \
        --queries ~/atrium-benchmarks/eval_v2/questions.jsonl \
        --out docs/measurements.json

Four numbers come out:

  ingest seconds   parse + chunk + embed + store, over the corpus
  peak MB          the highest RSS the process reached during the ingest
  resident MB      the RSS it settles at afterwards, serving search
  search p50/p95   the latency of the retrieval path the API uses

The script uses the production code paths — `atrium.pipeline.ingest_document`
and `atrium.retrieval.retrieve_and_rerank` — because a number measured
against a reimplementation of the product is a number about the
reimplementation.

Everything is measured in one process, so the embedder's own memory is
*not* in these numbers: Ollama (or whatever endpoint you point at) runs
in its own process. The footnote on the website says so, and this is
where that claim comes from.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def rss_mb() -> float:
    """The process's current resident set size, in MB."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip()) / 1024.0
    except Exception:
        return 0.0


def peak_mb() -> float:
    """The highest RSS the process reached, in MB."""
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def detect_mime(path: Path) -> str:
    import mimetypes

    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def collect_documents(corpus: Path, limit: int) -> list[tuple[str, Path]]:
    """The documents to ingest: one source file per fixture directory."""
    found: list[tuple[str, Path]] = []
    for entry in sorted(corpus.iterdir()):
        if not entry.is_dir():
            continue
        sources = [
            p
            for p in sorted(entry.iterdir())
            if p.is_file() and p.name.startswith("source")
        ]
        if sources:
            found.append((entry.name, sources[0]))
        if limit and len(found) >= limit:
            break
    return found


def load_queries(path: Optional[Path], limit: int) -> list[str]:
    """The questions to time the search with."""
    if path is None:
        return [
            "What is the retention policy?",
            "How do I request an extension?",
            "What are the payment terms?",
            "Who is responsible for approvals?",
            "What happens if a deadline is missed?",
            "How is the data classified?",
            "What is the escalation path?",
            "Which systems are in scope?",
            "What are the review intervals?",
            "How do exceptions get approved?",
        ][:limit]
    queries: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            text = record.get("query") or record.get("question") or ""
            if text:
                queries.append(str(text))
            if limit and len(queries) >= limit:
                break
    return queries




def _serve_in_a_fresh_process(args, store_path: str) -> Optional[dict]:
    """Run the serving phase in a new process and read its answer."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase", "serve",
        "--store", store_path,
        "--query-limit", str(args.query_limit),
        "--top-k", str(args.top_k),
    ]
    if args.queries:
        command += ["--queries", args.queries]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
    except Exception as exc:
        print(f"  the serving phase did not run: {exc}")
        return None
    text = proc.stdout.strip()
    if not text.startswith("{"):
        print(f"  the serving phase printed nothing usable: {proc.stderr[-200:]}")
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _percentile(values: list[float], pct: int) -> float:
    """A percentile of a list of samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[index]


def _time_searches(args, embedder, db, queries) -> tuple[list[float], list[float]]:
    """Time the retrieval path, and the whole request.

    Two numbers, because they answer different questions: the retrieval
    is what the vector store costs, and the end-to-end number is what a
    user waits for — which includes the round trip to the embedding
    endpoint.
    """
    from atrium.retrieval import retrieve_and_rerank

    retrieval_ms: list[float] = []
    total_ms: list[float] = []
    for query in queries:
        request_started = time.perf_counter()
        try:
            vector = embedder.embed([query])[0]
            embedded = time.perf_counter()
            retrieve_and_rerank(db, vector, query, args.top_k)
        except Exception as exc:
            print(f"  search failed: {type(exc).__name__}: {exc}")
            continue
        finished = time.perf_counter()
        retrieval_ms.append((finished - embedded) * 1000.0)
        total_ms.append((finished - request_started) * 1000.0)
    return retrieval_ms, total_ms


def _serve_phase(args, embedder) -> int:
    """Measure a process that only serves.

    A server that has just ingested a corpus holds that corpus in
    memory; a server that started and was handed the store does not.
    The number on the website is the second one, so it is measured
    here, in a process that never ingested anything.
    """
    from atrium.local_store import SqliteAdapter

    store_path = Path(args.store or "/tmp/atrium_measure.db")
    if not store_path.exists():
        print(f"no store at {store_path}", file=sys.stderr)
        return 1
    db = SqliteAdapter(path=str(store_path))
    queries = load_queries(
        Path(args.queries).expanduser() if args.queries else None, args.query_limit
    )
    retrieval_ms, total_ms = _time_searches(args, embedder, db, queries)
    measurement = {
        "resident_mb": round(rss_mb()),
        "search_p50_ms": round(_percentile(retrieval_ms, 50)),
        "search_p95_ms": round(_percentile(retrieval_ms, 95)),
        "search_end_to_end_p50_ms": round(_percentile(total_ms, 50)),
        "search_end_to_end_p95_ms": round(_percentile(total_ms, 95)),
        "queries": len(retrieval_ms),
    }
    print(json.dumps(measurement, indent=2))
    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure what Atrium costs.")
    parser.add_argument("--corpus", default=None, help="the fixture directory")
    parser.add_argument("--limit", type=int, default=0, help="how many documents")
    parser.add_argument("--queries", default=None, help="a JSONL file of questions")
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--out", default=None, help="write the result as JSON")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--phase",
        choices=["all", "ingest", "serve"],
        default="all",
        help="serve measures a fresh process against the store the ingest "
             "phase wrote, which is what a deployed server looks like",
    )
    parser.add_argument("--store", default=None, help="the store the serve phase reads")
    args = parser.parse_args(argv)

    from atrium.config import env_str
    from atrium.embedder_config import build_embedder
    from atrium.local_store import SqliteAdapter
    from atrium.pipeline import ingest_document
    from atrium.universal_parser import UniversalParser

    embedder = build_embedder()

    if args.phase == "serve":
        return _serve_phase(args, embedder)

    corpus = Path(args.corpus).expanduser() if args.corpus else None
    documents = collect_documents(corpus, args.limit) if corpus else []
    if not documents:
        print(f"no documents under {corpus}", file=sys.stderr)
        return 1

    parser_ = UniversalParser()

    # A scratch store, so the measurement is not polluted by whatever
    # corpus is already on this machine.
    scratch = Path(args.store or "/tmp/atrium_measure.db")
    if scratch.exists():
        scratch.unlink()
    db = SqliteAdapter(path=str(scratch))
    db.ensure_collection("documents", embedder.dimension)

    print(f"measuring {len(documents)} documents from {corpus}")
    print(f"  embedder: {type(embedder).__name__} dim={embedder.dimension}")
    print(f"  backend:  {env_str('EMBED_BASE_URL')}")
    print()

    started = time.perf_counter()
    chunks_written = 0
    for doc_id, path in documents:
        doc_started = time.perf_counter()
        try:
            result = ingest_document(
                doc_id=doc_id,
                file_bytes=path.read_bytes(),
                filename=path.name,
                mime_type=detect_mime(path),
                parser=parser_,
                embedder=embedder,
                db=db,
            )
            chunks_written += result.chunks_written
            print(
                f"  {doc_id:<32} {result.chunks_written:>5} chunks  "
                f"{time.perf_counter() - doc_started:>7.2f}s"
            )
        except Exception as exc:
            print(f"  {doc_id:<32} FAILED {type(exc).__name__}: {exc}")
    ingest_seconds = time.perf_counter() - started
    peak = peak_mb()
    resident = rss_mb()

    print()
    print(f"  ingest {ingest_seconds:.1f}s · peak {peak:.0f} MB · resident {resident:.0f} MB")

    # -- search latency ---------------------------------------------------
    queries = load_queries(
        Path(args.queries).expanduser() if args.queries else None, args.query_limit
    )
    retrieval_ms, total_ms = _time_searches(args, embedder, db, queries)
    print(
        f"  retrieval p50 {_percentile(retrieval_ms, 50):.0f} ms · "
        f"p95 {_percentile(retrieval_ms, 95):.0f} ms"
    )
    print(
        f"  end to end p50 {_percentile(total_ms, 50):.0f} ms · "
        f"p95 {_percentile(total_ms, 95):.0f} ms"
    )

    measurement = {
        "documents": len(documents),
        "chunks": chunks_written,
        "ingest_seconds": round(ingest_seconds, 1),
        "peak_mb": round(peak),
        "resident_after_ingest_mb": round(resident),
        "search_p50_ms": round(_percentile(retrieval_ms, 50)),
        "search_p95_ms": round(_percentile(retrieval_ms, 95)),
        "search_end_to_end_p50_ms": round(_percentile(total_ms, 50)),
        "search_end_to_end_p95_ms": round(_percentile(total_ms, 95)),
        "queries": len(retrieval_ms),
        "embedder": type(embedder).__name__,
        "embedder_dim": embedder.dimension,
        "backend": env_str("EMBED_BASE_URL"),
        "store": "sqlite",
        "machine": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip(),
        "python": sys.version.split()[0],
    }

    # The serving numbers come from a process that never ingested
    # anything: a server that has just read a corpus holds it in memory,
    # and that is not the number a deployed server runs at.
    serving = _serve_in_a_fresh_process(args, str(scratch))
    if serving:
        measurement["resident_mb"] = serving.get("resident_mb")
        measurement["serving_search_p50_ms"] = serving.get("search_p50_ms")
        measurement["serving_search_p95_ms"] = serving.get("search_p95_ms")
        measurement["serving_end_to_end_p50_ms"] = serving.get(
            "search_end_to_end_p50_ms"
        )
        measurement["serving_end_to_end_p95_ms"] = serving.get(
            "search_end_to_end_p95_ms"
        )
        print(
            f"  serving (fresh process) resident {serving.get('resident_mb')} MB · "
            f"retrieval p50 {serving.get('search_p50_ms')} ms · "
            f"end to end p50 {serving.get('search_end_to_end_p50_ms')} ms"
        )

    print()
    print(json.dumps(measurement, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(measurement, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")

    try:
        scratch.unlink()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
