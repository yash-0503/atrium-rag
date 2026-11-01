# Measurements

Every number the website quotes is measured, and every measurement says
what it was measured on. This file is where the numbers come from; the
website fetches it and cannot quote anything that is not in it.

The measurements are produced by `scripts/measure_performance.py`, which
uses the production code paths — `atrium.pipeline.ingest_document` and
`atrium.retrieval.retrieve_and_rerank` — because a number measured
against a reimplementation of the product is a number about the
reimplementation.

```bash
python scripts/measure_performance.py \
    --corpus <the fixture corpus> \
    --limit 19 \
    --queries <questions.jsonl> \
    --out docs/measurements.json
```

## What Atrium costs

**Machine:** Apple M4, 16 GB · Python 3.12.13 · **PDF backend:**
pdfplumber (MIT) · **store:** SQLite (`atrium_local.db`) · **embedder:**
bge-large, 1024-d, over HTTP at `http://localhost:11434/v1`

| Measurement | Value |
|---|---|
| Documents ingested | 19 |
| Chunks written | 1,451 |
| Ingesting the corpus (parse + chunk + embed + store) | 96.3 s |
| Resident, serving search | 88 MB |
| Peak, during the ingest | 425 MB |
| Search p50 / p95 — retrieval only | 27 ms / 36 ms |
| Search p50 / p95 — including the query embedding | 68 ms / 77 ms |

Two search numbers, because they answer different questions. *Retrieval
only* is what the vector store and the ranking cost. *End to end* is
what a user waits for, and it includes the round trip to the embedding
endpoint — which is in Ollama's process, not ours.

**Resident** is measured in a process that never ingested anything: a
server that has just read a corpus holds that corpus in memory, and
that is not the number a deployed server runs at. The same process
after the ingest holds 425 MB, because the allocator does not give the
memory back. A worker that ingests continuously should be recycled
between batches.

**The embedder is a separate process.** bge-large costs about 670 MB
inside Ollama, and it is in none of the numbers above.

## The comparison

Against Docling and Unstructured, on 23 documents and 700 questions,
each system using its own parser and chunker and the same embedding
model:

| system | R@5 | 95% CI | parse | peak RSS |
|---|---|---|---|---|
| atrium | 0.700 | 0.594 – 0.799 | 88 s | 1016 MB |
| docling | 0.686 | 0.565 – 0.798 | 554 s | 3482 MB |
| unstructured | 0.614 | 0.497 – 0.725 | 195 s | 855 MB |

Retrieval accuracy is on par with Docling: the difference is inside the
noise band at this corpus size, and the report prints the interval
rather than a number without one. The separation is in cost.

These numbers come from the evaluation harness in the `atrium-benchmarks`
repository, and so does everything behind them: the corpus, the
questions, the scoring, and the commands to re-run it.

```bash
python -m eval_v2.compare --root eval_v2 --corpus-root . --split all \
    --only-scored --chunkers native \
    --retrieval hybrid,bm25_only,embeddings_only \
    --bootstrap 1000 --top-k 10 --systems ours \
    --out docs/benchmark/run_ours.json
```

## The chunk-packing diagram

The website's chunk diagram is a drawing of what the shipped chunker
does, so it comes from the shipped chunker:

```bash
python scripts/dump_chunk_packing.py \
    --document tests/parsing_fixtures/26_gov_docx/source.docx \
    --out ../atrium-web/src/lib/chunk-packings.json
```
