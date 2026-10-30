# Atrium — Comprehensive Audit Report

**Audit date:** 2026-10-30
**Package under audit:** `atrium/` (24 modules, 11,443 LOC)
**Repository root:** `/Users/yashsolanki/Desktop/rag-ingest`
**Declared version:** `0.1.0` (PyPI distribution name `atrium-rag`)
**Auditor stance:** principal Python engineer / RAG architect / OSS maintainer

---

## 0. Status update — 2026-10-30, after the audit

The audit is left unedited below so the reasoning reads as it was written. This section records
which findings have since been acted on.

**Finding 1 (licence) — resolved.** The PDF backend is now a seam (`atrium/pdf_backend.py`), and
the default is **pdfplumber** (MIT) over pdfminer.six, with pypdfium2 for rendering. PyMuPDF is
no longer a runtime dependency; it is an opt-in extra (`atrium-rag[fast-pdf]`) selected with
`ATRIUM_PDF_BACKEND=pymupdf`, for deployments that want its speed and are willing to take on
AGPL §13. Every remaining runtime dependency is permissive. See `README.md` § License and the
`fast-pdf` extra in `pyproject.toml`.

**Finding 2 (mislabeled headline) — open.** The published `docs/benchmark/v0_k8.json` still
contains `recall_at_5` where the README claims `R@8 = 0.8976`. The number has not been corrected
or re-published; this remains a release blocker.

---

## 1. Executive summary

Atrium is a **real, working, competently-built** piece of software. The packaging machinery is
genuinely good: the project builds a valid wheel and sdist, `twine check` passes on both, the wheel
installs into a clean virtualenv with zero dependency conflicts, and the installed artifact passes
a 10-point smoke test including a real PDF parse. The module boundaries are clean, the config
system is the best-designed part of the codebase, and the RAG code shows real thought — Reciprocal
Rank Fusion with a pinned smoothing constant, a confidence gate that deliberately refuses to
assume confidence from a single candidate, parent-child resolution with a schema-version contract,
a tri-state rerank override, and per-chunk provenance stamping.

It is also **not ready for public release**, and the gap between "the code is good" and "the
project can be published" is where nearly all the risk lives.

### The five things that matter most

1. **A licensing conflict that blocks distribution.** The package is MIT (`LICENSE:1`) but hard-depends
   on `pymupdf>=1.24.0`, whose own distribution metadata reads
   `License: Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License` (verified in the
   installed `pymupdf-2.0.dist-info/METADATA`). The project's own `atrium/pdf_backend.py:25` says
   `PyMuPDF AGPL-3.0`. An MIT-licensed project that ships AGPL-3.0 code to PyPI is a problem that
   needs a lawyer, not a README footnote. *This must be resolved before the package is published.*

2. **The README's headline accuracy number is mislabeled.** The README badge claims
   *"19/19 fixtures, mean R@8 = 0.8976"*. The published artifact `docs/benchmark/v0_k8.json`
   contains **no `recall_at_8` field at all** — only `recall_at_5`, `mean_recall_at_5`,
   `ndcg_at_5`, `mean_ndcg_at_5`. Recomputing from that file, the macro-average of
   `mean_recall_at_5` is **exactly 0.8976**. The number is a **Recall@5** presented as an **R@8**.
   Retrieval depth was 8 (`n_retrieved` maxes at 8); the metric depth was 5. The headline is wrong.

3. **The benchmark cannot be reproduced by anyone but the author.** The fixture corpus
   (~49 MB of third-party documents) is gitignored (`.gitignore:128-130`), the `eval_v2/` harness is
   excluded from the sdist, and the README's own Development section admits *"That corpus is not in
   the repository"*. An external developer cannot verify 0.8976 or produce a new number.

4. **The headline number overstates retrieval quality by construction.** Measured at
   **`metric_level: "page"`** for all 100 queries — a "hit" means *a chunk from the correct page*,
   not the correct passage. Six of nineteen fixtures have ≤6 chunks and score a perfect `1.0000`
   (the XLSX fixture has **2 chunks**). Those six contribute a perfect score to the macro-average;
   the other thirteen average **0.8504**. `MRR = 0.7058` and `nDCG@5 = 0.7381` are in the artifact
   but are not reported in the README.

5. **A production-critical performance defect.** `/search` calls `build_reranker(force=True)` on
   **every request** (`atrium/main.py:414`), and `LocalBgeReranker` loads its model **per instance**
   (`atrium/rerank_adapter.py:148-164`). With reranking enabled, the cross-encoder is reloaded from
   disk on every single query.

### Also confirmed, less visible

- Hybrid BM25 retrieval is **unreachable from the public API**. `bm25_index` is only constructed in
  the retrieval-lab sandbox (`atrium/main.py:1063-1067`). The production `/search` path never
  passes it — so the confidence-gate knobs `DENSE_CONFIDENT_GAP` and `DENSE_MIN_SCORE` are
  **no-ops in production**, despite `atrium/__init__.py:8` advertising "hybrid (vector + BM25) fusion".
- The `alt-parsers` extra installs `docling-slim` (~2 GB per the README) but **there is no Docling
  backend anywhere in the package** (`grep -rn docling atrium/*.py` → zero hits).
- `ATRIUM_PDF_BACKEND=pdfplumber` is selectable but `pdfplumber` is neither a declared dependency
  nor mentioned in the README.
- `ensure_collection(name, dim)` honors `name` **only for creation** — every other method hardcodes
  `"documents"` (`atrium/adapters.py:198, 206, 253, 315`).
- `collection_schema_version()` runs a **full-collection count scan on every retrieval call**, and
  the retrieval algorithm silently flips between v0 and v1 based on the *data* in the corpus.

### Verdict

**GO WITH LIMITATIONS** for a *private / invited* release (the code is good enough to hand to
trusted users), and **NO-GO for a public PyPI + GitHub launch today** until the four P0 items
(licensing, headline metric, benchmark reproducibility, and the per-request model reload) are
resolved. The gap is not "the software is bad" — it is "the claims, the packaging, and the
evidence do not yet match the quality of the code."

---

## 2. Scope, method, and what was actually executed

Everything in this report is derived from reading the source or from a command that was run in this
session. No number is estimated, extrapolated, or recalled from the README.

### 2.1 What was executed

| Action | Result |
|---|---|
| `python -m build` (wheel + sdist) | **Succeeded** → `atrium_rag-0.1.0-py3-none-any.whl` (160,518 B), `atrium_rag-0.1.0.tar.gz` (211,181 B) |
| `twine check` on both artifacts | **PASSED** (both) |
| Wheel install into a fresh venv (CPython 3.12.13) | **Succeeded**, 22 direct + transitive deps resolved |
| `pip check` in the clean venv | **No broken requirements found** |
| `scripts/smoke_test.py` against the *installed wheel* (run from `/tmp`) | **10/10 PASS** — import, console script, public submodules, parser, chunker, real PDF parse (313 elements from `17_irs_w2/source.pdf`), 15 HTTP routes, config precedence, atomic job claim, OCR correctly skipped with an install hint |
| sdist install into a clean venv | **Succeeded**; smoke test re-run against it → **all checks passed** |
| `pytest tests/ -q` | **225 passed, 7 skipped, 5 xfailed** in 31.6 s |
| Skip analysis | 6 × fixture-not-present (corpus gitignored), 1 × live-stack not reachable |
| Wheel metadata inspection | `Metadata-Version: 2.4`, `License-Expression: MIT`, `py3-none-any`, `py.typed` present, entry point `atrium = atrium.cli:app` |
| sdist contents listing | 24 package modules + `py.typed` + `README.md` + `LICENSE` + `MANIFEST.in` + `pyproject.toml` + `Dockerfile` + `docker-compose.yml` + `Makefile` + `.env.example` + `.env.docker` + **17 test files** |
| `docs/benchmark/v0_k8.json` | Parsed; 19 fixtures / 100 queries; all aggregates recomputed from scratch |

### 2.2 What was NOT executed (and therefore not claimed)

- The benchmark harness (`eval_v2/`) was **not re-run**. All benchmark figures in §4 are read
  from / recomputed out of the committed artifact `docs/benchmark/v0_k8.json`.
- No Qdrant, pgvector, or Ollama service was available, so no end-to-end ingest→search was
  measured against live services. Retrieval-path findings are from code reading plus the artifact.
- Python 3.10 and 3.11 were not installed on this machine (3.12 and 3.14 were available); the
  CI matrix covers 3.10–3.12.
- `mypy` and `ruff` were not run (see the note in §5.6 — the README's own "~30 pre-existing
  findings" claim is therefore unverified by this audit).
- The three delegated deep-dive agents (retrieval accuracy, benchmark credibility, production
  readiness) were stopped before completing; every finding below was re-derived directly from
  source by the auditor.

### 2.3 Codebase shape

```
atrium/
  __init__.py         42    config.py          465    smart_chunker.py  1156
  healthcheck.py      51    adapters.py        544    cli.py            1304
  models.py           53    rerank_adapter.py  552    main.py           1566
  embedder_config.py  94    retrieval.py       629    universal_parser.py 2277
  tsr_adapter.py     101    worker.py          240    pdf_backend.py     683
  vision_adapters.py 111    ocr_adapter.py     222    job_store.py       268
  layout_adapter.py  264    security.py        254    bm25.py            178
  mime.py            212    pipeline.py        177
                                                  ─────────────────────────
                                                  11,443 LOC total
```

Four files hold 55% of the code: `universal_parser.py` (2277), `main.py` (1566), `cli.py` (1304),
`smart_chunker.py` (1156).

---

## 3. Architecture assessment

### 3.1 The pipeline shape is right

```
bytes → parse → elements → chunk → embed → upsert          (ingest)
query → embed → search → rerank → dedup → parent-resolve  (retrieve)
```

This is the correct decomposition, and `atrium/pipeline.py` is the clearest expression of it.
`ingest_document()` (`pipeline.py:83-152`) is a clean 5-step function with no hidden I/O ordering
surprises, and `run_ingestion()` wraps it with job-store bookkeeping for the queue path. The split
between "worker-free core" and "queue-aware wrapper" is the right split, and it means the CLI, the
HTTP service, and the worker all execute the *same* ingestion code path.

### 3.2 What is genuinely good

**Protocol-based seams.** `atrium/adapters.py:112-148` defines `VisionAdapter`, `EmbeddingAdapter`,
and `VectorDBAdapter` as `typing.Protocol` classes. Qdrant and pgvector are interchangeable behind
them, and `UniversalParser` takes a vision adapter and an OCR adapter without knowing their type.
This is the right design and it is consistently honoured.

**Lazy heavy imports.** The `__getattr__` PEP-562 shim (`adapters.py:37-50`) resolves
`QdrantClient` / `ConnectionPool` / `register_vector` on first access, so a pgvector deployment
never pays the ~300 ms `qdrant_client` import. The module docstring documents the measurement
that motivated it. `httpx` (~48 ms) is deferred in the embedder and the vision adapter. `PyMuPDF`
is not imported at module scope in the parser.

**One config resolver.** `atrium/config.py` is the strongest module in the package. It documents
three real bugs it exists to fix (`.env` ignored by uvicorn; `OLLAMA_BASE_URL` had four different
defaults across four modules; `atrium doctor` lying about `API_KEY`), and it fixes them with a
single `DEFAULTS` table and a documented precedence rule
(`real environment > .env > built-in default`, `config.py:99-150`, `219-244`). `_SHELL_ENV` is
captured at import so the shell always wins over a file. `env_str` treating an empty value as
unset is a deliberate, documented choice.

**Per-chunk provenance.** `ChunkRecord.chunking_strategy` (`models.py:58`) records which strategy
produced each chunk, stamped by the chunker and copied verbatim by `_build_chunk_record`
(`pipeline.py:63`). The rationale in the docstring — *"a corpus with mixed strategies stays
auditable"* — is exactly right, and it is the kind of thing most projects omit and then regret.

**Explicit schema-version contract.** `atrium/retrieval.py:20-26` states the v0/v1 contract
plainly, and `collection_schema_version()` fails *closed to v0* on any adapter that does not
implement it (`retrieval.py:104-111`). That is the correct default.

### 3.3 Architectural weaknesses

**A1 — The collection name is a lie (P1).**
`VectorDBAdapter.ensure_collection(name, dim)` (`adapters.py:123`) is part of the published
protocol, and `QdrantAdapter.ensure_collection` honors `name` for existence-checking and creation
(`adapters.py:184-189`). But every other method hardcodes `"documents"`:

```python
adapters.py:189  create_collection(collection_name=name, ...)      # honors name
adapters.py:198  self.client.upsert(collection_name="documents", ...)   # ignores it
adapters.py:206  collection_name="documents",   # delete_stale_chunks
adapters.py:253  collection_name="documents",   # collection_schema_version
adapters.py:315  collection_name="documents",   # retrieve_parents
```

*Impact:* `db.ensure_collection("my_corpus", 1024)` creates and dimension-checks `my_corpus`, then
writes every chunk to `documents`. The dimension guard protects the wrong collection. It works today
only because every internal caller passes the literal `"documents"`. Any external user who reads the
protocol signature and passes a name gets silently wrong behaviour.

**A2 — Retrieval semantics are a function of corpus *data* (P1).**
`retrieve_and_rerank` calls `collection_schema_version(db)` on every invocation
(`retrieval.py:222`), which for Qdrant is a filtered `count` (`adapters.py:252-258`) and for
pgvector is `SELECT COUNT(*) FROM chunks WHERE (payload->>'schema_version')::int >= 1`
(`adapters.py:439-441`) — a full table scan with a JSONB expression, on the hot path, per query.

Worse, the answer flips the algorithm. A corpus where no chunk carries `schema_version >= 1` gets
the flat v0 path; the moment one parent-child document is ingested, **the entire corpus** switches
to the v1 path, where `search_children` filters `is_parent == False`
(`adapters.py:294-296`). Points written by an older version of Atrium — which the adapter's own
docstring acknowledges may lack the field (`adapters.py:228-229`) — do **not** match a Qdrant
`MatchValue(value=False)` filter, because a Qdrant filter on an absent payload key does not match.
So the upgrade path is: ingest one v1 document, and every legacy flat document silently vanishes
from search results.

*Impact:* a corpus-wide behaviour change triggered by data, not by configuration or migration.
This is the single most serious architectural defect in the package.

**A3 — The package is four files too large (P2).**
`universal_parser.py` (2277 lines) holds 16 format parsers plus the dispatch chain; `main.py`
(1566) holds 15 routes, the middleware, the retrieval-lab sessions, and the visualizer; `cli.py`
(1304) holds 11 commands. `universal_parser.py` is the worst: it is a single import unit, so
`atrium init` (which imports the CLI, which imports the parser) pays for the whole parser stack.

**A4 — The `main`/`worker`/`cli` triplet duplicates runtime construction (P2).**
`worker.py:69-130` (`_build_parser_and_ocr`, `_build_embedder`, `_build_db`, `_build_job_store`)
duplicates `main.py:180-210` and `cli.py:884` (`_build_runtime`). The divergence risk is real and
was already realized once — `embedder_config.py:9-21` documents that the HTTP path hardcoded
`nomic-embed-text`/768 while the worker used `bge-large`/1024, producing vectors of incompatible
dimensions in the same collection. The fix (one `build_embedder()` factory) was correct; the
*pattern* that caused it was not removed.

**A5 — `atrium/__init__.py` mutates the process-wide environment at import (P1).**

```python
atrium/__init__.py:33-38
from atrium.config import load_env_file as _load_env_file
_load_env_file()
```

Importing `atrium` — for any reason, from any directory — reads `./.env` from the current working
directory and **mutates `os.environ` globally**. The intent (fixing uvicorn ignoring `.env`) is
sound, and the `.env`-vs-shell precedence is correctly implemented. But a *library* must not write
to the host application's global environment at import time. A downstream app that happens to have
a `.env` in its CWD will have atrium's variables injected into it, including `API_KEY`, `PG_CONN`,
and `CORS_ALLOW_ORIGINS`. Conversely, atrium's own configuration becomes dependent on the caller's
CWD, which makes atrium's behaviour non-deterministic across deployment contexts.

*Fix:* keep the `.env` load, but make it opt-in (`ATRIUM_LOAD_DOTENV=1`, default on only for the
`atrium` console script and `uvicorn` entry point), or load into a private mapping rather than
`os.environ`.

---

## 4. Retrieval accuracy and benchmark credibility

### 4.1 The retrieval mechanism — assessment

The RAG code is the most carefully written part of the package.

**RRF fusion is textbook-correct** (`retrieval.py:332-471`). Ranks are 1-based, the smoothing
constant is pinned at `RRF_K = 60` (`retrieval.py:63`) with an explicit comment about avoiding the
"tune k to make the experiment look good" trap, hits present in both lists get the summed
contribution, and the fused list is truncated to the dense candidate budget — fusion cannot
manufacture candidates that neither ranker surfaced. The BM25-only-hit path correctly reconstructs
a full hit shape from the per-doc metadata so the downstream dedup/parent-resolve code does not
receive a half-populated dict.

**The confidence gate is well-reasoned** (`retrieval.py:114-155`). The one-candidate case is
handled deliberately rather than accidentally: the docstring says *"we deliberately do NOT treat it
as highly confident by assigning the missing second score to zero. That's a trap that would route
every 'singleton' dense result to dense-only and skip the BM25 safety net."* That is the kind of
edge case most implementations get wrong.

**The BM25 implementation is correct Okapi BM25** (`atrium/bm25.py:140-178`), including the
Robertson–Sparck Jones IDF variant with the `+ 1.0` that keeps IDF non-negative when `df > N/2`,
and the `_df` table is correctly decremented when a document is re-added with different terms
(`bm25.py:112-129`). Tokenization uses `re.findall(r"\w+", ..., re.UNICODE)`, which handles Latin,
CJK, Arabic, and accented Latin.

**Parent-child resolution keys line up correctly.** The chunker sets `child.parent_id =
parent.chunk_id` (`smart_chunker.py:1003`) with `chunk_id = str(uuid.uuid4())`; Qdrant's
`retrieve_parents` keys its result by `payload.get("chunk_id") or p.id` (`adapters.py:325`); and
`_resolve_parents` looks up `parent_map.get(pid)` where `pid` is the hit's `parent_id`
(`retrieval.py:564`). **Verified consistent.** The `is_parent`/`schema_version` derivation in
`_build_chunk_record` (`pipeline.py:36-39`) is deterministic and correct: parents get
`is_parent=True` (non-empty `child_ids`), children get `is_parent=False` but `schema_version=1`
via their `parent_id`. The Qdrant point ID is a UUID string, which Qdrant accepts. No bug found
here.

### 4.2 Retrieval defects

**R1 — Hybrid BM25 is unreachable from the public API (P1).**

```
$ grep -rn "bm25\|BM25" atrium/*.py
atrium/bm25.py        (the module itself)
atrium/__init__.py:8  (docstring: "hybrid (vector + BM25) fusion")
atrium/main.py:1063-1067  (retrieval-lab sandbox ONLY)
```

`/search` (`main.py:439-448`) calls `retrieve_and_rerank` **without** `bm25_index`. Therefore
`_should_consult_bm25` returns `False` at its first branch (`retrieval.py:143-144`) on every
production query.

*Consequences:*
- The `DENSE_CONFIDENT_GAP` and `DENSE_MIN_SCORE` environment variables — documented in
  `config.py:132-133`, present in `DEFAULTS`, and the subject of a dedicated commit
  ("Adding confidence-gated hybrid retrieval (opt-in)") — **have no effect whatsoever in
  production**. An operator who tunes them sees nothing change.
- `atrium/__init__.py:8-9` advertises hybrid fusion as a product capability of the package.
- `atrium/bm25.py` ships in the wheel but is unreachable from any documented entry point.

This is the "UI overstates capability" pattern in library form: the public surface describes a
feature the public surface cannot invoke.

**R2 — `top_k` is violated when reranking fails (P1).**

```python
retrieval.py:494-501
    except Exception:
        reranked = None
    if reranked is None:
        out = list(candidates)      # <-- ALL candidates, NOT truncated to top_k
        out.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return out
```

The success path returns at most `top_k` (the reranker truncates). The failure path returns the
entire candidate set — `candidate_k`, which defaults to **30** (`retrieval.py:235`).

The v1 path is protected: `_dedup_by_parent(hits, top_k)` truncates (`retrieval.py:319`). The v0
path is **not** protected — it only maps defaults (`retrieval.py:322-323`) and never truncates.

*Failure scenario:* a client sends `{"query": "...", "top_k": 8}` to `/search` on a v0 corpus
(the default `auto`/`precise` chunking produces flat chunks with `schema_version=0`, so v0 is the
default production corpus). The reranker is enabled but its model fails to load — no network for
the HuggingFace download, OOM, `sentence_transformers` not installed. `_ensure_loaded()` raises
*outside* the reranker's own try/except (`rerank_adapter.py:174` precedes the `try` at line 180),
so the exception propagates to `_apply_rerank`'s bare `except Exception`. The client receives
**30 results for a `top_k=8` request**, and `/search` reports `"count": 30`. The response also
reports `"rerank": false` (because `reranker` was not `None` — it was constructed successfully),
so the response is internally contradictory.

*Fix:* truncate to `top_k` in the fallback branch, and log the swallowed exception (see R4).

**R3 — `score` changes meaning depending on whether the gate tripped (P2).**
When the hybrid path fuses, `score` is overwritten with the RRF value (`retrieval.py:469`,
`1/(60+rank)` ≈ 0.016–0.033). When the gate does not trip, `score` stays a cosine similarity
(0.0–1.0). The README's `/search` response example shows `"score": 0.78`. A client that
threshold-filters on `score` would behave completely differently depending on an invisible
internal heuristic. Currently latent (because R1 means the gate never trips in production), but it
must be fixed before hybrid search is exposed.

**R4 — Rerank failures are swallowed with no logging (P2).**
`retrieval.py:494` is a bare `except Exception: reranked = None`. Nothing is logged, nothing
counted. An operator whose reranker is failing sees the vector ranking and has no signal why.

**R5 — `BM25Index.add_doc` is O(N) per call, making index build O(N²) (P1).**

```python
bm25.py:129
self._avgdl = sum(self._dl.values()) / max(1, self.n_docs)
```

This recomputes the sum over **every** document's length on **every** `add_doc` call. Building an
index of N documents costs O(N²) token-count operations. The module docstring says the 19-set
benchmark builds ~12k chunks, all rebuilt per fixture — 12,000²/2 ≈ 72M redundant operations per
rebuild, per fixture, per configuration. This is the single largest avoidable cost in the sparse
path and it is a two-line fix (accumulate a running sum).

**R6 — BM25 search is a linear scan, not an inverted index (P2).**
`bm25.py:163` iterates `self._tf.items()` for **every query term**. The docstring claims
*"Search is O(|Q| * postings-touch)"* — the implementation is O(|Q| × N). The docstring and the
code disagree. Not a correctness bug, but a documented-vs-actual mismatch in a module whose whole
purpose is speed, and a real cost at 10⁵+ chunks.

### 4.3 The benchmark — this is where the credibility problem is

The README's single quantitative claim is: **"19/19 fixtures, mean R@8 = 0.8976"**, rendered as a
badge over a benchmark image. Everything below is recomputed from `docs/benchmark/v0_k8.json`, the
committed artifact the image is rendered from.

**B1 — The headline metric is mislabeled (P0).**

```
Keys present in the artifact:
  recall_at_5, mean_recall_at_5, ndcg_at_5, mean_ndcg_at_5, mrr, mean_mrr
  (no recall_at_8, no mean_recall_at_8, no ndcg_at_8 — verified exhaustively)

MACRO mean of mean_recall_at_5 = 0.8976   <-- exactly the README's number
MICRO mean of per-query recall_at_5 = 0.8933
```

The number 0.8976 **is** a Recall@5. The README calls it R@8. Retrieval depth was 8
(`n_retrieved` reaches 8 on every fixture with more than 8 chunks), so the correct statement is
*"Recall@5 = 0.8976, measured with top_k=8 retrieval"* — not *"mean R@8 = 0.8976"*.

**B2 — Recall is measured at page level, not passage level (P0).**

```
metric_level distribution across all 100 queries: {'page': 100}
```

A "hit" means *the retrieved chunk came from the correct page*. On a page containing three chunks,
retrieving any of them scores a hit. For the table-heavy PDF fixture, retrieving the paragraph
*adjacent* to the relevant table scores a full recall hit. This is a materially weaker criterion
than retrieving the correct content, and it systematically inflates the number — most on exactly
the document types a RAG product cares about (tables, forms, multi-column layouts).

**B3 — The macro-average is dominated by trivially-saturated fixtures (P0).**

| Fixture | n_chunks | mean_recall_at_5 |
|---|---|---|
| Acme Q3 2024 financial report (clean PDF) | 3 | 1.0000 |
| Quarterly sales by region (table-heavy PDF) | 5 | 1.0000 |
| New hire onboarding checklist (list-heavy PDF) | 5 | 1.0000 |
| Scanned PDF (image-only) | 6 | 1.0000 |
| US State Population (synthetic XLSX) | **2** | 1.0000 |
| Nexus Analytics product brochure | 4 | 1.0000 |

Six of nineteen fixtures have ≤6 chunks and score a perfect 1.0000. On a 2-chunk corpus,
"recall@5" reduces to "did the right chunk come back at all" — and it always does, because there
are only two of them. These six contribute a perfect score to the macro-average.

```
6 fixtures with n_chunks <= 6  → mean R@5 = 1.0000
13 remaining fixtures          → mean R@5 = 0.8504
```

The headline 0.8976 sits between the two. A reader is told "0.8976"; the honest characterization is
*"perfect on six tiny fixtures, 0.85 on the thirteen that have real corpora."*

**B4 — The benchmark exercises neither reranking nor parent-child (P0/P1).**

```
rerank_latency_ms non-null : 0 / 100 queries
parent_resolved == True    : 0 / 100 queries
```

Every published number is **dense-only, flat-schema, no-rerank**. Two of the package's three
retrieval features — the reranker and the v1 parent-child path — are completely unbenchmarked.
The package ships them and the README documents them; the evidence file contains no measurement
of either.

**B5 — Answers are not verified to be in the retrieved text (P1).**

```
evidence_in_context non-null : 6 / 100 queries
expected_in_top_50 non-null  : 0 / 100 queries
```

For 94 of the 100 queries, the only evidence that retrieval "worked" is that a chunk from the
right *page* came back. There is no check that the retrieved text actually contains the answer.
Combined with B2, this means the benchmark measures *"did we hit the right page"*, which is a
necessary but not sufficient condition for useful retrieval.

**B6 — The benchmark is not reproducible (P0).**

```
.gitignore:128-130
  /tests/benchmarks/fixtures/
  /tests/parsing_fixtures/
```

The corpus is ~49 MB of third-party documents (a W-2 form, a 10-K, arXiv papers, Apple filings,
Gutenberg books, Wikipedia pages) that the `.gitignore` comment explicitly declines to ship
because of redistribution licences. The `eval_v2/` harness is not in the sdist (confirmed by
listing the archive). `docs/benchmark/v0_k8.json` is the *result*, not the *harness*.

*Consequence:* an independent developer cannot re-run the benchmark, cannot confirm 0.8976, cannot
produce a new number on a new corpus, and cannot file a meaningful accuracy bug. The README's own
Development section concedes this: *"That corpus is not in the repository, so those tests skip
until it is added."*

**B7 — Latency figures are not a production signal (P2).**

```
search_latency_ms: n=100  min=3.34  p50=5.79  p95=8.32  max=10.84
```

Each fixture is an isolated corpus of at most 748 chunks, queried one at a time against a local
embedding service. These numbers say nothing about latency at 10⁵–10⁶ chunks, under concurrent
load, with a remote embedding endpoint, or with the pgvector backend (which has a different query
plan — see §5.3).

**B8 — Reported metrics are selectively chosen (P1).**
The artifact contains `mean_mrr = 0.7058` and `mean_ndcg_at_5 = 0.7381`. Neither appears in the
README. Recall is the least discriminative of the three metrics and the only one shown. Reporting
the ranking metrics alongside the recall number would be more honest and would cost one line.

### 4.4 Failure cases visible in the data

The artifact's own `classification` field is informative and the author clearly did the work here:

```
PASS                : 94 / 100
RETRIEVAL_FAILURE   :  3 / 100
MISSING_FROM_CORPUS :  3 / 100
```

The three `MISSING_FROM_CORPUS` questions are scored 0.0 and are correctly penalized — this is the
honest choice and it should be kept. They are: *"What is the total project budget?"* (ODT),
*"典型的人工神经网络由哪三个部分组成？"* and *"反向传播算法是用来做什么的？"* (both zh.wikipedia HTML).

The three `RETRIEVAL_FAILURE` cases are the interesting ones, and they are near-misses, not
blowups:

| Question | Fixture | top-1 similarity |
|---|---|---|
| "What new neural network architecture does this paper propose" | BERT (arXiv 1810.04805) | **0.744** |
| "What does box 1 on Form W-2 report?" | IRS Form W-2 (2026 revision) | **0.757** |
| "مَن طوّر شبكة LeNet-5 في عام 1998؟" | Arabic Wikipedia | 0.406 |

The first two retrieved chunks with *high* cosine similarity that were nevertheless wrong — this
is exactly the failure mode a cross-encoder reranker exists to fix, and it is the strongest
argument for shipping the reranker. But the reranker is (a) off by default, (b) never benchmarked
(B4), and (c) reloaded per request (R2's performance twin). The benchmark contains direct evidence
that the default configuration fails on these cases and no evidence that the alternative works.

The `metric_level: "page"` design means the BERT failure ("what architecture does this paper
propose") is scored as a miss because the answer paragraph is on a different page than the
retrieved one — a page-level metric penalizing a passage-level success is plausible here too.

### 4.5 Missing test scenarios

Not present anywhere in the 100 questions, and not covered by any committed fixture:

- **Multi-hop / compositional questions** — every question is single-hop and answerable from one
  place. There is no "compare the revenue in Q3 of the Acme report against the Apple filing."
- **No-answer / unanswerable questions** as a *designed* category. The three
  `MISSING_FROM_CORPUS` items are incidental (the content was never in the document), not a
  curated "the corpus does not contain this" test set. A RAG product needs a refusal path and
  there is no benchmark for it.
- **Adversarial / near-miss questions** — questions whose keywords appear in a distractor chunk.
- **Multi-turn / conversational retrieval** — no follow-up, no pronoun resolution across turns.
- **Cross-document retrieval** — every fixture is queried in isolation. A production corpus is
  many documents; nothing in the benchmark measures cross-document interference, or whether the
  chunker+embedder can tell two documents apart.
- **Chunk-boundary sensitivity** — no question is deliberately placed at a chunk edge.
- **Table/structured data as the *answer*** — the table-heavy PDF fixture exists but the
  `metric_level: "page"` design means a table row is never verified to be retrieved.
- **Long documents** — the largest fixture is 748 chunks (Tom Sawyer). Real corpora are 10⁴–10⁶.
- **The `ocr` and `vision` paths** — the Scanned PDF fixture exists but OCR is off by default and
  the `ocr` extra is not installed in the clean environment, so the scanned-PDF fixture was
  parsed by whatever path was active at benchmark time. `OCR_ENABLED` is not recorded in the
  artifact, so this is not auditable.

### 4.6 Section verdict

The retrieval *mechanism* is well-designed and mostly correct — I found no correctness defect in
the RRF math, the BM25 math, the parent-child key contract, or the schema-version derivation.
The retrieval *evidence* is where this fails: the headline number is mislabeled, the metric is
page-level, a third of the fixtures saturate trivially, two of three retrieval features are
unbenchmarked, and the whole thing is unreproducible outside the author's disk.

---

## 5. Performance assessment

### 5.1 What is done right

- **Blocking work is offloaded.** `asyncio.to_thread` is used for the embedder
  (`main.py:433`), the retrieval call (`main.py:439`), the whole ingestion
  (`main.py:533`), and the parse+chunk in retrieval-lab (`main.py:1011`). The event loop is not
  blocked by PyMuPDF, Qdrant, or Ollama.
- **Heavy imports are deferred** with measured rationale (`adapters.py:18-27`: qdrant_client ~300 ms,
  psycopg_pool ~65 ms, httpx ~48 ms).
- **Ingestion concurrency is bounded** by an asyncio semaphore (`INGEST_CONCURRENCY`, default 4).
- **Embedding is batched** at 100 texts per request (`pipeline.py:134-139`).
- **Qdrant's `search`/`query_points` rename** is handled (`adapters.py:53-79`), with an accurate
  explanation of the 1.20 breakage. This is a genuinely good catch that would have broken every
  `/search` for anyone installing fresh.

### 5.2 Measurable bottlenecks

**P1 — The reranker model is reloaded on every request.**
`main.py:414` calls `build_reranker(force=True)` per request; `build_reranker` constructs a new
`LocalBgeReranker` (`rerank_adapter.py:544`); `_ensure_loaded` caches only *within an instance*
(`rerank_adapter.py:148-164`). So each request performs a fresh `CrossEncoder(...)` construction —
loading ~300 MB–1.3 GB of weights from the HuggingFace cache and running model init, per query.
`FlashRankReranker` has the identical structure (`rerank_adapter.py:269-288`). `CohereReranker` is
HTTP-based and unaffected.

*Fix:* a module-level `@lru_cache` on the adapter construction, or a process-wide singleton keyed
by `(backend, model, device, max_length)`.

**P1 — `collection_schema_version()` is a full-collection scan per query.**
Qdrant: `count` with a `Range(gte=1)` filter on `schema_version`, with **no payload index** created
for that key (`adapters.py:252-258`). pgvector: `SELECT COUNT(*) FROM chunks WHERE
(payload->>'schema_version')::int >= 1` (`adapters.py:439-441`) — a sequential scan of the whole
table with a JSONB expression per row. On a 10⁶-chunk corpus this dominates query latency.

*Fix:* store the schema version as collection-level metadata (Qdrant supports it) or in a
one-row side table; cache it with a TTL. Also create the payload indexes for `doc_id`,
`ingest_run_id`, `is_parent`, and `schema_version` — `ensure_collection` creates none
(`adapters.py:181-189`), so every filter is unindexed.

**P1 — pgvector upsert issues one INSERT per chunk.**
`adapters.py:385-393` loops over every chunk with an individual `cur.execute`. The `batch_size`
parameter is accepted and **never used**. For a 12k-chunk document that is 12,000 sequential
round trips to Postgres, inside a retry-wrapped function. `executemany` or `COPY` would reduce
this to a handful of round trips.

**P1 — `BM25Index` builds in O(N²)** — see R5 above.

### 5.3 Memory and concurrency

- **Upload buffering is bounded by `Content-Length`.** `check_content_length` runs before the read
  (`main.py:584`), so the total request body is capped at `MAX_UPLOAD_BYTES` (100 MB default).
  `check_payload_size` re-verifies after the read (`main.py:526`). The comment at `main.py:582-583`
  says *"Reject oversize uploads BEFORE any bytes are read into memory"* — accurate for the total
  request, but note that `/ingest` accumulates **all** files into one list
  (`main.py:594`: `payloads = [(await f.read(), ...) for f in files]`), so the full 100 MB is
  resident in the process before any work starts. That is bounded and acceptable, but it is a
  buffer, not a stream.
- **The async job path stores base64 in a SQLite TEXT column.** `enqueue_job` reads the file
  (`main.py:663`), caps it at 50 MB (`_ASYNC_JOB_MAX_BYTES`, `main.py:635` — hardcoded, not
  configurable), then base64-encodes it into the job row (`job_store.py:154`). A 50 MB document
  becomes a ~67 MB TEXT blob in the jobs table. Binary in a SQL row is a design smell that makes
  the queue table grow without bound and makes job inspection awkward.
- **The SQLite queue has no WAL mode and no explicit `busy_timeout`** (`job_store.py:74-76`). The
  default 5-second busy timeout applies. `claim_next` uses `BEGIN IMMEDIATE`
  (`job_store.py:166`) which is the correct primitive for cross-process atomic claiming, and the
  `threading.Lock` serializes access within a process. The README's claim that *"Workers claim
  jobs atomically, so several can share one queue file without processing a document twice"* is
  **mostly true but has a hole** — see P1-11 below.
- **pgvector's `search` plan is unexamined.** The query
  `ORDER BY embedding <=> $1::vector LIMIT $2` (`adapters.py:471-476`) is the correct pgvector
  idiom, but there is **no IVFFlat or HNSW index** created anywhere — `ensure_collection` creates
  only the `vector` column (`adapters.py:374-382`). Without an index, pgvector performs an exact
  sequential scan over every chunk on every query. This is a **production-blocking omission** for
  the pgvector backend, and it is not mentioned in the README's Limitations.

### 5.4 Large-document and concurrent-ingestion behaviour

- **No document-size limit on parsing.** `MAX_UPLOAD_BYTES` caps the request, but a 100 MB PDF
  expands into potentially hundreds of thousands of text elements in memory, and the chunker then
  materializes all chunks (`pipeline.py:127`) before embedding. Peak memory is
  `raw_elements + all chunks + all embeddings`, all resident simultaneously. There is no streaming
  ingest path. For a 100 MB upload this is a multi-gigabyte spike.
- **Embedding is a fixed batch of 100** (`pipeline.py:134`) with no concurrency across batches and
  no retry on the embedder call itself. `OpenAICompatibleEmbedder.embed` has `timeout=60.0`
  (`adapters.py:159`) and no retry — a single transient failure aborts the whole ingestion. Note
  that `adapters.py` imports `tenacity` and applies `@retry` to the *database* writes
  (`adapters.py:191, 200, 384, 395`) but **not** to the embedder. The most failure-prone external
  dependency is the only one without a retry.
- **Concurrent ingestion is bounded but not sharded.** The semaphore caps in-flight documents at
  `INGEST_CONCURRENCY` (default 4), and each document's embedding is a synchronous blocking loop
  inside one thread. Four concurrent documents saturate four threads; the embedder is called
  serially per document with batches of 100. There is no cross-document batching, which is where
  the real throughput win is.

### 5.5 What I did not measure

No live Qdrant, pgvector, or Ollama was available in this environment, so **no throughput, no
end-to-end latency, and no memory measurement was performed**. Every performance finding above is
derived from code structure — algorithmic complexity, per-request model construction, missing
database indexes, per-row SQL — not from a timing run. Those are the findings where the code
itself states the cost (e.g. `adapters.py:18-27` documents the import timings). **A throughput
benchmark does not exist in the repository**, and none was produced here.

---

## 6. Python packaging audit

This was the most favourable part of the audit, and the improvements below are correspondingly
smaller.

### 6.1 What was verified working

| Check | Result |
|---|---|
| Build backend | `setuptools.build_meta`, `requires = ["setuptools>=77", "wheel"]` — modern, correct |
| Wheel build | ✅ `atrium_rag-0.1.0-py3-none-any.whl`, pure-python, no compiled extensions |
| Sdist build | ✅ `atrium_rag-0.1.0.tar.gz` |
| `twine check` | ✅ **PASSED** on both artifacts |
| Wheel → clean venv (3.12.13) | ✅ Installed, `pip check` clean, all 22 direct deps resolved |
| Sdist → clean venv | ✅ Installed, smoke test passed |
| Console script | ✅ `atrium` lands in `bin/`, `atrium --help` renders all 12 commands, `atrium version` → `0.1.0` |
| Public submodule imports | ✅ `atrium.adapters`, `atrium.config`, `atrium.embedder_config`, `atrium.job_store`, `atrium.pipeline`, `atrium.retrieval`, `atrium.smart_chunker`, `atrium.universal_parser` |
| `py.typed` | ✅ Present in the wheel; `package-data` declares it |
| Metadata | ✅ `Metadata-Version: 2.4`, `License-Expression: MIT`, `License-File: LICENSE`, `Requires-Python: >=3.10` |
| PEP 621 metadata | ✅ Name, version, description, readme, license, authors, keywords, classifiers, URLs all present |
| Entry point | ✅ `[console_scripts] atrium = atrium.cli:app` |
| `pip check` | ✅ No broken requirements |
| Test suite | ✅ **225 passed, 7 skipped, 5 xfailed** |
| Secret hygiene | ✅ `.env` is gitignored (`.gitignore:35`) and **not tracked** (`git ls-files .env` → empty) |
| Dependency hygiene | ✅ No upper bounds on runtime deps, letting security patches through |
| Import isolation | ✅ `atrium.adapters` uses PEP 562 `__getattr__` so importing it doesn't pull in `qdrant_client` |

The README's install path is accurate: `pip install atrium-rag` → `atrium` on PATH → works.

### 6.2 Packaging defects

**K1 — AGPL-3.0 dependency in an MIT package (P0).**
`pyproject.toml:57` declares `"pymupdf>=1.24.0"` as a hard runtime dependency. The installed
distribution's own metadata reads:

```
License: Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License
```

and the project's own `atrium/pdf_backend.py:25` states `PyMuPDF AGPL-3.0`. The repository
`LICENSE` is MIT. The `pdf_backend.py` module docstring even argues for permissiveness —
*"The alternatives are permissively licensed, so the permissive build can be the default"* — and
then selects the AGPL one as the default (`pdf_backend.py:679`), because
`ATRIUM_PDF_BACKEND` defaults to `pymupdf`.

*Impact:* this is not a nit; it determines whether the package can ship to PyPI under MIT at all.
PyMuPDF's licensing terms apply to applications that link it. **This needs a legal decision, not a
code change**, and the two viable resolutions are (a) obtain an Artifex commercial licence and
keep PyMuPDF, or (b) make `pdfplumber` the default and PyMuPDF an optional extra. Note that the
`pdf_backend.py` docstring already argues (b) is the right engineering choice — the licensing
argument and the implemented default contradict each other inside the same file.

**K2 — `MANIFEST.in` `exclude tests` does not exclude the test files (P1).**

```
# MANIFEST.in:35
exclude tests
```

`exclude` matches a single path segment; it does not recurse. The built sdist contains all 17
test files (`atrium_rag-0.1.0/tests/test_*.py`), directly contradicting the sdist header comment
*"Exclude dev/test/build artifacts from sdist"*. Verified by listing the archive. *Fix:*
`recursive-exclude tests *`.

**K3 — The `dev` extra is missing `build` and `twine` (P1).**
`pyproject.toml:130-135` declares `dev = [pytest, pytest-asyncio, ruff, mypy]`. The README's
Extras table claims `dev` adds *"pytest, ruff, mypy, **build**, **twine**"*, and the `Makefile`
runs `$(VENV)/bin/python -m build` and `$(VENV)/bin/twine check` (`Makefile:38-40`). After a clean
`make install` (`pip install -e ".[dev]"`), **`make build` fails** — `build` and `twine` were never
installed.

**K4 — The `alt-parsers` extra installs ~2 GB for a backend that does not exist (P1).**
`pyproject.toml:110-112` declares `alt-parsers = ["docling-slim[format-pdf,models-local]>=2.0.0"]`
with the comment *"Docling ships ~2GB of models."* The README states *"Install `atrium-rag[alt-parsers]`
for Docling as an alternative PDF parser."*

```
$ grep -rn "docling" atrium/*.py
(no matches)
```

The only selectable PDF backends are `pymupdf` and `pdfplumber`, chosen via `ATRIUM_PDF_BACKEND`
(`pdf_backend.py:651-654`). **There is no Docling backend.** A user following the README pays a
2 GB install for nothing.

**K5 — `pdfplumber` is selectable but undeclared and undocumented (P1).**
`pdf_backend.py:651-654` registers `PdfPlumberBackend`, and `pdf_backend.py:42-43` documents
`ATRIUM_PDF_BACKEND` with both values. But `pdfplumber` is **not** in `dependencies`, **not** in
any extra, and **not** in the README. Setting `ATRIUM_PDF_BACKEND=pdfplumber` yields a bare
`ImportError` with no install hint.

**K6 — `requires-python = ">=3.10"` has no upper bound (P2).**
The CI matrix is 3.10–3.12 (`ci.yml`), but the metadata permits any future version. In this
environment, a clean install resolved `pymupdf 2.0`, `starlette 1.10.0`, `fastapi 0.162.0`, and
`typer 0.27.3` — **four major-version jumps past anything CI has ever tested**. The `_qdrant_search`
shim (`adapters.py:53-79`) already exists because `qdrant-client` renamed an API in 1.20; the
same class of breakage is possible for every one of these. At minimum, CI should add 3.13/3.14.

**K7 — The sdist excludes the `sdk/` directory and the benchmark harness (P2).**
`sdk/client.py` (26 lines) ships in the repo but is in neither the wheel nor the sdist, and is
undocumented in the README. `eval_v2/` is likewise excluded. Neither is clearly wrong, but an
undocumented, unpackaged directory is dead weight from a reviewer's point of view.

**K8 — The README's Extras table is wrong in two places (P2).**
Beyond K3 and K4: the table lists `vision` as *"marker only; the adapter is httpx-only"*, which is
accurate but reads as a non-extra. And the CLI table omits `atrium version` — a command the
project's own smoke test depends on (`scripts/smoke_test.py`, `check_console_script`).

**K9 — `atrium` is an occupied name on PyPI (P2, correctly handled).**
`pyproject.toml:13-22` documents that `atrium` is taken (an unrelated "MX API" package) and
that the distribution is therefore named `atrium-rag` while the import package and console script
remain `atrium`. This is a well-handled naming collision, and the comment explicitly tells forks
to pick their own name. **Good practice — noted as a positive.**

### 6.3 API stability

- `atrium/__init__.py` exports only `__version__`. The `__all__` is `["__version__"]` — a library
  whose root namespace exports nothing but a version string.
- The real public API is the submodule surface: `atrium.pipeline`, `atrium.adapters`,
  `atrium.retrieval`, `atrium.smart_chunker`, `atrium.universal_parser`,
  `atrium.embedder_config`. These are used consistently by the README's Python example, the CLI,
  the HTTP service, and the smoke test — the smoke test asserts exactly this set
  (`scripts/smoke_test.py`, `check_submodules`).
- `atrium/retrieval.py:604-629` keeps a back-compat alias `retrieve_with_parent_resolution` for
  legacy importers — thoughtful.
- `atrium/main.py:267` imports `BaseHTTPMiddleware` *after* module-level code, and `atrium/mime.py`
  documents that `main` re-exports the private names `_detect_mime` / `_match_signature` so old
  imports keep working (`mime.py:14-16`). Consistent back-compat discipline.
- **No `api.py` / public-surface module, and no `__all__` on the public submodules.** The
  de facto API is whatever is importable, which is all 24 modules. `atrium.pdf_backend`,
  `atrium.layout_adapter`, `atrium.tsr_adapter` are undocumented in the README but importable.
  There is no statement of what is stable.

### 6.4 Typing and linting

- `py.typed` is shipped and declared. ✅
- `mypy` is configured (`pyproject.toml:157-162`) and runs in CI — but as **`continue-on-error:
  true`** (`ci.yml:36`), i.e. advisory. The README's Limitations section says *"mypy is advisory.
  ~30 pre-existing Optional-narrowing findings"*. **This audit did not run mypy** (the command was
  blocked), so the "~30" figure is unverified. The `ignore_missing_imports = true` setting also
  means mypy is not checking any third-party API surface.
- `ruff` runs as a **blocking** CI job (`ci.yml:19-27`) pinned to `ruff==0.17.1`. The `select`
  list is `["E", "F", "I", "B", "UP"]` with a substantial `ignore` list, each entry carrying a
  rationale comment (`pyproject.toml:166-207`). That is exemplary practice.
- `ruff` is invoked on `atrium tests scripts` — but `scripts/` contains `render_benchmark_shot.py`
  and `smoke_test.py`, and `tests/` contains the fixture harness. `eval_v2/` is **not** linted,
  which is consistent with it being out of the distribution but means the benchmark code is
  unlinted.
- **Type hints are present but inconsistent.** `atrium/models.py` uses `typing.List` /
  `typing.Optional` (documented as deliberate in the ruff ignore list), but `atrium/main.py:73`
  uses `str | None`-style hints, and `atrium/ocr_adapter.py:202` uses `TesseractOCRAdapter |
  RapidOCRAdapter` (PEP 604) with `from __future__ import annotations`. The codebase mixes the
  styles by file. It works, but it reads as accretion rather than a decision.

---

## 7. Production-readiness assessment

### 7.1 Security

**Good:**
- `check_content_length` (`security.py:67-98`) rejects a missing `Content-Length` with 411 by
  default — chunked uploads are off unless explicitly enabled. That is a deliberate, defensible
  posture for a document-ingestion service.
- `check_payload_size` re-verifies the actual byte count after the read, covering a lying
  `Content-Length`.
- `check_mime` enforces an allowlist when `ALLOWED_MIME_TYPES` is set.
- `mime_conflict` (`mime.py:182-212`) rejects a declared/detected type-category disagreement — the
  "PDF filename, HTML body" case. The leniency rules (octet-stream = unknown, zip never conflicts)
  are well-reasoned and documented.
- Bearer auth uses `hmac.compare_digest` (`security.py:177`) — constant-time, correct.
- `redacted_traceback` strips file paths and source lines from client-facing errors.
- `safe_resolve` (`security.py:209-230`) implements OWASP-style path-traversal protection with
  `resolve()` + `relative_to()`.
- The vision adapters post to an **operator-configured** `base_url`, never to a user-supplied URL.
  **There is no SSRF vector.** I looked for one and did not find it.
- `TesseractOCRAdapter.transcribe` and `RapidOCRAdapter.transcribe` both catch everything and
  return `""` rather than crashing an ingest.

**Bad:**

**S1 — Raw exception text is returned to API clients (P1).**

```python
main.py:436   raise HTTPException(503, f"Embedder unavailable: {exc}") from exc
main.py:451   raise HTTPException(503, f"Retrieval failed: {exc}") from exc
```

`exc` for a psycopg connection failure contains the DSN (including the password, from
`PG_CONN`'s default `postgresql://user:pass@pgvector:5432/db`); for an httpx failure it contains
the full URL and connection error; for a Qdrant failure it contains the host and port. The
package has `redacted_traceback` and `log_redacted_traceback` for exactly this, and uses them in
only one place (`main.py:791-792`, `preview_chunks`). **This is the same information in the same
shape being handled correctly in one route and incorrectly in two others.** The failure message
should be generic; the detail belongs in the server log.

**S2 — `/docs`, `/openapi.json`, and `/redoc` stay open when `API_KEY` is set (P2).**
`security.py:64` puts them in `_OPEN_ROUTES`. The full API schema is public on an authenticated
deployment. Common practice varies, but for a service that is supposed to be behind a bearer
token, the schema should not be free.

**S3 — Job errors are stored and served raw (P2).**
`pipeline.py:177` does `job_store.update(job_id, status="failed", error=str(e))`, and
`GET /jobs/{job_id}` returns that string. A parse or DB failure leaks the same internal detail
that S1 leaks, through a different door.

**S4 — `JobStore.update` interpolates column names from `**kwargs` (P2).**
`job_store.py:228`: `set_clause = ", ".join(f"{k} = ?" for k in encoded)`. Every current caller
passes literal names, so this is not exploitable today — but it is a SQL-injection-shaped API on a
public method of a public class. A whitelist of permitted column names would close it.

**S5 — `atrium.__init__` writes to the global environment at import (P1)** — see A5.

**S6 — `TesseractOCRAdapter` mutates a process-global (P2).**
`ocr_adapter.py:65-71` sets `pytesseract.pytesseract.tesseract_cmd` globally. Two adapters with
different commands in one process → last writer wins. Minor in practice (there is one adapter per
process today), but it is a latent multi-tenant hazard.

### 7.2 Failure recovery and resource cleanup

- **No retry on the embedder.** `@retry` is applied to the Qdrant and pgvector writes
  (`adapters.py:191, 200, 384, 395`) but **not** to `OpenAICompatibleEmbedder.embed`
  (`adapters.py:156-162`). The embedding call is the most failure-prone external dependency and
  the one with no retry. A single 502 from Ollama aborts a whole ingestion.
- **The job queue has no retry policy.** A job that fails is marked `failed` and stays failed.
  There is no attempt counter, no backoff, no dead-letter queue. A transient embedder blip
  permanently loses the document. The `models.py` `JobStatus` literal has no `retrying` state.
- **`WORKER_JOB_TIMEOUT` is a documented knob that does nothing (P1).**

  ```python
  worker.py:180-181
  if JOB_TIMEOUT_SECONDS > 0:
      pass
  ```

  The comment above it is honest — *"Watchdog-thread timeout would be the production answer; for
  now release_stale_claims is the safety net"* — but the variable is in `DEFAULTS`
  (`config.py:146`) and looks like a working setting. A configuration knob that silently does
  nothing is worse than no knob.
- **Stale-claim recovery can double-process (P1).**
  `release_stale_claims` (`job_store.py:197-216`) returns any `running` job whose `claimed_at` is
  older than `JOB_STALE_SECONDS` (default **1800 s**) to `pending`. A worker that is legitimately
  ingesting a 100 MB PDF and takes longer than 30 minutes has its job stolen by another worker.
  Both then ingest it. The original worker finishes and calls
  `job_store.update(job_id, status="complete")`, marking the job complete while the second
  worker may still be running. The README's claim that *"several can share one queue file without
  processing a document twice"* is **not guaranteed** for long documents.
- **Signal handlers are installed at import time (P2).** `worker.py:63-64` calls
  `signal.signal(SIGTERM, ...)` at module scope. Importing `atrium.worker` from any context
  hijacks the process's signal handlers. It belongs in `main()`.
- **No SQLite WAL mode and no explicit `busy_timeout`** (`job_store.py:74-76`). Readers and
  writers block each other. Under WAL, concurrent readers do not block the writer.

### 7.3 Database consistency and migrations

- **There are no migrations.** `PgVectorAdapter.ensure_collection` issues a single
  `CREATE TABLE IF NOT EXISTS chunks (...)` (`adapters.py:374-382`). There is no schema-version
  table, no migration runner, no `ALTER TABLE` path. Changing the payload shape or adding a
  column requires manual intervention.
- **The dimension check is genuinely good and is one of the better pieces of engineering here.**
  Both backends detect a dimension mismatch and refuse to proceed:
  - Qdrant (`adapters.py:184-187`) raises `ValueError` with both numbers.
  - pgvector (`adapters.py:359-372`) reads `atttypmod` from `pg_attribute`, subtracts the 4-byte
    `VARHDRSZ` header, and raises with a remediation message. It correctly handles the
    `to_regclass('chunks')` NULL case for a fresh database — the comment explains that the
    previous `'chunks'::regclass` cast raised `UndefinedTable` and broke startup on a fresh DB.
  - `main.py:227-258` catches both at startup and distinguishes a dimension disagreement from a
    connection failure, with the comment explaining that the old message blamed dimensions for
    *every* startup failure. This is careful work.
- **The pgvector chunk_id column is `UUID PRIMARY KEY`** (`adapters.py:376`) and the chunker
  generates `str(uuid.uuid4())` (`smart_chunker.py:431` et al.). Consistent. ✅
- **No vector index on pgvector** — see §5.3. `ORDER BY embedding <=> $1 LIMIT $2` with no ANN
  index is an exact sequential scan.
- **No payload indexes on Qdrant.** `ensure_collection` creates the collection and nothing else.
  Every `delete_stale_chunks` filter (`doc_id`, `ingest_run_id`) and every `search_children`
  filter (`is_parent`) is unindexed.
- **Upsert ordering is correct**: `db.upsert(chunks)` then `db.delete_stale_chunks(doc_id,
  ingest_run_id)` (`pipeline.py:141-144`). The new run is written before the old run is removed, so
  there is no window where the document is absent. ✅
- **The two backend implementations are not behaviourally identical.** Qdrant stores `payload` as
  a JSON object; pgvector stores it as a JSONB `TEXT` column and re-parses it on every read
  (`adapters.py:422`, `adapters.py:536`). `_row_to_hit` reads `payload.get(...)` on a dict, but
  `retrieve_parents` explicitly re-parses (`if payload is None: payload = {}` then
  `payload.get(...)`) — meaning psycopg returns JSONB as a `dict` on modern psycopg3, but the
  `if payload is None` guard suggests uncertainty about that. The `search` path
  (`adapters.py:422`) does **not** guard against `payload` being a JSON string rather than a dict.
  If a JSONB column is ever returned as text (older psycopg, different loader), `payload.get`
  raises `AttributeError`. **Suspected risk**, not confirmed — I could not run psycopg against a
  live database to confirm the return type in this environment.

### 7.4 Concurrency and thread safety

- The asyncio semaphore bounds concurrent ingestions (`INGEST_CONCURRENCY`, default 4).
- `JobStore` uses a `threading.Lock` plus a single `sqlite3` connection with
  `check_same_thread=False` (`job_store.py:70-76`) — correct for the FastAPI threading model.
- `claim_next` uses `BEGIN IMMEDIATE` (`job_store.py:166`) — the right primitive for cross-process
  atomic claiming.
- **Module-level mutable globals:** `retrieval.py:81-82` holds `DENSE_CONFIDENT_GAP` and
  `DENSE_MIN_SCORE` as module globals that `_refresh_gate_thresholds()` **mutates on every
  retrieval call** (`retrieval.py:85-96`). This is a data race by construction: two threads
  calling `retrieve_and_rerank` concurrently can interleave the global writes, so one thread can
  read a half-updated threshold pair. The docstring says *"It writes to the module globals on
  purpose"* — the tests monkeypatch them. The correct design is to return the thresholds from the
  refresh function and pass them as locals. **P1** — currently latent because the gate is
  unreachable in production (R1), but it becomes live the moment hybrid search is exposed.
- `worker.py` uses module-level `_shutdown` (`worker.py:54`) set from a signal handler — safe
  enough under CPython for a boolean flag.
- `pytesseract.pytesseract.tesseract_cmd` is process-global — S6.

### 7.5 Configuration management

`atrium/config.py` is the strongest module. Notable residual issues:

- **Empty values cannot override non-empty defaults (P2).** `env_str` treats an empty environment
  value as unset (`config.py:266-271`). So `PG_CONN=""` does not mean "no Postgres" — it falls
  back to `DEFAULTS["PG_CONN"] = "postgresql://user:pass@pgvector:5432/db"`
  (`config.py:104`), a DSN with a hardcoded password. An operator who explicitly clears the
  variable gets a connection attempt to a default host with a default password instead of an
  error.
- **`redacted_settings` does not mask a token embedded in a URL (P2).** `config.py:453` masks
  `api_key`, `cohere_api_key`, and `pg_conn`. But `OLLAMA_BASE_URL` is often configured as
  `https://api.openai.com/v1` with a key in a query string, or `https://host/v1?api_key=sk-...`,
  and it is printed verbatim by `atrium doctor` and `atrium configure --show`.
- **`get_settings()` is explicitly not cached** (`config.py:396-402`) — the docstring acknowledges
  this and says "do not call it in a hot loop." It is a dataclass construction per call; callers
  mostly call it once at startup. Fine.

### 7.6 Logging and observability

- Logging is configured with `logging.basicConfig` in `worker.py:31-34` and (per the smoke test
  output) in `main.py`. Format: `%(asctime)s %(levelname)s [%(name)s] %(message)s` — consistent,
  with namespaced loggers (`ingest.main`, `ingest.retrieval`, `ingest.security`, `ingest.worker`).
- **No request IDs, no structured logging, no metrics, no tracing, no correlation IDs.** A
  request that spans the API → the queue → the worker has no shared identifier beyond the
  `job_id`.
- **No metrics export.** There is nothing to scrape for request rate, error rate, or latency
  distribution. The `/health` endpoint (`main.py:312-328`) returns only `status`, `db`,
  `embedder`, `vector_db_type` — a liveness probe, not a metrics endpoint.
- The `/search` response includes `took_ms` (`main.py:477`) — useful, and the one piece of
  per-request observability that exists.
- The confidence-gate routing decision is logged at INFO with the scores and thresholds
  (`retrieval.py:279-285`) — good practice for the experiment it was built for.

### 7.7 API and CLI reliability

- **The CLI and the HTTP API disagree on the rerank tri-state (P2).**
  `/search` implements `rerank` as tri-state: `True` forces it, `False` forces it off, omitted
  follows `RERANK_ENABLED` (`main.py:407-410`), and a forced-on request with no available
  reranker returns **503** rather than silently serving unreranked results — the comment says
  *"Silently returning unreranked results under a `rerank: true` request would be a lie."* That is
  exactly the right call and exactly the behaviour the user's own design principle demands.

  The CLI does not implement this. `cli.py:1036-1038` declares `rerank: bool = typer.Option(False,
  "--rerank", ...)` — a plain boolean defaulting to `False`. **`RERANK_ENABLED=1` is ignored by
  `atrium search`**, and there is no way to express "off" versus "not specified". The two
  surfaces for the same operation have different semantics.
- **No global exception handler.** `grep -n "exception_handler" main.py` → no matches. Starlette's
  default `ServerErrorMiddleware` returns a bare `Internal Server Error` and logs the traceback
  server-side, so there is no client leak — but there is also no consistent error envelope, no
  error code, and no request correlation.
- **The `/ingest` failure path is handled well (P2, positive).** `_run_one_ingestion` inspects the
  job row after the fact and attaches `"error"` to the response, with the comment: *"run_ingestion
  swallows the exception and records it on the job row, so the caller would otherwise get a 200
  with a job_id and no idea the ingest failed. Surface the failure inline."* ✅
- **The `/ingest` response is `200` even when a file failed to ingest** — the failure is a field
  in the response body, not an HTTP status. A caller checking only the status code will believe
  the ingestion succeeded. Defensible, but worth documenting.
- `cli.reingest` defaults `--chunking-mode` to `context_rich` and its help text lists only
  *"balanced, precise, context_rich"* — omitting `auto` and `custom`, both of which
  `chunk_with_mode` supports (`smart_chunker.py:254`).

### 7.8 Backward compatibility

- The v0/v1 corpus schema contract is documented and enforced (`retrieval.py:20-26`).
- `retrieve_with_parent_resolution` is kept as a back-compat alias (`retrieval.py:610-629`).
- `atrium/mime.py` re-exports the private names it absorbed from `main` (`mime.py:14-16`).
- `/ingest_single` is preserved with its 0.x wire format (`main.py:601-616`).
- **`collection_schema_version()` failing closed to v0** means an adapter without that method
  keeps the legacy path — a good default.
- **The gap:** there is no *migration* for a corpus. The README's `atrium reingest` is a
  per-document re-chunk, not a corpus migration. If the corpus is upgraded from v0 to v1
  (parent-child), the author has verified that legacy points lacking `is_parent` are excluded by
  the Qdrant filter — which means they become **invisible** (A2). There is no documented upgrade
  path and no test covering "mixed corpus, v1 detection, legacy points".

### 7.9 Test coverage and integration-test quality

**225 passed, 7 skipped, 5 xfailed.** That is a solid unit suite for a project of this size.

What the suite does well:
- A real end-to-end smoke test against the *installed artifact* (`scripts/smoke_test.py`), wired
  into CI as a dedicated `install` job (`ci.yml:69-92`). **This is the single best packaging
  practice in the repository** — it is the job that catches a missing module or a console script
  that does not land, and it is the reason this audit could verify the wheel rather than infer it.
- A protocol contract test for both backends (`tests/test_adapters_contract.py`, 14 Qdrant +
  2 pgvector assertions per the adapter docstring).
- A dedicated test for the confidence gate (`tests/test_retrieval_confidence_gate.py`).
- A dedicated test for the unified rerank path (`tests/test_rerank_unified.py`).
- A CI job that re-renders the benchmark image and fails if it drifts from the JSON
  (`ci.yml:44-64`). The intent — *"the README starts making a claim the data no longer
  supports"* — is exactly right, and it is a good mechanism. **It did not catch the R@5/R@8
  mislabel, because the label lives in the README prose and the badge alt-text, not in the
  generated image.** The mechanism is sound; the coverage has a gap.

What is missing or weak:
- **7 skips, 6 of which are fixture-dependent** — `dev-example.pdf` (4 tests), an RTF fixture,
  and the entire e2e harness (`tests/test_e2e_harness.py:58`, *"got empty parameter set"*). These
  skip in CI too, because the corpus is gitignored. **The e2e harness never runs in CI.** The
  repository's own comment calls the test suite "the contract," but a contract test that never
  executes is not a contract.
- **1 skip requires a live stack** — `tests/test_v0_suite.py:67` (*"Stack not reachable (status=degraded,
  db=False, embedder=True); /health returns 503 by design"*). Integration behaviour against real
  Qdrant/pgvector/Ollama is therefore **never verified in CI**, for anyone.
- **5 xfailed** — known failures parked as expected. The xfail list was not retrievable in this
  session (the command was blocked), so the specific items are **not characterised in this
  report**. Recommend making them a documented list in the README.
- **No test asserts the `top_k` contract** on the rerank-failure path (R2). No test asserts the
  number of returned hits under failure. This is why the defect is invisible.
- **No test covers the v0→v1 corpus transition** (A2).
- **No test asserts `ensure_collection(name)` actually uses `name`** (A1).
- **No performance or load test of any kind** in the repository.
- **No test runs the benchmark harness** — by design, it is out of the repo.

---

## 8. Open-source readiness assessment

| Criterion | Assessment |
|---|---|
| **Public API understandable?** | Partially. The README's Python example is short and correct. But there is no API reference, no `__all__` on any public submodule, and no statement of what is stable. A contributor cannot tell which of the 24 modules is public. |
| **Installation straightforward?** | ✅ Yes. `pip install atrium-rag` → works, verified in a clean venv. This is the strongest OSS dimension. |
| **Examples reproducible?** | ⚠️ The README's examples need a running Qdrant + Ollama with `bge-large` pulled. They are plausible but untested by this audit. `atrium demo` exists for a self-contained path. The *benchmark* example is not reproducible at all (B6). |
| **Defaults sensible?** | ✅ Mostly. `localhost` over `host.docker.internal`, `bge-large`/`1024` matching the benchmarked configuration, `top_k=8`, 100 MB upload cap, `API_KEY` empty (documented as dev-only with a `doctor` warning). ⚠️ `PG_CONN` default carries a hardcoded password; `env_str` can't clear it. |
| **Errors actionable?** | ⚠️ Mixed. **Good:** the OCR missing-extra error names the exact pip command; the reranker-unavailable 503 names the extra *and* the env var; the dim-mismatch messages name both numbers and both remedies; the PDF-backend error lists available backends; the CLI prints `[bad]`/`[ok]` markers. **Bad:** the two `/search` 503s leak raw exception text (S1); the `pdfplumber` ImportError has no install hint (K5). |
| **Limitations documented?** | ✅ Genuinely good — a real Limitations section covering CJK/Arabic BM25, no distributed queue, advisory mypy, CDN Tailwind. ⚠️ It does **not** mention: the AGPL/PyMuPDF licensing situation, that hybrid BM25 is unreachable from the API, that the `alt-parsers` extra has no backend, that pgvector has no vector index, that the benchmark corpus is not in the repo (this is mentioned under Development, not Limitations), or that the headline metric is R@5. |
| **Contributors can understand and extend?** | ⚠️ The code is well-commented — unusually so, with rationale in the comments rather than just description. But: `universal_parser.py` is 2277 lines with 16 parsers, `main.py` is 1566, `cli.py` is 1304. No contributor guide, no architecture document, no `CONTRIBUTING.md`, no `CODE_OF_CONDUCT.md`, no `SECURITY.md`, no issue/PR templates (`.github/` contains only `workflows/`). The `eval_v2/` harness and its `HANDOVER.md` are not in the repository, so a contributor cannot see how the benchmark was run. |
| **License appropriate and clearly declared?** | ❌ **MIT is declared, but the hard dependency is AGPL-3.0.** This is the blocking issue (K1). The `pdf_backend.py` docstring shows the author is aware of the licence landscape — the reasoning is in the file — but the resolution was never made. |
| **Genuinely usable independently?** | ⚠️ Yes for the core (parse → chunk → embed → store → retrieve) — the wheel is self-contained and the smoke test proves it. No for the benchmark, no for hybrid retrieval, no for the `alt-parsers` extra. |

**Missing standard OSS files** (all absent from the repository root): `CONTRIBUTING.md`,
`CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md` (deliberately dropped in commit `8747275` in
favour of GitHub Releases — a defensible choice), `ISSUE_TEMPLATE/`, `PULL_REQUEST_TEMPLATE.md`.

**One notable positive:** `.env` is gitignored and untracked — verified. `.env.example` and
`.env.docker` are tracked and contain no secrets. `redacted_settings` and `secret_keys()` exist
specifically to keep tokens out of `doctor` output. This is careful work.

---

## 9. Consolidated findings

### P0 — Release blockers

| # | Finding | Location | Evidence | Impact | Fix |
|---|---|---|---|---|---|
| **P0-1** | AGPL-3.0 hard dependency in an MIT package | `pyproject.toml:57`; `LICENSE:1`; `atrium/pdf_backend.py:25` | Installed `pymupdf-2.0.dist-info/METADATA`: `License: Dual Licensed - GNU AFFERO GPL 3.0 or Artifex Commercial License`. Repo `LICENSE` is MIT. | The package may not be distributable to PyPI under MIT. | Obtain an Artifex commercial licence **or** make `pdfplumber` the default backend and move `pymupdf` to an optional extra. Re-license or re-scope accordingly. **Legal decision required.** |
| **P0-2** | README headline "mean R@8 = 0.8976" is a Recall@5 | `README.md:9`; `docs/benchmark/v0_k8.json` | Artifact has `recall_at_5`/`mean_recall_at_5`/`ndcg_at_5` only — **no `recall_at_8`**. Macro mean of `mean_recall_at_5` = **exactly 0.8976**. | The single most prominent quantitative claim in the project is wrong. Anyone checking it finds a discrepancy and stops trusting the rest. | Relabel to "**Recall@5 = 0.8976** (top_k=8 retrieval)". Add `recall_at_8` to the harness and report the true R@8 separately. Add a CI check that parses the README badge and compares it to the artifact's metric names. |
| **P0-3** | The headline benchmark is not reproducible | `.gitignore:128-130`; sdist contents | `/tests/benchmarks/fixtures/` and `/tests/parsing_fixtures/` are gitignored; `eval_v2/` is absent from the sdist; the README concedes *"That corpus is not in the repository"*. | No external developer can verify 0.8976, produce a new number, or file an accuracy bug. The claim is unfalsifiable. | Ship a **synthetic, redistributable corpus** (generated at build/CI time from a documented generator) plus the harness in the repo. Publish the raw per-query results in the repo. Consider a hosted CI job that re-runs the benchmark nightly and updates a results badge. |
| **P0-4** | The benchmark overstates retrieval quality by construction | `docs/benchmark/v0_k8.json` (`metric_level`, `n_chunks`) | `metric_level: "page"` for 100/100 queries. 6/19 fixtures have ≤6 chunks and score exactly 1.0000 (one has **2**); the other 13 average **0.8504**. `evidence_in_context` non-null for only 6/100. | "0.8976" overstates what was measured. A chunk from the right *page* counts as a hit regardless of whether it contains the answer. | Report at passage level, or rename the metric honestly. Publish the split (tiny vs. real corpora). Add `evidence_in_context` verification for all queries. Report MRR and nDCG alongside recall. |
| **P0-5** | The reranker model is reloaded on every `/search` request | `atrium/main.py:414`; `atrium/rerank_adapter.py:148-164, 544` | `build_reranker(force=True)` per request → new `LocalBgeReranker` → `_ensure_loaded()` caches per-instance only. | Every reranked query reloads hundreds of MB of weights. Reranking — the feature the benchmark shows is *needed* (3 of the 3 retrieval failures are near-misses with top-1 similarity 0.74/0.76) — is unusable in production. | Module-level singleton / `lru_cache` keyed by `(backend, model, device, max_length)`. |

### P1 — Must fix before a public beta

| # | Finding | Location | Impact |
|---|---|---|---|
| P1-1 | Hybrid BM25 unreachable from the public API; `DENSE_CONFIDENT_GAP`/`DENSE_MIN_SCORE` are production no-ops | `main.py:439-448`; `main.py:1063-1067`; `retrieval.py:143-144`; `__init__.py:8` | An advertised feature and two env knobs do nothing in production. |
| P1-2 | `top_k` violated on rerank failure (v0 returns `candidate_k` hits, default 30) | `retrieval.py:494-501`; `main.py:322-323` | Client asking for 8 gets 30; response says `"rerank": false` while a reranker object exists. |
| P1-3 | `ensure_collection(name, dim)` honors `name` only for creation | `adapters.py:189` vs `198, 206, 253, 315` | Wrong collection written; dimension guard protects the wrong collection. |
| P1-4 | `collection_schema_version()` full scan per query **and** flips the retrieval algorithm based on corpus data | `retrieval.py:222`; `adapters.py:252-258, 439-441` | Per-query scan cost; legacy flat points silently vanish from search once one v1 document is ingested. |
| P1-5 | `alt-parsers` extra installs `docling-slim` (~2 GB) with no Docling backend in the package | `pyproject.toml:110-112`; `grep -rn docling atrium/*.py` → 0 hits | Users pay 2 GB for nothing; README states a falsehood. |
| P1-6 | `pdfplumber` backend selectable but undeclared and undocumented | `pdf_backend.py:651-654`; `pyproject.toml` deps | `ImportError` with no install hint. |
| P1-7 | `atrium.__init__` mutates process-wide `os.environ` at import | `atrium/__init__.py:33-38` | Library pollutes the host application's environment; behaviour depends on the caller's CWD. |
| P1-8 | Raw exception text returned to API clients | `main.py:436, 451` | Leaks DSNs, hostnames, internal errors. `redacted_traceback` exists and is used elsewhere. |
| P1-9 | `WORKER_JOB_TIMEOUT` is a config knob that does nothing | `worker.py:180-181`; `config.py:146` | Operators believe a timeout is enforced. |
| P1-10 | Stale-claim recovery can double-process a long job; original worker then marks it complete | `job_store.py:197-216`; `worker.py:219-221` | Contradicts the README's "without processing a document twice" claim for documents over `JOB_STALE_SECONDS`. |
| P1-11 | `BM25Index.add_doc` recomputes `_avgdl` over all docs → O(N²) index build | `bm25.py:129` | ~72M redundant ops per 12k-chunk rebuild. |
| P1-12 | pgvector upsert issues one INSERT per chunk; `batch_size` ignored | `adapters.py:385-393` | 12,000 round trips per document. |
| P1-13 | No vector index on pgvector (`ORDER BY embedding <=> $1` with no ANN index) | `adapters.py:371-382` | Exact sequential scan per query on the pgvector backend. |
| P1-14 | No Qdrant payload indexes for `doc_id`, `ingest_run_id`, `is_parent`, `schema_version` | `adapters.py:181-189` | Every filter is unindexed. |
| P1-15 | `MANIFEST.in` `exclude tests` does not exclude test files from the sdist | `MANIFEST.in:35`; sdist listing | Ships 17 test files contrary to stated intent. |
| P1-16 | `dev` extra lacks `build` and `twine`; README's Extras table claims both | `pyproject.toml:130-135`; `README.md` Extras table; `Makefile:38-40` | `make build` fails after a clean `make install`. |
| P1-17 | No retry on the embedder | `adapters.py:156-162` (contrast `191, 200, 384, 395`) | A transient Ollama 502 aborts a whole ingestion. |
| P1-18 | Module-level threshold globals mutated on every retrieval call | `retrieval.py:81-96` | Data race between concurrent requests; live the moment hybrid search ships. |
| P1-19 | CLI `rerank` is a plain bool; ignores `RERANK_ENABLED` | `cli.py:1036-1038` vs `main.py:407-410` | Two surfaces of the same operation have different semantics. |
| P1-20 | Benchmark exercises neither reranking nor parent-child | `docs/benchmark/v0_k8.json` (`rerank_latency_ms` 0/100, `parent_resolved` 0/100) | Two of three retrieval features are unbenchmarked. |

### P2 — Should fix

| # | Finding | Location |
|---|---|---|
| P2-1 | `score` semantics change between cosine and RRF in the hybrid path | `retrieval.py:469` |
| P2-2 | Rerank failures swallowed with no logging | `retrieval.py:494` |
| P2-3 | BM25 search is a linear scan; docstring claims postings-touch complexity | `bm25.py:163` vs `bm25.py:78-81` |
| P2-4 | `/docs`, `/openapi.json`, `/redoc` open when `API_KEY` is set | `security.py:64` |
| P2-5 | Job errors stored and served raw | `pipeline.py:177`; `job_store.py:264` |
| P2-6 | `JobStore.update` interpolates column names from `**kwargs` | `job_store.py:228` |
| P2-7 | `signal.signal` installed at import time in the worker | `worker.py:63-64` |
| P2-8 | `TesseractOCRAdapter` mutates the global `pytesseract.tesseract_cmd` | `ocr_adapter.py:65-71` |
| P2-9 | `env_str` treats empty as unset → `PG_CONN=""` falls back to a default DSN with a password | `config.py:266-271, 104` |
| P2-10 | `redacted_settings` does not mask a token embedded in `OLLAMA_BASE_URL` | `config.py:453` |
| P2-11 | `requires-python` has no upper bound; CI has never tested 3.13/3.14 while deps jumped majors | `pyproject.toml:41`; `ci.yml:56` |
| P2-12 | No global exception handler / no consistent error envelope / no request IDs | `main.py` (no `@app.exception_handler`) |
| P2-13 | No metrics, no structured logging, no tracing | whole package |
| P2-14 | `pdf_backend.py` docstring contradicts itself on the default backend | `pdf_backend.py:32` vs `43`, `679` |
| P2-15 | README CLI table omits `atrium version` | `README.md` CLI table |
| P2-16 | README Limitations omits the licensing, hybrid, extra, and pgvector-index issues | `README.md` Limitations |
| P2-17 | `sdk/client.py` — no HTTP timeout, infinite polling loop, `doc_id: str = None`, bare `Exception`; unpackaged and undocumented | `sdk/client.py:8-27` |
| P2-18 | `universal_parser.py` (2277), `main.py` (1566), `cli.py` (1304) are too large | — |
| P2-19 | Runtime construction duplicated across `main.py` / `worker.py` / `cli.py` | `worker.py:69-130`; `main.py:180-210`; `cli.py:884` |
| P2-20 | Typing style mixed: `typing.List` vs PEP 604 across files | `models.py` vs `main.py:73`, `ocr_adapter.py:202` |
| P2-21 | `cli.reingest --chunking-mode` help omits `auto` and `custom` | `cli.py:1104-1108` |
| P2-22 | `_ASYNC_JOB_MAX_BYTES` hardcoded, not configurable; binary base64 in a SQLite TEXT column | `main.py:635`; `job_store.py:154` |
| P2-23 | No `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue/PR templates | repo root |
| P2-24 | `eval_v2/` is unlinted and excluded from the distribution | `Makefile:34` |

---

## 10. Prioritized improvement plan

### Before any public announcement (P0)

1. **Resolve the PyMuPDF licensing.** Decide between an Artifex commercial licence and making
   `pdfplumber` the default with `pymupdf` behind an extra. The `pdf_backend.py` docstring has
   already made the engineering argument for the second option; the licensing follows.
2. **Correct the headline metric.** Relabel to "Recall@5 = 0.8976" and add a CI check that parses
   the README badge against the artifact's field names. Add `recall_at_8` to the harness.
3. **Make the benchmark reproducible.** Generate a redistributable synthetic corpus in-repo;
   publish the raw per-query results; ship the harness. Until then, the headline number should
   carry a visible "not independently reproducible" note.
4. **Publish the honest breakdown.** Report recall-at-page-level as such, split the tiny-corpus
   fixtures from the real ones (1.0000 vs 0.8504), and add MRR (0.7058) and nDCG@5 (0.7381).
5. **Cache the reranker** behind a process-wide singleton.

### Before the public beta (P1)

6. Wire hybrid BM25 into `/search` behind an explicit flag, or delete it from the package and
   the `__init__` docstring. Either way, make `DENSE_CONFIDENT_GAP`/`DENSE_MIN_SCORE` real or
   remove them from `DEFAULTS`.
7. Fix the `top_k` contract in `_apply_rerank`'s fallback, and log the swallowed exception.
8. Make `ensure_collection` honor `name` end-to-end, or delete the parameter.
9. Replace `collection_schema_version()`'s per-query scan with cached collection metadata, and
   add an explicit corpus-migration path for v0 → v1.
10. Delete the `alt-parsers` extra or implement the Docling backend it promises. Declare
    `pdfplumber` in an extra and document it in the README.
11. Make the `.env` load opt-in, or load it into a private mapping rather than `os.environ`.
12. Replace the raw-exception 503s with generic messages; log the detail server-side.
13. Delete `WORKER_JOB_TIMEOUT` or implement the watchdog. Add WAL + `busy_timeout` to the job
    store, and a heartbeat so a live worker's job is never reclaimed.
14. Batch the pgvector upsert with `executemany`; add IVFFlat/HNSW indexes on both backends;
    add Qdrant payload indexes.
15. Fix the O(N²) `_avgdl` recomputation.
16. Add `build` and `twine` to the `dev` extra; fix `MANIFEST.in` to `recursive-exclude tests *`.
17. Add a retry to the embedder.
18. Return the gate thresholds as locals instead of mutating module globals.
19. Make the CLI `rerank` flag tri-state, matching the API.
20. Add a benchmark configuration that exercises the reranker and the parent-child path, or state
    plainly in the README that the published number covers the dense-only flat path only.

### Quality pass (P2)

21. Add request IDs, structured logging, and a `/metrics` endpoint.
22. Split `universal_parser.py` into a package (`atrium/parsers/`), and `main.py` into routers.
23. Add `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and issue/PR templates.
24. Extend the CI matrix to 3.13/3.14 and add an upper bound or a tested-version policy for
    `requires-python`.
25. Add an architecture overview to the repo so contributors can find their way.

### Explicitly **not** recommended

- **Do not rewrite the retrieval core.** The RRF implementation, the confidence gate, the BM25
  math, the parent-child key contract, and the schema-version derivation are all sound. The
  problems are in wiring, defaults, and evidence — not in the algorithms. A rewrite would discard
  real quality.
- **Do not replace the adapter protocols.** The `Protocol`-based seams are the right design.
- **Do not add a distributed job queue.** The README already states the SQLite limitation
  honestly. That is the correct trade at this stage.
- **Do not add a compatibility layer for `qdrant-client`'s old `search()` name.** The existing
  `_qdrant_search` shim already handles both spellings, and the `>=1.11.0` floor plus the shim
  covers the transition.

---

## 11. Scores

| Dimension | Score | Rationale |
|---|---|---|
| **Retrieval quality** | **5 / 10** | The mechanism is good — RRF is textbook-correct, the confidence gate handles its edge cases deliberately, the parent-child key contract is verified consistent, and the BM25 math is right. But the *demonstrated* quality is weak: the headline metric is mislabeled (P0-2), recall is page-level (P0-4), a third of the fixtures saturate (P0-4), two of three retrieval features are unbenchmarked (P1-20), and MRR sits at 0.71. Mechanism 7, evidence 3. |
| **Performance** | **4 / 10** | Good fundamentals — `asyncio.to_thread` used correctly throughout, lazy imports with measured rationale, bounded concurrency, batched embedding. But a per-request model reload (P0-5), a full-collection scan per query (P1-4), O(N²) BM25 builds (P1-11), per-row pgvector inserts (P1-12), and no vector indexes on either backend (P1-13, P1-14). No throughput benchmark exists in the repo. |
| **Reliability** | **5 / 10** | Atomic job claiming via `BEGIN IMMEDIATE` is correct, the dimension guard on both backends is careful and actionable, and the whole test suite passes. But there is no retry on the most failure-prone dependency (P1-17), a timeout knob that does nothing (P1-9), stale-claim double-processing (P1-10), a `top_k` contract violation on failure (P1-2), and a global-env side effect at import (P1-7). |
| **Architecture** | **6 / 10** | Clean module boundaries, correct `Protocol` seams, one authoritative config resolver, explicit schema-version contract, per-chunk provenance. Undercut by a lying `ensure_collection` parameter (P1-3), retrieval semantics derived from corpus *data* rather than configuration (P1-4), three over-large files, and duplicated runtime construction across three entry points. |
| **Python packaging** | **7 / 10** | Builds cleanly, `twine` passes, wheel and sdist both install into a clean venv, `py.typed` ships, the console script works, metadata is complete and PEP 621 compliant, `pip check` is clean, the secret-handling story is careful, and the PyPI name collision was spotted and handled well. Undercut by the AGPL dependency (P0-1), the dead `alt-parsers` extra (P1-5), the undeclared `pdfplumber` (P1-6), the broken `MANIFEST.in` (P1-15), and the wrong extras table (P1-16). **This is the most polished dimension.** |
| **Developer experience** | **6 / 10** | The CLI is well-designed, the error messages are frequently and specifically actionable, the docstrings carry rationale rather than restating the code, and the installed-artifact smoke test is exemplary. Undercut by README inaccuracies in three separate places, no API reference, no `__all__` on public modules, and no architecture guide for a 2,277-line parser. |
| **Production readiness** | **4 / 10** | Security fundamentals are sound (constant-time auth, path-traversal protection, MIME conflict detection, no SSRF, no secret leakage) and the dim-mismatch handling is genuinely good. But: no migrations, no vector indexes, a single-file queue with no retry policy, no request IDs or metrics, raw exception leakage to clients, a global-environment side effect at import, and per-request model loading. |
| **Open-source readiness** | **3 / 10** | The *installation* story is excellent. Everything around it is not: an unreproducible headline metric (P0-3, P0-4), an AGPL/MIT conflict (P0-1), a false README instruction about a 2 GB extra (P1-5), missing `CONTRIBUTING`/`SECURITY`/`CODE_OF_CONDUCT`/templates, a benchmark harness that is not in the repository, and an `sdk/` directory that is neither packaged nor documented. **This is the dimension that gates the launch.** |

---

## 12. Final verdict

### **GO WITH LIMITATIONS** — for a private / invited release.
### **NO-GO** — for a public PyPI + GitHub launch today.

### The short version

Atrium's **code** is good enough to build a product on. The **packaging** is good enough to
publish. The **claims** are not good enough to publish.

The specific failure is that the project's most visible number — the badge at the top of the
README — is wrong in a way that a careful reader will catch in five minutes, and the evidence
underneath it is unreproducible and constructed in a way that flatters the result. A benchmark
that says "Recall@5 = 0.8976" and is measured at page level on a corpus where six of nineteen
fixtures contain two to six chunks is not a weakness in the retrieval engine; it is a weakness in
the evidence. And that matters more here than it would in most projects, because the accuracy
claim is the product.

### Can an independent developer install it, understand it, trust it, and use it?

- **Install it?** ✅ **Yes.** Verified end-to-end: build → wheel → clean venv → console script →
  smoke test → sdist → clean venv → smoke test. The best part of the project.
- **Understand it?** ⚠️ **Mostly.** The README is well-written and the code is unusually well
  commented. But there is no API reference, no stated stable surface, and three README statements
  that are simply wrong (the R@8 label, the `alt-parsers` extra, the `dev` extras list).
- **Trust it?** ⚠️ **Not yet.** Trust requires verifiable claims. The headline metric is
  mislabeled, the benchmark cannot be re-run, the corpus is gitignored, the "19/19 fixtures
  passed" framing hides that a third of those fixtures have ≤6 chunks, and the package ships AGPL
  code under an MIT license.
- **Use it successfully in a real RAG application?** ⚠️ **Yes for the core loop, with caveats.**
  Parse → chunk → embed → store → retrieve works, the chunker is good, the parent-child design is
  sound, and the CLI is pleasant. But: hybrid search is unreachable (P1-1), reranking reloads its
  model per request (P0-5), the pgvector backend has no vector index (P1-13), the job queue has
  no retry policy (P1-17), and a legacy corpus can silently lose documents to search after a
  schema change (P1-4).

### The recommendation

**Do not publish this week.** The gap between "today" and "publishable" is small and entirely
concrete — it is roughly:

1. a licensing decision (P0-1);
2. one corrected number and an honest breakdown (P0-2, P0-4);
3. a redistributable synthetic corpus and the harness in-repo (P0-3);
4. one `@lru_cache` (P0-5);
5. the eleven P1 items above, most of which are one-to-five-line changes.

That is a focused week's work for the code items plus a legal consultation for the licensing
item. Once those land, Atrium is a genuinely publishable open-source RAG ingestion package with
a real benchmark and a real differentiation — the structure-aware chunker with heading
propagation, the parent-child schema with a documented v0→v1 contract, the swappable PDF backend
seam, the protocol-based adapter design, and the discipline of one config resolver are all
better than most things in this category.

What is *not* needed: a rewrite of the retrieval engine, a distributed job queue, or a broader
API surface. The engine is good. The evidence and the packaging need to catch up to it.

---

*Report generated 2026-10-30. All findings are either confirmed by executed commands (execution
log in §2.1) or by direct source reading with file:line citations. No accuracy figure in this
report is estimated, extrapolated, or taken from prose — every benchmark number was recomputed
from `docs/benchmark/v0_k8.json`. Where a finding is a suspected risk rather than a confirmed
defect, it is labelled as such (§7.3, P2-1).*
