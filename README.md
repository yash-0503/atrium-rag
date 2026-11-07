# Atrium

**Put a folder of PDFs, Word documents and spreadsheets somewhere, and ask it questions in plain language.**

Atrium is a document-ingestion library and HTTP service. It parses a
document into structure, cuts it into chunks that keep that structure
attached, embeds those chunks, and stores them in a vector database — so
a later question can find the paragraph that answers it, not just the
document that contains it.

```bash
pip install atrium-rag
```

![atrium benchmark: R@5 0.700 on 23 documents — Docling 0.686, Unstructured 0.614](docs/images/benchmark.png)

- **Python:** 3.10+
- **Vector DB:** SQLite (default), Qdrant or pgvector
- **Embeddings:** any OpenAI-compatible `/v1/embeddings` endpoint (Ollama, OpenAI, vLLM, LM Studio, TEI) — or none, with the in-process embedder
- **License:** MIT

---

## Why

The usual way to make documents searchable is to assemble four
libraries yourself: a parser, a chunker, an embedder and a vector
store, then write the glue that keeps their document IDs in agreement.
The parsers are heavy — Docling pulls in a model runtime and a few
gigabytes of dependencies — and most of them want a GPU or a paid API
key before they will read a page.

Atrium takes the opposite position. The parsing is pure Python and
heuristic: no model, no API key, no GPU, and no licence obligations
beyond MIT. It reads the document's own structure — headings, tables,
list items — and carries that structure into the chunk, so a sentence
pulled out of the middle of a section still knows where it came from.

The cost of that choice is measured, not asserted. On the same 23
documents, with each system using its own parser and chunker and the
same embedding model:

| system | R@5 | R@10 | MRR | parse time | peak RSS |
|---|---|---|---|---|---|
| **atrium** | **0.700** | **0.769** | **0.574** | 88 s | 1016 MB |
| docling | 0.686 | 0.726 | 0.540 | 554 s | 3482 MB |
| unstructured | 0.614 | 0.688 | 0.486 | 195 s | 855 MB |

Retrieval accuracy is on par with Docling at this corpus size — the
difference is inside the confidence interval. The separation is in
cost: atrium parses the corpus in a sixth of the time and a third of
the memory.

*(The harness, the corpus and the exact commands are published
separately in the [`atrium-benchmarks`](https://github.com/yash-0503/atrium-benchmarks)
repository, so the numbers can be reproduced. The comparison is
described in more detail under [Evaluation](#evaluation).)*

---

## Quickstart

### Set it up

```bash
pip install atrium-rag
atrium init             # checks your machine, writes .env, verifies it works
```

`atrium init` looks at what is installed, asks which vector store and
which embedding provider you want, offers to pull a model or start a
container, shows you the configuration before it is written, and then
checks it by making a real embedding call. Everything it offers to do,
it asks first.

Nothing else is required. With no server running, `atrium init` can
point you at the in-process embedder — pick **in-process** when it
asks, and it offers to install the extra itself. The default vector
store is a SQLite file next to your project.

### Ingest and search over HTTP

```bash
atrium serve            # http://localhost:8000, docs at /docs
```

```bash
curl -X POST http://localhost:8000/ingest \
    -F "files=@./report.pdf"

curl -X POST http://localhost:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"query": "What is the retention policy?", "top_k": 8}'
```

One file is one entry in `files`; several files in one request are
ingested concurrently and each gets its own record back.

### From the shell

```bash
atrium ingest ./docs/
atrium search "What is the retention policy?"
atrium reingest ./docs/policy.md --doc-id policy --chunking-mode context_rich
```

### From Python

```python
from atrium.pipeline import ingest_document
from atrium.adapters import QdrantAdapter
from atrium.embedder_config import build_embedder
from atrium.universal_parser import UniversalParser
from atrium.retrieval import retrieve_and_rerank

db = QdrantAdapter(host="localhost", port=6333)
embedder = build_embedder()
db.ensure_collection("documents", embedder.dimension)

ingest_document(
    doc_id="report",
    file_bytes=open("report.pdf", "rb").read(),
    filename="report.pdf",
    mime_type="application/pdf",
    parser=UniversalParser(),
    embedder=embedder,
    db=db,
)

hits = retrieve_and_rerank(db, embedder.embed(["retention policy"])[0],
                            "retention policy", top_k=8)
for hit in hits:
    print(hit["score"], hit["doc_id"], hit["text"][:80])
```

---

## The API

Five routes carry the contract. The rest are for looking at what the
parser produced, and are not part of it.

| Endpoint | What it does |
|---|---|
| `POST /ingest` | Ingest files inline — parsed, embedded and stored before the response |
| `POST /jobs` | Queue a file for the worker; returns `202` + `job_id` |
| `GET /jobs/{job_id}` | Job status: `pending`, `running`, `complete`, `failed` |
| `POST /search` | Search the corpus |
| `GET /health` | Liveness + dependency probe |

`POST /preview` sits with them: it parses and chunks without embedding
or storing anything, which is how you check what the parser made of a
document before committing it to the corpus.

### `POST /search`

```bash
curl -X POST http://localhost:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"query": "retention policy", "top_k": 8, "rerank": false}'
```

Only `query` is required. `top_k` falls back to `DEFAULT_TOP_K`;
`rerank` is a tri-state — `true` forces it on, `false` forces it off,
omitted follows `RERANK_ENABLED`.

```json
{
  "query": "retention policy",
  "top_k": 8,
  "rerank": false,
  "count": 8,
  "corpus_chunks": 1451,
  "took_ms": 41,
  "results": [
    {
      "rank": 1, "score": 0.78, "chunk_id": "…", "doc_id": "policy",
      "text": "…", "page_number": 3, "section_path": ["Retention"],
      "element_type": "paragraph", "chunking_strategy": "structure_aware",
      "is_parent": false, "parent_id": null
    }
  ]
}
```

### `POST /jobs`

Queue a document for asynchronous ingestion. The response is
immediate; a worker does the work.

```bash
curl -X POST http://localhost:8000/jobs -F "file=@./big-report.pdf"
```

Bodies over 50 MB get `413`; a file whose declared content type
contradicts its bytes gets `415`.

### Chunking modes

`auto` (default), `balanced`, `precise`, `context_rich`, or `custom`
with an explicit config:

```bash
curl -X POST http://localhost:8000/ingest \
    -F "files=@./doc.pdf" \
    -F "chunking_mode=custom" \
    -F 'chunking_config={"target_chars": 900}'
```

`auto` resolves to the `precise` strategy — structure-aware chunking
at 700 characters with 100 characters of overlap — and adjusts the
parameters to the shape of the document. `context_rich` adds a parent
chunk over each section. The strategy that actually ran is stamped on
every chunk, so a corpus with mixed strategies stays auditable.

---

## The playground

Two pages ship with the service. They are for looking at the parser's
output, and they are the fastest way to find out what a document
actually contains.

![the chunk visualiser: a document dropped in, the chunks it produced](docs/images/visualize.gif)

| Route | What it is |
|---|---|---|
| `GET /visualize` | Drop a document in; see the chunks it produced, with their headings, pages and sizes |
| `GET /retrieval_lab` | Load one document into an in-memory sandbox and query it without touching the corpus |

The clip above is NIST SP 800-61r2, parsed by the default backend:
398 chunks, each one with its page, its element type and a content
hash.

`/visualize` loads Tailwind from a CDN, so it needs a network
connection; there is no offline build.

---

## Configuration

Every setting is an environment variable. Put them in `.env` (run
`atrium init` to generate one) or export them. A real environment
variable beats `.env`, and `.env` beats the built-in default. The full
list, with comments, is in `.env.example`.

| Variable | Default | What it does |
|---|---|---|
| `VECTOR_DB` | `sqlite` | `sqlite`, `qdrant` or `pgvector` |
| `ATRIUM_BIND` | `127.0.0.1` | The address `atrium serve` binds |
| `LOCAL_DB_PATH` | `atrium_local.db` | Where the local store keeps its file |
| `QDRANT_HOST` / `QDRANT_PORT` | `localhost` / `6333` | Qdrant address |
| `PG_CONN` | — | Postgres DSN, when `VECTOR_DB=pgvector` |
| `EMBED_BASE_URL` | `http://localhost:11434/v1` | Any OpenAI-compatible `/v1` endpoint |
| `EMBED_API_KEY` | — | Key for that endpoint; blank for local servers |
| `EMBED_PROVIDER` | `http` | `http`, or `fastembed` for the in-process embedder |
| `EMBED_MODEL` | `bge-large` | Model name, as that endpoint knows it |
| `EMBED_DIM` | `1024` | Must match the model's dimension |
| `DEFAULT_TOP_K` | `8` | Results returned by `/search` |
| `OCR_ENABLED` | `0` | OCR for scanned pages |
| `OCR_PROVIDER` | `tesseract` | `tesseract` or `rapidocr` |
| `OCR_LANG` | `eng` | Language code |
| `RERANK_ENABLED` | `0` | Cross-encoder reranking |
| `RERANK_PROVIDER` | `bge` | `bge`, `flashrank` or `cohere` |
| `RERANK_MODEL` | provider default | Cross-encoder id |
| `VISION_ENABLED` | `0` | Caption images with a vision model |
| `ATRIUM_PDF_BACKEND` | `pdfplumber` | PDF library: `pdfplumber` or `pymupdf` |
| `API_KEY` | — | Bearer token; empty = no auth (dev only) |
| `ALLOWED_MIME_TYPES` | — | Comma-separated allowlist; empty = any |

Reranking is off by default: it improves ranking on most corpora but
regresses on forms, slides, math and OCR'd scans, because the
cross-encoder was trained on web QA.

The service binds `127.0.0.1` by default. `API_KEY` empty means every
endpoint is open, so set it for anything that is not a laptop —
`atrium doctor` tells you if you have not.

---

## Vector stores

| `VECTOR_DB` | What it needs | When to use it |
|---|---|---|
| `sqlite` | nothing — a file next to your project | the default; a personal or team corpus |
| `qdrant` | a running Qdrant | the corpus outgrows a file on disk |
| `pgvector` | Postgres with the `pgvector` extension | you already run Postgres, and want one database |

The local store is a SQLite file plus a cosine search over it — the
standard library and nothing else, no server, no Docker, no container.
It is the default so that a fresh install works before you install
anything else.

Its search scans every stored vector, so its cost grows with the
corpus: about 5 ms at a thousand chunks, about 30 ms at five thousand,
about 300 ms at twenty thousand. It is the right store for a personal
or team corpus; past roughly twenty thousand chunks, move to Qdrant or
pgvector. `atrium doctor` says so when the corpus gets there.

The embedding provider is configured the same way whatever the store:
any server that speaks the OpenAI `/v1/embeddings` API (Ollama, OpenAI,
vLLM, LM Studio, TEI, Together), or the in-process ONNX embedder from
the `local` extra, which needs no server and no key.

---

## CLI

| Command | What it does |
|---|---|
| `atrium init` | Set the machine up: detect, configure, offer to fix what it can, verify |
| `atrium configure` | Set one key: `atrium configure --key EMBED_DIM --value 768` |
| `atrium validate` | Check `.env` against the installed runtime |
| `atrium doctor` | Probe the stack: embedder, vector DB, security posture |
| `atrium ingest` | Ingest files or directories |
| `atrium search` | Query the corpus |
| `atrium reingest` | Re-chunk an existing document |
| `atrium demo` | Ingest one document and search it back |
| `atrium serve` | Run the API |
| `atrium version` | Print the version |

---

## Worker

`POST /jobs` needs a worker to drain the queue:

```bash
python -m atrium.worker
```

Workers claim jobs atomically, so several can share one queue file
without processing a document twice. The queue is a single SQLite
file (`JOB_DB_PATH`); to run workers on other hosts, put it on a
shared volume.

---

## Docker

```bash
docker build -t atrium-rag .
docker run -p 8000:8000 --env-file .env atrium-rag

# or the whole stack (API + worker + Qdrant)
docker compose --profile qdrant up -d
```

The image runs as a non-root user and carries a health check. Build
with extras via `--build-arg EXTRAS="ocr,rerank"`.

The Compose stack needs an embedding endpoint on the host — it points
`EMBED_BASE_URL` at `host.docker.internal:11434`, which is where Ollama
listens by default. Run `ollama serve` and `ollama pull bge-large`
first, or point `.env.docker` at whatever endpoint you use.

---

## Supported formats

| Format | Parser |
|---|---|
| PDF | pdfplumber / pdfminer.six, layout-aware heuristics (MIT) |
| DOCX | `python-docx` |
| PPTX | `python-pptx` |
| XLSX / XLS | `openpyxl` / `xlrd` |
| HTML | `beautifulsoup4` |
| ODT / ODS / ODP | `odfpy` |
| EPUB | built-in |
| RTF | `striprtf` |
| Markdown, TXT, CSV, JSON | built-in |
| JPG / PNG / WebP | vision adapter (needs Ollama + a vision model) |

The PDF library is a configuration choice, not a dependency of the
architecture: `atrium/pdf_backend.py` defines a seam over the seven
primitives the parser needs, and `ATRIUM_PDF_BACKEND` selects the
implementation. `pdfplumber` is the default; `pymupdf` is faster
(pdfminer.six is pure Python) and is installed through an extra —
see [License](#license).

Install `atrium-rag[alt-parsers]` for Docling as an alternative PDF
parser.

---

## Extras

| Extra | Adds |
|---|---|
| `ocr` | `pytesseract`, `rapidocr-onnxruntime` |
| `rerank` | `sentence-transformers` |
| `flashrank` | `flashrank` — ~30MB ONNX reranker |
| `alt-parsers` | `docling-slim` |
| `local` | `fastembed` — the in-process ONNX embedder, no server, no key |
| `fast-pdf` | `pymupdf` — the faster PDF backend (AGPL-3.0) |
| `vision` | no packages; the adapter talks to Ollama over httpx |
| `dev` | `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pymupdf` |

The `local` extra downloads its model on first use (about 130 MB, cached
afterwards) — models are tens of megabytes and belong on the machine,
not in a wheel.

OCR with `OCR_PROVIDER=tesseract` also needs the Tesseract binary on
`PATH` (`brew install tesseract`, `apt install tesseract-ocr`).
`OCR_PROVIDER=rapidocr` is pip-only and needs no binary.

---

## Development

```bash
pip install -e ".[dev]"

pytest tests/ -q
ruff check atrium tests scripts
mypy atrium            # advisory for now — see Limitations
make build             # wheel + sdist, then twine check
```

The synthetic parsing fixtures under `tests/parsing_fixtures/` are in
the repository. A few tests also parse real third-party documents
(a W-2 form, a 10-K); that corpus is not in this repository, so those
tests skip rather than pass quietly.

---

## Evaluation

Retrieval quality is measured against Docling and Unstructured on a
corpus of 29 documents — academic papers, government reports, forms,
scanned pages, CJK and Arabic — with 700 questions across five
difficulty tiers. Each system parses and chunks with its own pipeline
and embeds with the same model, so the comparison isolates the part
that is usually bought off the shelf.

The numbers are in [Why](#why), above. Two caveats belong with them:

- The hard, implicit and unanswerable tiers are machine-drafted and
  queued for human sign-off; `python -m eval_v2.review_cli status`
  reports how many are still open, and the harness prints that count on
  every run. The published numbers come from a run in which those
  questions were still open, so read them as provisional.
- Two document types are known weak spots: two-column academic papers
  and government forms. The harness reports per-document coverage, so
  the gap is visible rather than averaged away.

The harness, the corpus and the exact commands are published in the
`atrium-benchmarks` repository, so the numbers can be reproduced. See
`docs/benchmark/final_result.md` there for the intervals, the per-tier
breakdown and the failure attribution.

---

## Limitations

- **The sparse ranker is not reachable.** BM25 and the confidence gate
  that fuses it are in the library, but no endpoint or command passes
  them a sparse index, so retrieval is dense (+ an optional rerank).
- **OCR on a noisy scan is weak.** The OCR path returns an empty string
  for a page it cannot read; the count of unreadable pages is reported
  on the job row, but the text is not recovered.
- **CJK / Arabic BM25** tokenization is not script-aware yet.
- **No distributed job queue** — workers share a SQLite file. There is
  no Redis / Kafka / Celery and no Postgres job store.
- **The local store is a file, not a search engine.** Its query scans
  every stored vector; past roughly twenty thousand chunks it is the
  wrong tool (see *Vector stores*).
- **mypy is advisory.** ~30 pre-existing `Optional`-narrowing findings,
  mostly in the PDF parser's geometry helpers.
- **`/visualize`** loads Tailwind from a CDN; there is no offline build.

---

## License

MIT. See `LICENSE`.

Every runtime dependency is permissive (MIT, Apache-2.0, BSD, HPND,
LGPL). PDFs are parsed by pdfplumber over pdfminer.six, both MIT.

**Using a different PDF backend.** The PDF library is a configuration
choice, selected with `ATRIUM_PDF_BACKEND`. `pdfplumber` (the default)
is MIT and is the licence-clean option for everything, including a
proprietary or network-hosted deployment. `pymupdf` is faster — pdfminer
is pure Python — but it is AGPL-3.0, and AGPL §13 reaches a service
offered over a network. If you want it, install it explicitly and
accept what that means for your distribution:

```
pip install "atrium-rag[fast-pdf]"
ATRIUM_PDF_BACKEND=pymupdf
```

Atrium itself is MIT under both settings; the AGPL covers PyMuPDF,
which is yours to choose.
