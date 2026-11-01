import logging
import threading
import uuid
from dataclasses import dataclass
from typing import List, Optional

from atrium.adapters import EmbeddingAdapter, VectorDBAdapter
from atrium.models import ChunkRecord, SourceMetadata

log = logging.getLogger("ingest.pipeline")

# 2026-10-17: was `chunk_elements` (legacy structure-aware path). Now
# routes through the dispatcher so /jobs picks up the same chunking
# the consumer-facing /ingest + /retrieval_lab endpoints use, and so
# future preset/strategy changes propagate without a pipeline edit.
from atrium.smart_chunker import chunk_with_mode
from atrium.universal_parser import UniversalParser

# ---------------------------------------------------------------------------
# Per-document locking
# ---------------------------------------------------------------------------
#
# Two ingests of the same document at the same time interleave: one
# writes its chunks, the other deletes them as "stale", and the corpus
# ends up holding a mix of two versions that belongs to neither.
#
# The lock is per document, so unrelated documents still ingest in
# parallel, and it is per process, so the API and the worker each
# serialise their own ingests. Across processes the deterministic
# chunk ids are what make a repeated ingest idempotent.
_DOC_LOCKS: dict = {}
_DOC_LOCKS_GUARD = threading.Lock()


def doc_lock(doc_id: str):
    """A lock for one document's ingest."""
    with _DOC_LOCKS_GUARD:
        lock = _DOC_LOCKS.get(doc_id)
        if lock is None:
            lock = threading.Lock()
            _DOC_LOCKS[doc_id] = lock
    return lock



def _check_document_size(elements, chunks) -> None:
    """Refuse a document that is too big to ingest.

    The upload cap bounds the bytes; this bounds what those bytes turn
    into. A 100 MB PDF can be ten thousand pages, and every page
    becomes chunks, and every chunk becomes a vector that is held in
    memory until the whole document has been embedded. Without a
    ceiling, a legal upload is a way to take the process out of memory.
    """
    from atrium.config import env_int

    max_chunks = env_int("MAX_CHUNKS_PER_DOCUMENT")
    max_pages = env_int("MAX_PAGES_PER_DOCUMENT")

    if max_chunks and len(chunks) > max_chunks:
        raise ValueError(
            f"this document produces {len(chunks)} chunks, which is over the "
            f"limit of {max_chunks}. Split it, or raise "
            f"MAX_CHUNKS_PER_DOCUMENT."
        )
    pages = [e.page for e in elements if getattr(e, "page", None)]
    if max_pages and pages and max(pages) > max_pages:
        raise ValueError(
            f"this document has {max(pages)} pages, which is over the limit "
            f"of {max_pages}. Split it, or raise MAX_PAGES_PER_DOCUMENT."
        )


def _build_chunk_record(
    sc,  # atrium.smart_chunker.ChunkRecord (dataclass; type-avoided for import cost)
    ingest_run_id: str,
) -> ChunkRecord:
    """Project a smart-chunker ChunkRecord into the FastAPI ChunkRecord,
    preserving the v1 (parent-child) metadata that downstream
    `retrieve_and_rerank` reads from the persisted payload.

    `is_parent` and `schema_version` are derived deterministically:
      - `is_parent` = bool(child_ids). A parent has at least one child.
      - `schema_version` = 1 iff the chunk participates in a parent-
        child relationship (is_parent or parent_id is set), else 0.
    These are not passed by the chunker; deriving them here keeps the
    chunker API unchanged and avoids a class of "forgot to set
    schema_version" bugs at the producer.

    `chunking_strategy` is copied off the chunk: the chunker stamps
    the strategy it actually ran, so the recorded value is the truth
    rather than a re-derivation of the requested mode.
    """
    parent_id = getattr(sc, "parent_id", None)
    child_ids = list(getattr(sc, "child_ids", []) or [])
    is_parent = bool(child_ids)
    schema_version = 1 if (is_parent or parent_id) else 0

    source = SourceMetadata(
        file_name=sc.source.file_name,
        page_number=sc.source.page_number,
        bbox=sc.source.bbox,
        section_path=sc.source.section_path,
        page_numbers=getattr(sc.source, 'page_numbers', []) or [sc.source.page_number]
            if sc.source.page_number else [],
    )
    return ChunkRecord(
        doc_id=sc.doc_id,
        chunk_id=sc.chunk_id,
        ingest_run_id=ingest_run_id,
        text=sc.text,
        embed_text=getattr(sc, 'embed_text', None) or sc.text,
        element_type=sc.element_type,
        source=source,
        char_span=sc.char_span,
        content_hash=sc.content_hash,
        parent_id=parent_id,
        is_parent=is_parent,
        child_ids=child_ids,
        schema_version=schema_version,
        chunking_strategy=getattr(sc, "chunking_strategy", "") or "",
    )


@dataclass
class IngestResult:
    """What one ingestion run produced.

    Returned by `ingest_document` so a caller that is not a queue
    worker — the CLI, the HTTP service — can report counts and check
    for failure without reading a job row.
    """

    doc_id: str
    ingest_run_id: str
    chunks_written: int
    elements_parsed: int
    chunking_mode: str
    #: Pages the OCR engine could not read. Zero for a document that was
    #: not scanned. It is reported rather than swallowed, because a
    #: scanned document that OCRed to nothing still looks like a
    #: successful ingest otherwise.
    ocr_failures: int = 0


def ingest_document(
    doc_id: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    parser: UniversalParser,
    embedder: EmbeddingAdapter,
    db: VectorDBAdapter,
    chunking_mode: str = "auto",
    chunking_config: Optional[dict] = None,
) -> IngestResult:
    """Parse, chunk, embed and upsert one document.

    Raises on failure — this is the worker-free core. `run_ingestion`
    wraps it with job-store bookkeeping for the queue path.

    Chunking strategy is a parameter, not a fixed setting: the same
    document can be re-ingested later with a different strategy, and
    the versioned upsert (`delete_stale_chunks`) replaces the previous
    chunks rather than mixing granularities in one collection.
    """
    ingest_run_id = str(uuid.uuid4())

    # 1. Parse using the ML-free universal parser.
    raw_elements = parser.parse(file_bytes, filename, mime_type)
    if not raw_elements:
        raise ValueError("No content parsed.")

    # 2. Smart Chunking (handles overlap, table splitting, and content hashing).
    # 2026-10-17: dispatch through the mode-aware entry point. Default
    # `mode='auto'` now resolves to PRESETS['precise'] (D-precise,
    # structure_aware strategy) — was previously the baseline
    # sliding-window path that silently degraded /jobs accuracy.
    smart_chunks = chunk_with_mode(
        raw_elements, doc_id=doc_id, file_name=filename,
        mode=chunking_mode, config=chunking_config,
    )

    # 3. Refuse a document that would cost more than the deployment
    #    allows, before the embedding cost is incurred.
    _check_document_size(raw_elements, smart_chunks)

    # 4. Map to FastAPI/DB-compatible ChunkRecord. v1 metadata
    # (parent_id, is_parent, child_ids, schema_version) is propagated
    # in _build_chunk_record — flat chunkers leave everything at the
    # v0 defaults; parent-child chunkers populate the four fields
    # deterministically. `chunking_strategy` is copied off the chunk,
    # which the chunker stamped with the strategy it actually ran.
    chunks: List[ChunkRecord] = [
        _build_chunk_record(sc, ingest_run_id) for sc in smart_chunks
    ]

    # 5. Embed & Upsert in batches — embed against embed_text (text +
    # heading breadcrumb) so retrieval gets the section context the
    # chunker already built, not just the bare display text.
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = embedder.embed([c.embed_text or c.text for c in batch])
        for c, emb in zip(batch, embeddings):
            c.embedding = emb

    # 6. Write this version and remove the previous one, as one
    #    operation, under a lock on this document.
    #
    #    `replace_chunks` is atomic where the store can be (SQLite runs
    #    both statements in one transaction; Postgres likewise), and is
    #    ordered delete-then-write where it cannot be, so a crash
    #    leaves the document absent — recoverable by re-ingesting —
    #    rather than present twice, which is not.
    with doc_lock(doc_id):
        db.replace_chunks(doc_id, chunks, ingest_run_id)

    return IngestResult(
        doc_id=doc_id,
        ingest_run_id=ingest_run_id,
        chunks_written=len(chunks),
        elements_parsed=len(raw_elements),
        chunking_mode=chunking_mode,
        ocr_failures=int(getattr(parser, "last_parse_ocr_failures", 0) or 0),
    )


def run_ingestion(job_id: str, doc_id: str, file_bytes: bytes, filename: str, mime_type: str,
                  parser: UniversalParser, embedder: EmbeddingAdapter, db: VectorDBAdapter,
                  job_store,
                  chunking_mode: str = "auto",
                  chunking_config: Optional[dict] = None):
    try:
        job_store.update(job_id, status="running")
        result = ingest_document(
            doc_id=doc_id,
            file_bytes=file_bytes,
            filename=filename,
            mime_type=mime_type,
            parser=parser,
            embedder=embedder,
            db=db,
            chunking_mode=chunking_mode,
            chunking_config=chunking_config,
        )
        job_store.update(
            job_id,
            status="complete",
            chunks_written=result.chunks_written,
            ocr_failures=result.ocr_failures,
        )
        if result.ocr_failures:
            # The document is in the corpus, but some of its pages were
            # not readable. Saying so is the difference between "your
            # scan was not OCRed" and "why is my document empty?".
            log.warning(
                "job %s: %d page(s) could not be read by OCR",
                job_id, result.ocr_failures,
            )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e))
        # The failure is recorded on the job row *and* raised. Swallowing
        # it here meant the worker logged "complete" for a job that had
        # failed, and the HTTP path answered 200 for a document that was
        # never indexed. The row keeps the detail for anyone polling
        # GET /jobs/{id}; the exception carries the type, which is what
        # the HTTP layer maps to a status.
        raise
