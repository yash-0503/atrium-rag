import uuid
from dataclasses import dataclass
from typing import List, Optional

from atrium.adapters import EmbeddingAdapter, VectorDBAdapter
from atrium.models import ChunkRecord, SourceMetadata

# 2026-10-17: was `chunk_elements` (legacy structure-aware path). Now
# routes through the dispatcher so /jobs picks up the same chunking
# the consumer-facing /ingest + /retrieval_lab endpoints use, and so
# future preset/strategy changes propagate without a pipeline edit.
from atrium.smart_chunker import chunk_with_mode
from atrium.universal_parser import UniversalParser


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

    # 3. Map to FastAPI/DB-compatible ChunkRecord. v1 metadata
    # (parent_id, is_parent, child_ids, schema_version) is propagated
    # in _build_chunk_record — flat chunkers leave everything at the
    # v0 defaults; parent-child chunkers populate the four fields
    # deterministically. `chunking_strategy` is copied off the chunk,
    # which the chunker stamped with the strategy it actually ran.
    chunks: List[ChunkRecord] = [
        _build_chunk_record(sc, ingest_run_id) for sc in smart_chunks
    ]

    # 4. Embed & Upsert in batches — embed against embed_text (text +
    # heading breadcrumb) so retrieval gets the section context the
    # chunker already built, not just the bare display text.
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = embedder.embed([c.embed_text or c.text for c in batch])
        for c, emb in zip(batch, embeddings):
            c.embedding = emb

    db.upsert(chunks)

    # 5. Versioned Upsert: Delete stale chunks from previous runs.
    db.delete_stale_chunks(doc_id, ingest_run_id)

    return IngestResult(
        doc_id=doc_id,
        ingest_run_id=ingest_run_id,
        chunks_written=len(chunks),
        elements_parsed=len(raw_elements),
        chunking_mode=chunking_mode,
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
            job_id, status="complete", chunks_written=result.chunks_written,
        )
    except Exception as e:
        job_store.update(job_id, status="failed", error=str(e))
