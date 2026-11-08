from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    file_name: str
    page_number: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    section_path: List[str] = []  # Updated to List[str] to match smart_chunker.py
    page_numbers: List[int] = []  # every distinct page this chunk spans (first page also goes here)

class ChunkRecord(BaseModel):
    doc_id: str
    chunk_id: str
    ingest_run_id: str
    text: str
    embed_text: Optional[str] = None   # text + heading breadcrumb — what the embedder should see
    embedding: Optional[List[float]] = None
    element_type: str  # Removed Literal constraint to support dynamic types
    source: SourceMetadata
    char_span: Tuple[int, int]
    content_hash: str
    # ---- v1 (parent-child) metadata --------------------------------------
    # Persisted in the Qdrant/PG payload so `retrieve_and_rerank` can
    # detect a v1 corpus via `collection_schema_version` and route to
    # the parent-child branch (search_children + dedup + retrieve_parents).
    #
    # Defaults preserve v0 (legacy flat) behavior — flat chunkers
    # (Strategies A/B/C/D) leave child_ids empty and parent_id None, so
    # `is_parent` is False and `schema_version` is 0. Strategy E
    # (parent-child) populates these fields; the pipeline computes
    # is_parent and schema_version deterministically from child_ids
    # and parent_id (see atrium/pipeline.py:_build_chunk_record).
    parent_id: Optional[str] = None
    is_parent: bool = False
    child_ids: List[str] = Field(default_factory=list)
    schema_version: int = 0
    # Which chunking strategy produced this chunk ("baseline",
    # "sentence_bounded", "recursive_token", "structure_aware",
    # "parent_child"). Recorded so a corpus with mixed strategies —
    # which is what per-request chunking overrides make possible — can
    # be audited after the fact: a hit that looks wrong can be traced
    # back to the strategy that shaped it. Persisted in the
    # Qdrant/PG payload alongside the text, so it survives a restart.
    chunking_strategy: str = ""


def source_fields(payload: dict) -> tuple:
    """The page number and the heading path of a stored chunk.

    `ChunkRecord` keeps them in `SourceMetadata`, so in the persisted
    payload they sit under `source` — not at the top level. Every
    adapter used to read them from the top level, which returned
    `None` and `[]`: the key was there, the value was not, and every
    hit looked like it came from a page nobody could cite.

    Both levels are accepted, so a payload written by an older build
    (or by hand) still reads correctly.
    """
    source = payload.get("source") or {}
    page = source.get("page_number")
    if page is None:
        page = payload.get("page_number")
    section = source.get("section_path")
    if section is None:
        section = payload.get("section_path")
    return page, list(section or [])

class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "complete", "failed"]
    doc_id: Optional[str] = None
    chunks_written: Optional[int] = None
    error: Optional[str] = None
