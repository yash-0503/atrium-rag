"""
atrium/adapters.py

Embedding and vector-DB adapters behind small protocols. Implementations:
- OpenAICompatibleEmbedder (Ollama, OpenAI, vLLM, LM Studio, TEI — anything
  OpenAI-shaped).
- QdrantAdapter and PgVectorAdapter for storage; both implement the
  VectorDBAdapter protocol, so callers swap backends without touching
  pipeline code.
"""

from typing import Any, Dict, List, Optional, Protocol

from tenacity import retry, stop_after_attempt, wait_exponential

from atrium.models import ChunkRecord

# The vector-store client libraries are imported *inside* the adapter
# that uses them, not here at module scope.
#
# `qdrant_client` costs ~300 ms to import (its generated HTTP client
# is most of it) and `psycopg_pool` ~65 ms. A pgvector deployment
# should not pay for the Qdrant client, a Qdrant deployment should
# not pay for the Postgres pool, and neither should a process that
# only imports `atrium.adapters` for the embedder. The trade is a
# local import in a handful of methods; the win is a cold start that
# only loads the backend you actually configured.
#
# The three client entry points the test suite patches
# (`QdrantClient`, `ConnectionPool`, `register_vector`) are resolved
# through the helpers below instead of a bare local import, so
# `patch("atrium.adapters.QdrantClient")` still works: mock writes a
# real module attribute, and the helpers prefer a module attribute
# over the real import.


def __getattr__(name: str):
    """Resolve the heavy vector-DB client classes on first access
    (PEP 562). Keeps them out of the import path without removing
    them from the module's public surface."""
    if name == "QdrantClient":
        from qdrant_client import QdrantClient
        return QdrantClient
    if name == "ConnectionPool":
        from psycopg_pool import ConnectionPool
        return ConnectionPool
    if name == "register_vector":
        from pgvector.psycopg import register_vector
        return register_vector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _qdrant_search(client, **kwargs):
    """Vector search, across qdrant-client's API rename.

    qdrant-client renamed ``search()`` to ``query_points()`` in 1.10
    and removed the old name in 1.20. Since ``pip install atrium-rag``
    resolves the dependency to the newest release, an adapter that
    only spoke ``search()`` breaks the moment anyone installs it —
    every ``/search`` returns 503 with
    ``'QdrantClient' object has no attribute 'search'``.

    Both spellings take the same arguments and return the same
    points; ``query_points`` wraps them in a response object with a
    ``.points`` attribute, so unwrap it.

    Found by the performance measurement, which installs the package
    from the built wheel rather than from this working tree.
    """
    query_points = getattr(client, "query_points", None)
    if query_points is not None:
        # `search()` took `query_vector=`; `query_points()` takes `query=`.
        # Everything else (query_filter, limit, score_threshold) is
        # spelled the same in both.
        if "query_vector" in kwargs:
            kwargs = dict(kwargs, query=kwargs.pop("query_vector"))
        response = query_points(**kwargs)
        return getattr(response, "points", response)
    return client.search(**kwargs)


def _qdrant_client_cls():
    """The Qdrant client class — a patched module attribute wins."""
    cls = globals().get("QdrantClient")
    if cls is not None:
        return cls
    from qdrant_client import QdrantClient as cls
    return cls


def _connection_pool_cls():
    """The psycopg pool class — a patched module attribute wins."""
    cls = globals().get("ConnectionPool")
    if cls is not None:
        return cls
    from psycopg_pool import ConnectionPool as cls
    return cls


def _register_vector_fn():
    """The pgvector psycopg registration helper — a patched module
    attribute wins."""
    fn = globals().get("register_vector")
    if fn is not None:
        return fn
    from pgvector.psycopg import register_vector as fn
    return fn


# Protocols

class VisionAdapter(Protocol):
    def describe_image(self, image_bytes: bytes, mime_type: str = "image/png") -> str: ...
    def health(self) -> bool: ...

class EmbeddingAdapter(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    @property
    def dimension(self) -> int: ...
    def health(self) -> bool: ...

class VectorDBAdapter(Protocol):
    def ensure_collection(self, name: str, dim: int) -> None: ...
    def upsert(self, chunks: List[ChunkRecord], batch_size: int = 100) -> None: ...
    def delete_stale_chunks(self, doc_id: str, current_run_id: str) -> None: ...
    def health(self) -> bool: ...
    # Task B (2026-10-10): retrieval protocol shared by atrium.retrieval.
    # Adapters must implement all four for `retrieve_and_rerank` to
    # exercise the unified v1 path; collection_schema_version returning
    # 0 + missing search_children keeps the helper on the legacy v0
    # branch, but tests/test_adapters_contract.py asserts the full
    # surface (Qdrant: 14 tests, PgVector: 2 tests, both backends).
    def collection_schema_version(self) -> int: ...
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]: ...
    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]: ...
    def retrieve_parents(
        self, parent_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]: ...

class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, model: str = "nomic-embed-text:v1.5", dim: int = 768):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        import httpx  # deferred; ~48 ms, and only the embedder needs it.

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{self.base_url}/embeddings", json={"input": texts, "model": self.model})
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]
            
    @property
    def dimension(self) -> int: return self._dim

    def health(self) -> bool:
        import httpx  # deferred; see above.

        try:
            with httpx.Client(timeout=2.0) as c: return c.get(f"{self.base_url}/models").status_code == 200
        except Exception: return False

# Vector DBs -- both implementations of VectorDBAdapter. Tenacity retries
# on transient failures, dim mismatch checked at collection ensure.

class QdrantAdapter:
    def __init__(self, host: str, port: int = 6333):
        self.client = _qdrant_client_cls()(host=host, port=port)

    def ensure_collection(self, name: str, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if self.client.collection_exists(name):
            col = self.client.get_collection(name)
            if col.config.params.vectors.size != dim:
                raise ValueError(f"Qdrant collection '{name}' exists with dim {col.config.params.vectors.size}, but config requests {dim}. Aborting to prevent corruption.")
        else:
            self.client.create_collection(collection_name=name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def upsert(self, chunks: List[ChunkRecord], batch_size: int = 100) -> None:
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            points = [PointStruct(id=c.chunk_id, vector=c.embedding, payload=c.model_dump(exclude={"embedding"})) for c in batch]
            self.client.upsert(collection_name="documents", points=points)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def delete_stale_chunks(self, doc_id: str, current_run_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        # Delete points belonging to this doc_id that DO NOT match the new run_id
        self.client.delete(
            collection_name="documents",
            points_selector=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            ], must_not=[
                FieldCondition(key="ingest_run_id", match=MatchValue(value=current_run_id))
            ])
        )

    def health(self) -> bool:
        try: return self.client.get_collections() is not None
        except Exception: return False

    # ------------------------------------------------------------------
    # Retrieval protocol (Task B, 2026-10-10). Mirror the shape used by
    # the benchmark harness's _BenchQdrant + _BenchHelperDB so the
    # production adapter is drop-in for atrium.retrieval.retrieve_and_rerank.
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_to_dict(h) -> Dict[str, Any]:
        """Map a qdrant_client ScoredPoint to the retrieval hit shape.
        Default `is_parent=False` and `parent_id=None` so legacy v0
        points (which never had those fields in their payload) flow
        through `search()` without TypeError downstream."""
        payload = h.payload or {}
        return {
            "id": str(h.id),
            "score": float(h.score or 0.0),
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id", ""),
            "chunk_id": payload.get("chunk_id", str(h.id)),
            "page_number": payload.get("page_number"),
            "section_path": payload.get("section_path") or [],
            "element_type": payload.get("element_type", ""),
            "is_parent": bool(payload.get("is_parent", False)),
            "parent_id": payload.get("parent_id"),
            "chunking_strategy": payload.get("chunking_strategy", ""),
        }

    def collection_schema_version(self) -> int:
        """Count points with payload.schema_version >= 1 in 'documents'.
        Empty collection, missing collection, or any error → 0 (v0
        legacy default — the helper treats that corpus as flat)."""
        from qdrant_client.models import FieldCondition, Filter, Range

        try:
            result = self.client.count(
                collection_name="documents",
                count_filter=Filter(must=[
                    FieldCondition(key="schema_version", range=Range(gte=1)),
                ]),
            )
            return 1 if int(getattr(result, "count", 0)) > 0 else 0
        except Exception:
            return 0

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Flat v0 path: no filter. Legacy points and v1 children both
        returned; the caller sorts/dedups by score."""
        kwargs: Dict[str, Any] = {
            "collection_name": "documents",
            "query_vector": list(query_vector),
            "limit": int(top_k),
        }
        if score_threshold is not None:
            kwargs["score_threshold"] = float(score_threshold)
        hits = _qdrant_search(self.client, **kwargs)
        return [self._hit_to_dict(h) for h in hits]

    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """v1 path: filter `is_parent=False`. Returns children only;
        parents are excluded from the candidate set."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        kwargs: Dict[str, Any] = {
            "collection_name": "documents",
            "query_vector": list(query_vector),
            "limit": int(top_k),
            "query_filter": Filter(must=[
                FieldCondition(key="is_parent", match=MatchValue(value=False)),
            ]),
        }
        if score_threshold is not None:
            kwargs["score_threshold"] = float(score_threshold)
        hits = _qdrant_search(self.client, **kwargs)
        return [self._hit_to_dict(h) for h in hits]

    def retrieve_parents(
        self, parent_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch fetch parent points by id. Empty input → {} without
        touching the DB. Any error during the fetch → {}. Returned
        dict is keyed by `payload.chunk_id` (falling back to str(point
        id) when the payload omits chunk_id), matching the shape
        atrium.retrieval._resolve_parents expects at lines 561–569."""
        if not parent_ids:
            return {}
        try:
            points = self.client.retrieve(
                collection_name="documents",
                ids=list(parent_ids),
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for p in points:
            payload = p.payload or {}
            key = str(payload.get("chunk_id") or p.id)
            out[key] = {
                "text": payload.get("text", ""),
                "page_number": payload.get("page_number"),
                "section_path": payload.get("section_path") or [],
                "element_type": payload.get("element_type", ""),
            }
        return out

class PgVectorAdapter:
    def __init__(self, conn_info: str):
        self.pool = _connection_pool_cls()(conn_info)

    def ensure_collection(self, name: str, dim: int) -> None:
        with self.pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # Check the existing table's embedding dimension (if any)
            # and fail clearly on mismatch. pgvector stores the dim
            # in `pg_attribute.atttypmod` as `dim + VARHDRSZ` (the
            # typmod header is 4 bytes per pgvector's vector.c
            # typmod_in convention). The previous stub at this site
            # did `pass` — silently creating a `chunks` table with
            # `embedding VECTOR({dim})` even when the table already
            # existed with a different dim, which produced vectors of
            # the wrong dim landing in a mismatched column.
            #
            # `_VECTOR_TYPMOD_HEADER = 4` mirrors pgvector's source
            # and is the only magic number in this block.
            _VECTOR_TYPMOD_HEADER = 4
            # `to_regclass('chunks')` returns NULL when the table does
            # not exist yet; the plain cast `'chunks'::regclass` raises
            # UndefinedTable, which made the pgvector backend fail at
            # startup on a fresh database — before CREATE TABLE could
            # run. The check below has to tolerate the table's absence.
            existing_dim = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass('chunks') "
                "  AND attname = 'embedding'"
            ).fetchone()
            if existing_dim is not None:
                stored = int(existing_dim[0]) - _VECTOR_TYPMOD_HEADER
                if stored != dim:
                    raise ValueError(
                        f"PgVector table 'chunks' exists with embedding dim "
                        f"{stored}, but config requests {dim}. Aborting to "
                        f"prevent corruption. Re-create the table with the "
                        f"correct dim or align EMBED_DIM with the table."
                    )

            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id UUID PRIMARY KEY,
                    doc_id TEXT,
                    ingest_run_id TEXT,
                    embedding VECTOR({dim}),
                    payload JSONB
                );
            """)

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def upsert(self, chunks: List[ChunkRecord], batch_size: int = 100) -> None:
        with self.pool.connection() as conn:
            _register_vector_fn()(conn)
            with conn.cursor() as cur:
                for chunk in chunks:
                    cur.execute(
                        "INSERT INTO chunks (chunk_id, doc_id, ingest_run_id, embedding, payload) VALUES (%s, %s, %s, %s, %s) ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, payload = EXCLUDED.payload;",
                        (chunk.chunk_id, chunk.doc_id, chunk.ingest_run_id, chunk.embedding, chunk.model_dump_json(exclude={"embedding"}))
                    )

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def delete_stale_chunks(self, doc_id: str, current_run_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = %s AND ingest_run_id != %s", (doc_id, current_run_id))

    def health(self) -> bool:
        try:
            with self.pool.connection() as conn: return conn.execute("SELECT 1;").fetchone()[0] == 1
        except Exception: return False

    # ------------------------------------------------------------------
    # Retrieval protocol (Task B, 2026-10-10). Same hit-dict shape as
    # QdrantAdapter so atrium.retrieval.retrieve_and_rerank is backend-
    # agnostic. `score = 1 - (embedding <=> query)` for cosine
    # similarity (pgvector operator; lower distance = closer).
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_hit(chunk_id, doc_id, payload, score) -> Dict[str, Any]:
        """Map a (chunk_id, doc_id, payload, score) row to the hit
        shape. `chunk_id` is the UUID primary key; `id` mirrors it
        so callers can use either interchangeably."""
        payload = payload or {}
        return {
            "id": str(chunk_id),
            "score": float(score),
            "text": payload.get("text", ""),
            "doc_id": doc_id or "",
            "chunk_id": str(chunk_id),
            "page_number": payload.get("page_number"),
            "section_path": payload.get("section_path") or [],
            "element_type": payload.get("element_type", ""),
            "is_parent": bool(payload.get("is_parent", False)),
            "parent_id": payload.get("parent_id"),
            "chunking_strategy": payload.get("chunking_strategy", ""),
        }

    def collection_schema_version(self) -> int:
        """Count rows whose payload.schema_version >= 1. Empty table
        or any error → 0 (legacy default)."""
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM chunks "
                        "WHERE (payload->>'schema_version')::int >= 1"
                    )
                    (count,) = cur.fetchone()
            return 1 if int(count) > 0 else 0
        except Exception:
            return 0

    def _vector_search(
        self,
        query_vector: List[float],
        top_k: int,
        score_threshold: Optional[float],
        where_clause: str = "",
        where_params: tuple = (),
    ) -> List[Dict[str, Any]]:
        """Shared SELECT for `search` and `search_children`. Cosine
        similarity is `1 - (embedding <=> %s::vector)`. `where_clause`
        gates the v1 `is_parent` filter; passed as a parameterizable
        SQL fragment because pgvector's `<=>` operator doesn't bind
        into a single placeholder."""
        qvec = list(query_vector)
        threshold_sql = ""
        threshold_params: List[Any] = []
        if score_threshold is not None:
            threshold_sql = (
                " AND 1 - (embedding <=> %s::vector) >= %s"
                if where_clause
                else " WHERE 1 - (embedding <=> %s::vector) >= %s"
            )
            threshold_params = [qvec, float(score_threshold)]

        sql = (
            "SELECT chunk_id, doc_id, payload, "
            "1 - (embedding <=> %s::vector) AS score "
            f"FROM chunks{where_clause}{threshold_sql} "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        params = [qvec, *where_params, *threshold_params, qvec, int(top_k)]
        with self.pool.connection() as conn:
            _register_vector_fn()(conn)
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()
        return [
            self._row_to_hit(chunk_id, doc_id, payload, score)
            for (chunk_id, doc_id, payload, score) in rows
        ]

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Flat v0 path: no filter. v0 + v1 corpus both surface."""
        return self._vector_search(query_vector, top_k, score_threshold)

    def search_children(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """v1 path: filter `is_parent = false` via JSONB lookup.
        Legacy rows without the field default to `false` via
        COALESCE so the helper's v1/v0 branching stays correct."""
        return self._vector_search(
            query_vector,
            top_k,
            score_threshold,
            where_clause=(
                " WHERE COALESCE((payload->>'is_parent')::boolean, false) = false"
            ),
        )

    def retrieve_parents(
        self, parent_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Batch fetch parents by chunk_id. Empty input → {} without
        touching the DB. Any error → {}. Keyed by chunk_id (UUID
        string) — matches the Qdrant adapter shape so the helper
        is backend-agnostic."""
        if not parent_ids:
            return {}
        try:
            with self.pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_id, payload FROM chunks "
                        "WHERE chunk_id = ANY(%s::uuid[])",
                        (list(parent_ids),),
                    )
                    rows = cur.fetchall()
        except Exception:
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for chunk_id, payload in rows:
            if payload is None:
                payload = {}
            out[str(chunk_id)] = {
                "text": payload.get("text", ""),
                "page_number": payload.get("page_number"),
                "section_path": payload.get("section_path") or [],
                "element_type": payload.get("element_type", ""),
            }
        return out