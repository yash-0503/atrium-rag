# `tests/legacy/` — orphaned chunker generation

These tests are **not collected** (see `conftest.py` in this directory).
They are kept because the reasoning behind them is worth keeping, and
because deleting them would throw away the record of a design decision.

## What they test

A **token-based chunker generation** built during the 2026-10-10 chunking
strategy grid:

| Aspect | Legacy generation | Shipped chunker |
|---|---|---|
| Budget unit | tokens (`prose_target_tokens`, `table_max_rows`, `code_max_lines`, `child_target_tokens`, `parent_target_tokens`) | characters (`target_chars`, `overlap_chars`, `parent_target_chars`) |
| Tokenizer | real bge-large BPE (`get_tokenizer_kind()` → `"bpe"`) | none — `CHARS_PER_TOKEN_ESTIMATE = 4` character estimate |
| Per-element-type knobs | yes (prose / table / code / list each tuned separately) | no — one `target_chars`/`overlap_chars` pair |
| Config entry point | `ChunkConfig.from_dict` with strict validation → `ChunkConfigValidationError` | `chunk_with_mode(mode=..., config=...)` with unknown-key rejection |
| Auto mode | `_derive_auto_config` + heuristics (long_prose / table_heavy / code_heavy) | `auto` → `PRESETS["precise"]` unconditionally |
| Aliases | `chunk_auto`, `chunk_default`, `DEFAULT_CHUNK_CONFIG` | `chunk_default`, `DEFAULT_CHUNK_CONFIG` (restored) |

## Why they are here and not in `tests/`

The legacy generation was **never committed**. `git log -S` across the
entire history finds `ChunkConfigValidationError`, `chunk_auto`,
`_derive_auto_config`, `get_tokenizer_kind`, `_AUTO_LONG_PROSE_CHARS`,
`def chunk_with_config` and `def chunk_with_parent_child` in **zero
commits** — the implementation, the tests and the writeup
(`docs/audits/CHUNKING_EXPERIMENTS.md`) all lived only in the working
tree.

(`chunk_default`, `DEFAULT_CHUNK_CONFIG` and `_match_signature` were in
that same never-committed set. Unlike the rest, they were small enough
to restore into the shipped tree in v0.2.0 — see the CHANGELOG.)

The shipped `atrium/smart_chunker.py` is the character-based generation.
It carries the *strategy* work from the grid (the `balanced` / `precise` /
`context_rich` presets, the `chunk_with_mode` dispatcher, the 5-strategy
dispatch table) but not the token-based *config surface*.

## Why the design in here is still right

The strict-validation contract in `test_chunking_config_validation.py` is
exactly what a config API should do, and the shipped dispatcher only
partially implements it — `chunk_with_mode` rejects unknown keys but does
**not** yet type-check, range-check, or cross-field-check overrides.

The gap is real. What is missing is the machinery to express the checks
against, not the idea of the checks.

## What it would take to revive it

Not a small job — this is the reason the generation was not committed:

1. Extend `ChunkConfig` with the per-element-type token knobs, or
   re-express the existing character knobs under token names.
2. Decide the tokenizer story: adding `transformers` as a hard dependency
   for one chunker is a large cost for the "lightweight + fast" motto,
   and the alternative (a character estimate) is what ships today.
3. Implement `from_dict` + `ChunkConfigValidationError` with the type,
   range and cross-field rules.
4. Port `_derive_auto_config` and its three heuristics.
5. **Re-measure.** The grid's headline (+0.0314 R@5) was measured against
   a benchmark state that was later found to be unreproducible — see the
   "0.8717 is a stale outlier" finding in the project memory. Any port
   has to re-earn its number on the 19-fixture set before it ships.

## Files

- `test_chunking_auto.py` — the auto-mode heuristic tests (24)
- `test_chunking_config_validation.py` — the `from_dict` strict-validation
  contract tests
- `test_parent_child_e2e.py` — a live-Qdrant parent-child e2e script that
  imports `chunk_with_config` / `chunk_with_parent_child`; both were
  never defined in any commit. Moved here from `scripts/` because pytest
  was collecting it and failing at import.

## Running them anyway

They will fail against the shipped chunker — that is the point; they
describe an API that does not exist. To see the failures:

```bash
pytest tests/legacy --collect-only   # shows 0 collected (ignored)
```

To actually run them, delete `collect_ignore_glob` from `conftest.py`.
