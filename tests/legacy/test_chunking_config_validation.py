"""
test_chunking_config_validation.py

Strict-validation contract tests for `ChunkConfig.from_dict` — from a
token-based chunker generation that was **never committed**.

The user pushed back on silent fallbacks, and the contract here is the
right one: unknown field names → raise, bool-for-int → raise, overlap >=
target → raise, out-of-range → raise, null on an int field → raise.

But the `ChunkConfig` this file targets had token-based per-element-type
knobs (`prose_target_tokens`, `table_max_rows`, `code_max_lines`,
`child_target_tokens`, `parent_target_tokens`, `code_structural`) and a
real bge-large BPE tokenizer behind `get_tokenizer_kind()`. The shipped
chunker is character-based (`target_chars`, `overlap_chars`) and has no
tokenizer. `git log -S` finds none of these symbols in any commit.

See `tests/legacy/README.md` for why these live here and what it would
take to revive them.
"""

from __future__ import annotations

import pytest

from atrium.smart_chunker import ChunkConfig, ChunkConfigValidationError


class TestChunkConfigFromDictRejectsUnknownKeys:
    """Unknown field names (typos) must raise, not silently drop."""

    def test_unknown_key_raises(self):
        # Common mistake: `prose_target_tokes` (missing 'n').
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({"prose_target_tokes": 128})
        msg = str(ei.value)
        assert "prose_target_tokes" in msg, f"unknown key name not in message: {msg}"
        assert "valid fields" in msg, f"valid fields hint not in message: {msg}"

    def test_multiple_unknown_keys_listed(self):
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({
                "prose_target_tokes": 128,
                "table_max_rws": 10,
            })
        msg = str(ei.value)
        # Both unknown keys must be named in the error so the caller can
        # fix the payload in one round-trip.
        assert "prose_target_tokes" in msg
        assert "table_max_rws" in msg

    def test_known_key_with_unknown_key_still_rejects(self):
        # The presence of a valid field must not mask an invalid one.
        with pytest.raises(ChunkConfigValidationError):
            ChunkConfig.from_dict({
                "prose_target_tokens": 128,   # valid
                "prose_target_tokes": 128,    # typo
            })


class TestChunkConfigFromDictRejectsWrongTypes:
    """`1` is not `True` on this API. None on an int field is a bug."""

    @pytest.mark.parametrize("bad_value", [None, "128", 128.5, [128]])
    def test_null_or_non_int_rejected_for_int_field(self, bad_value):
        # None is a special "must not be null" rejection (different
        # message) so the caller can tell `omit the key` from
        # `pass null`. Non-int types fall through to the type check.
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({"prose_target_tokens": bad_value})
        msg = str(ei.value)
        assert "prose_target_tokens" in msg
        if bad_value is None:
            assert "must not be null" in msg
        else:
            assert "must be int" in msg
            # The actual type the caller sent must be surfaced so a
            # "128.5" typo is distinguishable from a True/False alike.
            assert type(bad_value).__name__ in msg

    def test_bool_rejected_for_int_field(self):
        # In Python, isinstance(True, int) is True. We reject bool
        # explicitly so prose_target_tokens=True isn't silently
        # interpreted as 1.
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({"prose_target_tokens": True})
        msg = str(ei.value)
        assert "prose_target_tokens" in msg
        assert "must be int" in msg
        assert "bool" in msg, f"actual type not surfaced: {msg}"

    @pytest.mark.parametrize("bad_value", [None, 1, "yes", [True]])
    def test_non_bool_rejected_for_bool_field(self, bad_value):
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({"code_structural": bad_value})
        msg = str(ei.value)
        assert "code_structural" in msg
        if bad_value is None:
            assert "must not be null" in msg
        else:
            assert "must be bool" in msg
            assert type(bad_value).__name__ in msg

    def test_non_str_rejected_for_str_field(self):
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({"mode": 1})
        msg = str(ei.value)
        assert "mode" in msg
        assert "must be str" in msg


class TestChunkConfigFromDictRejectsOutOfRange:
    """Bounds anchored to bge-large 512-token window + obvious positive."""

    @pytest.mark.parametrize("bad_value,field,bound", [
        (-1, "prose_target_tokens", "[16, 512]"),
        (0, "prose_target_tokens", "[16, 512]"),
        (513, "prose_target_tokens", "[16, 512]"),
        (9999, "prose_target_tokens", "[16, 512]"),
        (-1, "table_max_rows", "[1, 1000]"),
        (1001, "table_max_rows", "[1, 1000]"),
        (-1, "child_target_tokens", "[16, 512]"),
        (1025, "parent_target_tokens", "[32, 1024]"),
        (-5, "prose_overlap_tokens", "[0, 511]"),
        (512, "prose_overlap_tokens", "[0, 511]"),
    ])
    def test_out_of_range_int(self, bad_value, field, bound):
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({field: bad_value})
        msg = str(ei.value)
        assert field in msg
        assert bound in msg, f"valid range not surfaced for {field}={bad_value}: {msg}"


class TestChunkConfigFromDictCrossFieldOverlap:
    """Overlap ≥ target produces zero useful chunk content."""

    def test_overlap_equal_to_target_rejected(self):
        # prose_overlap=128 with prose_target=128 means each chunk is
        # entirely its own overlap with itself.
        with pytest.raises(ChunkConfigValidationError) as ei:
            ChunkConfig.from_dict({
                "prose_target_tokens": 128,
                "prose_overlap_tokens": 128,
            })
        msg = str(ei.value)
        assert "overlap" in msg.lower() or "overlap_tokens" in msg

    def test_overlap_greater_than_target_rejected(self):
        with pytest.raises(ChunkConfigValidationError):
            ChunkConfig.from_dict({
                "prose_target_tokens": 128,
                "prose_overlap_tokens": 200,
            })

    def test_overlap_less_than_target_accepted(self):
        # The inverse direction is fine.
        cfg = ChunkConfig.from_dict({
            "prose_target_tokens": 128,
            "prose_overlap_tokens": 16,
        })
        assert cfg.prose_target_tokens == 128
        assert cfg.prose_overlap_tokens == 16

    def test_child_target_gte_parent_target_rejected_when_parent_child_enabled(self):
        # The child must be strictly smaller than the parent for
        # parent-child retrieval to map children → parents.
        with pytest.raises(ChunkConfigValidationError):
            ChunkConfig.from_dict({
                "parent_child_enabled": True,
                "child_target_tokens": 512,
                "parent_target_tokens": 256,
            })

    def test_child_target_lt_parent_target_accepted_when_parent_child_enabled(self):
        cfg = ChunkConfig.from_dict({
            "parent_child_enabled": True,
            "child_target_tokens": 128,
            "parent_target_tokens": 512,
        })
        assert cfg.parent_child_enabled is True
        assert cfg.child_target_tokens == 128
        assert cfg.parent_target_tokens == 512


class TestChunkConfigFromDictDefaults:
    """Missing keys fall through; only what's provided is overridden."""

    def test_empty_dict_returns_custom_mode(self):
        # Empty {} still goes through the from_dict path, so the audit
        # trail forces mode='custom'. None is the "didn't invoke at
        # all" sentinel and keeps mode='auto' (tested separately).
        cfg = ChunkConfig.from_dict({})
        assert cfg.mode == "custom"
        # The rest of the fields keep their dataclass defaults.
        assert cfg.prose_target_tokens == ChunkConfig().prose_target_tokens

    def test_none_dict_returns_defaults_with_auto_mode(self):
        # None means "caller never invoked from_dict" — that's the
        # implicit / auto path. mode='auto' keeps the audit trail honest.
        cfg = ChunkConfig.from_dict(None)
        assert cfg.mode == "auto"

    def test_partial_overrides_only_touch_supplied_keys(self):
        cfg = ChunkConfig.from_dict({"prose_target_tokens": 256})
        assert cfg.prose_target_tokens == 256
        # Untouched fields keep defaults.
        default_cfg = ChunkConfig()
        assert cfg.prose_overlap_tokens == default_cfg.prose_overlap_tokens
        assert cfg.table_max_rows == default_cfg.table_max_rows
        assert cfg.code_max_lines == default_cfg.code_max_lines

    def test_mode_forced_to_custom(self):
        # Even if the caller sets mode='precise', the from_dict path
        # forces it to 'custom' so the audit trail is accurate.
        cfg = ChunkConfig.from_dict({"mode": "precise"})
        assert cfg.mode == "custom"
