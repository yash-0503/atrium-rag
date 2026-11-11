# GitHub issues — atrium-rag

Twelve issues, ready to file. Each has a title, a body, and the label
it should carry. They are ordered so the small, well-scoped ones come
first — those are the ones a first-time contributor can pick up and
finish, and a repository with a few of those open looks alive in a way
a repository with only roadmap issues does not.

**How to file them without the CLI.** Install it once:
```bash
brew install gh
gh auth login
```
Then, from the repository root:
```bash
gh issue create --title "..." --body-file issues/01-machine-readable-doctor.md --label "enhancement"
```
Or open <https://github.com/yash-0503/atrium-rag/issues/new> and paste
the title and body. The bodies are below in plain text so either
works.

**Labels to create first** (Settings → Labels): `enhancement`,
`bug`, `good first issue`, `help wanted`, `documentation`,
`performance`, `needs-decision`.

---

## 1. `atrium doctor --json` — machine-readable diagnostics

**Labels:** enhancement, good first issue

`atrium doctor` prints a human report: the embedder, the vector store,
the security posture. That is exactly what you want on a terminal and
exactly what you do not want in a script.

Add a `--json` flag that prints the same findings as a machine-readable
document — one object per check, each with `name`, `ok`, `detail`, and
`severity`. A CI job should be able to run it and fail on a
non-healthy check without parsing prose.

**Acceptance:** `atrium doctor --json` exits 0 when every required
check passes and 1 otherwise, and the output parses with
`json.loads`. The human output is unchanged when the flag is absent.

---

## 2. Expose the OCR engine's supported languages in the init wizard

**Labels:** bug, good first issue, help wanted

`atrium init` offers nine OCR languages (eng, ara, chi_sim, jpn, kor,
deu, fra, spa, rus). `atrium/ocr_adapter.py` has
`SUPPORTED_LANGS = frozenset({"eng", "en", "chi", "ch", "zh", "zho"})`
for RapidOCR — six of them. Tesseract reads all nine.

So a user who picks RapidOCR and then Arabic gets a page that comes
back empty, and the wizard never said the combination was a mistake.

**Acceptance:** the language list offered in `atrium init` depends on
the engine chosen in the previous question. RapidOCR offers only the
languages it can read; Tesseract offers all of them. A test asserts
that no offered combination is unsupported.

---

## 3. Script-aware BM25 tokenization for CJK and Arabic

**Labels:** enhancement, help wanted

`atrium/bm25.py` splits on whitespace and punctuation. That works for
Latin scripts and fails completely for two others:

- **Arabic** has no inter-word spaces, so every "word" in the corpus
  becomes a whole sentence and IDF is meaningless.
- **CJK** has no spaces at all, so every "word" is a whole document.

This is not a theoretical problem. It was measured: turning on dense +
BM25 fusion with Reciprocal Rank Fusion regressed the 19-fixture suite
from R@5 0.8158 to 0.80, and the two catastrophic losses were Arabic
(−0.600) and a scanned PDF (−0.333) — both cases of the tokenizer
failing on the corpus.

**Acceptance:** BM25 tokenizes CJK and Arabic into terms rather than
into whole lines — per-character for CJK, and for Arabic either a
proper segmenter or a stemmer. The regression fixture is a mixed
corpus with one Arabic and one CJK document, and the dense + BM25
result on it is not worse than dense alone.

---

## 4. Make the sparse ranker reachable from `/search`

**Labels:** enhancement, needs-decision

`atrium/retrieval.py` implements dense + BM25 fusion with Reciprocal
Rank Fusion, a confidence gate that decides when to consult the sparse
ranker, and the threshold constants `DENSE_CONFIDENT_GAP` and
`DENSE_MIN_SCORE`. All of it is tested.

**No production code path reaches it.** `POST /search` calls
`retrieve_and_rerank` without a `bm25_index`, so retrieval is dense
plus an optional cross-encoder rerank. The Retrieval Lab has its own
separate RRF implementation, written inline in `atrium/main.py`, which
has no confidence gate. There are two BM25 paths and they have drifted
apart.

The measured numbers on the benchmark corpus: dense alone 0.655 R@5,
dense + BM25 fused 0.700 R@5.

**What this needs, before it is an implementation task:** a decision
on where the sparse index lives. A server needs a per-corpus index it
can consult per query — built at startup and invalidated on write, or
persisted. This is why the feature was left unreachable: not because
the fusion was hard, but because nobody decided where the index lives.

**Acceptance:** `/search` can be configured to consult the sparse
ranker; the index is built and invalidated correctly; the two RRF
implementations are one implementation.

---

## 5. Parse two-column academic papers without interleaving the columns

**Labels:** enhancement, help wanted

The parser does multi-column reading-order correction, applied to text
and tables together. It gets the common case right and the hard case
wrong.

Measured against the other parsers, per-document text coverage on the
benchmark corpus:

| document | atrium | docling | unstructured |
|---|---|---|---|
| `04_bert_paper` | 0.558 | 0.886 | 0.892 |
| `21_arxiv_shear_instability` | 0.564 | 0.848 | 0.720 |
| `14_cjk_wikipedia` | 0.669 | 0.852 | 0.759 |
| `17_irs_w2` | 0.552 | 0.715 | 0.754 |

The two-column arXiv papers are the largest single gap in the parser.
The corpus and the coverage report are in the `atrium-benchmarks`
repository.

**Acceptance:** text coverage on `04_bert_paper` reaches 0.85 or
better, with no regression on any other fixture. The reading-order
metric stays at 1.0.

---

## 6. Table cells with row and column indices

**Labels:** enhancement, needs-decision

The parser emits a table as one text element, kept whole and never
split across chunks. That is the right behaviour for a chunk, and it
means a chunk cannot answer "what is in row 3, column 2".

The benchmark scores Docling's table cells at F1 0.592. Ours is not
scoreable at all, because the harness has no cell structure to compare
against. `tables_found` is 0.0 per document for atrium and unstructured
against Docling's 2.217.

**What this needs:** a decision on the shape. Cells as a structured
field on the element (a list of rows, each a list of cell strings, with
a row/column index), or as text plus a separate side table. The
chunking side is straightforward either way; the question is whether
the cell structure survives into the vector store's payload, because
that is what makes it addressable at query time.

**Acceptance:** the harness can score our table cells against the
competition's, with an F1 number in the same report.

---

## 7. `context_rich` produces a parent per section, not per 3000 characters

**Labels:** documentation, good first issue

The README says `context_rich` "adds a parent chunk over each
section". The implementation (`_chunk_parent_child` in
`atrium/smart_chunker.py`) generates parents by running the
structure-aware chunker at `parent_target_chars` — 3000 characters.
A parent is therefore a 3000-character chunk, not a section.

Both are defensible; only one is documented. Either the behaviour
changes or the sentence does.

**Acceptance:** the README and the docs describe what the code does.

---

## 8. `auto` resolves to `precise` for every document

**Labels:** documentation, good first issue

`auto` is documented in the README as resolving to the precise
strategy "and adjusts the parameters to the shape of the document". It
does not: `_resolve_chunk_config` in `atrium/smart_chunker.py` returns
`PRESETS["precise"]` — structure-aware, 700 characters, 100 overlap —
for every document, with no shape analysis.

The site's own copy is correct ("Picks a strategy per document — not
yet"). The README is not.

**Acceptance:** either the README says what the code does, or the
shape heuristics are implemented. If they are implemented: a document
that is mostly prose, mostly tables, or mostly code gets a different
configuration, and the behaviour is measurable on the benchmark corpus.

---

## 9. Publish the benchmark as a scheduled CI job

**Labels:** enhancement, help wanted

The evaluation harness lives in `atrium-benchmarks` and is run by
hand. A change to the parser or the chunker can move the headline
number, and nothing notices until someone remembers to re-run it.

**Acceptance:** a scheduled workflow runs the harness on the benchmark
corpus, writes the results to `docs/benchmark/`, and fails when the
headline R@5 moves by more than the noise band. The site's
`check-numbers` job then picks up the new file.

---

## 10. Review the 360 machine-drafted gated questions

**Labels:** good first issue, help wanted, documentation

The published benchmark has 700 questions across five difficulty
tiers. The hard, implicit and unanswerable tiers are machine-drafted
and 360 of them are still unreviewed. Every published number was
produced with those questions open, which the report states.

This is a genuine contribution opportunity: reviewing a question is
checking that it has one right answer, that the answer is actually in
the document, and that it is not answerable by pattern-matching the
question text.

**Acceptance:** the review CLI reports zero unreviewed questions, and
the numbers are re-run against the reviewed set.

---

## 11. Add per-document coverage to the comparison table

**Labels:** enhancement, help first issue

The benchmark's headline is a corpus average. The average hides the
two places we are worst: two-column papers (0.558 coverage against
Docling's 0.886) and government forms (0.552 against 0.715). A reader
of the headline cannot see them.

**Acceptance:** the comparison report includes a per-document table —
coverage, precision and reading order for every system, every
document — and the summary states the two weakest document types
rather than only the mean.

---

## 12. A `CONTRIBUTING` note on which documents to test against

**Labels:** documentation, good first issue

The parsing heuristics and the chunking strategies are tuned against
documents rather than against tests, so a change that looks right on
one PDF is often wrong on the next. `CONTRIBUTING.md` says this and
then stops.

**Acceptance:** `CONTRIBUTING.md` names the document categories a
parser or chunker change should be checked against — single-column
prose, two-column papers, forms, tables, slides, scans, CJK, Arabic —
and says how to run the harness against one of them.

---

## Labels

Create these in the repository if they do not exist: `enhancement`,
`bug`, `good first issue`, `help wanted`, `documentation`,
`performance`, `needs-decision`.

`needs-decision` is the important one: issues 4, 6 and 8 are blocked
on a design choice, not on work, and marking them as such stops them
looking like neglect and invites the discussion.
