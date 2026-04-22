# travel-rag — agent context

## Project overview

Local RAG over personal travel notes: **ChromaDB** (persistent store under `./chroma_db`) + **mixedbread-ai/mxbai-embed-large-v1** (SentenceTransformers) for embeddings + **Anthropic Messages API** (Claude) for answers.

Flow: `question` → embed → Chroma similarity search → filter by distance → join chunks into `context` → single user message to Claude → answer and deduplicated source filenames.

Sentence-level chunking with **2-sentence overlap** (see `chunk_text` in `main.py`).

**Retrieval filter (unambiguous):** a chunk is **kept for the LLM only if** `distance < 0.5` (the default `distance_threshold` in `query()`). That condition is applied in the loop in `main.py` lines 112–120. The `distances` array comes from `collection.query()` (`main.py` lines 96–105); **smaller values mean closer / more similar** matches. The numeric scale is whatever Chroma returns for this index (treat `0.5` as a **tuned constant** in this project, not a named metric like “cosine distance” unless you confirm Chroma’s configuration for this collection version).

**Default `query()` args:** `n_results=5`, `distance_threshold=0.5` — `main.py` line 94.

## Environment

| Item | Required | Notes |
|------|----------|--------|
| **Python** | — | **3.10+ recommended** (Chroma, SentenceTransformers, and FastAPI are version-sensitive; pin what you test when you add `pyproject` or CI). |
| `ANTHROPIC_API_KEY` | Yes | Read at `main.py` line 30 on module load. If unset, the process fails when creating `anthropic.Anthropic`. |

There is no `.env` loading in code; export the variable in the shell or use a tool that injects env before starting Python.

## Common commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=<YOUR_KEY>

# Run the FastAPI app (on startup, lifespan runs ingest() every time)
uvicorn main:app --reload

# Interactive CLI — see "Ingest and Chroma state" below
python cli.py

# Tests (Chroma and Anthropic are mocked in query tests; no network required)
pytest
```

**Lint / typecheck:** not configured in this repo yet. `requirements.txt` lists `pytest` only as extra tooling.

## Repository layout (main pieces)

| Path | Role |
|------|------|
| `main.py` | `FastAPI` app, `ingest()`, `chunk_text()`, `query()`, module-level Chroma + Anthropic clients, `GET /status` |
| `cli.py` | REPL; imports and calls `query()` from `main` in-process (not HTTP) |
| `data/*.txt` | Source files for ingest |
| `chroma_db/` | Persistent Chroma data (gitignored) |
| `test_main.py` | Pytest: chunking + mocked `query` tests |

## Design decisions (short)

- **No `POST` /ask HTTP endpoint (deliberate for this repo)** — The app is **CLI-first and local demo shaped**: the interactive path is `python cli.py` calling `query()` in process. The FastAPI surface only exposes **`GET /status`** (see `main.py` lines 176–183). **This is not an accidental omission**; adding a public question endpoint would also need input validation, auth, and CORS in a real deployment. If you add one, do it as an explicit product decision, not a “bugfix.”

- **Ingest and Chroma state**  
  - **FastAPI / Uvicorn:** `lifespan` always calls `ingest()` on startup (`main.py` lines 84–88, 49–80).  
  - **Ingest implementation:** for each `data/*.txt` file, `ingest()` builds chunk ids as `"{file_stem}_{i}"` and uses **`collection.upsert(...)`** (`main.py` lines 68–74). The **same id is overwritten** on the next run, so **updated** chunks for indices `0..n-1` stay fresh. There is **no** `delete` of ids that are **no longer produced** (e.g. you shorten a file and the old `filestem_8` id would **remain** in the store). Stale extra chunks are possible; **a full reset** is: stop the server, **delete the `chroma_db/`** directory, restart (or re-run a clean ingest from CLI when the collection is empty).  
  - **`cli.py`:** runs `ingest()` **only if** `collection.count() == 0` (`cli.py` lines 19–21). If a previous ingest **partially** failed, you can end up in an inconsistent state; there is no built-in “repair” or “force re-ingest” flag — use **`chroma_db/` deletion** or extend the app with an explicit `delete_collection` / full replace path.

- **Model IDs in code (intentional, not a bug to “fix” by default)**  
  - **Claude** `model` is a **string literal in `query()`** — `main.py` line **156** (currently `claude-sonnet-4-20250514`). It is **intentionally pinned in source for reproducibility**; when you upgrade, **bump the string in code and test**, and **verify the identifier** against the current [Anthropic model list](https://docs.anthropic.com/en/docs/about-claude/models) (ids change over time). **Not env-driven** in this project to avoid “works on my machine” drift unless you add that as a product requirement.  
  - **Embeddings** use `model_name="mixedbread-ai/mxbai-embed-large-v1"` in `main.py` lines 21–22 — same “pinned in code” idea.

- **Local embeddings** — no embedding API; first run downloads the SentenceTransformers model.

## Testing

- `pytest` from repo root; `test_main.py` mocks `main.collection` and `main.anthropic_client` for `query` tests.
- For changes to `ingest` or Chroma setup, add or run integration checks manually (not automated here).

## README alignment

- README `Running locally` should include setting `ANTHROPIC_API_KEY` (easy to miss).
- README mentions `requests` for the CLI; `cli.py` does not use HTTP—call `query()` directly. Treat README as slightly stale until updated.

## Known limitations (with code pointers)

- **No token budget on retrieved text** — After filtering, all remaining chunks are joined with no size cap. **Where to change first:** `context = "\n\n---\n\n".join(chunks)` in `main.py` line **138**; prompt assembly immediately after at lines **140–150**. Together with unbounded `question`, this is the main “large prompt” footgun.
- **No metadata pre-filter, query rewriting, or retrieval diversity** (see project `README.md`); the retrieval loop and threshold are `main.py` **94–135**.

When editing, keep changes focused; match existing style in `main.py` and `cli.py`.
