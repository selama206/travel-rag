# Overview
A local RAG pipeline for querying personal travel itineraries, built with [ChromaDB](https://www.trychroma.com/), [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1), and [Claude](https://www.anthropic.com/claude) API. 
ChromaDB was chosen for its zero-infrastructure setup with no server required and no external dependencies. mxbai-embed-large-v1 was selected as the embedding model for its large parameter count (335M) producing richer semantic representations, open source access, and compatibility with ChromaDB. Claude LLM API was selected after comparing output against OpenAI's LLM API and finding Claude's output more natural-sounding for travel recommendations.

Text is chunked at the sentence level with a 2-sentence overlap to preserve context across segments while maintaining retrieval precision. Retrieved results are filtered using a distance threshold of 0.5 to increase semantically relevant matches in the final prompt.

For transparency and debugging, all retrieved chunks are logged with their associated distance score and labeled as either RELEVANT (included in the prompt, less than 0.5 threshold) or FILTERED (excluded, greater than 0.5 distance threshold).

Travel recommendation prompt was developed following Claude's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices).

## Architecture
```
question → embed (mxbai-embed-large-v1) → similarity search (ChromaDB) → ranked chunks → Claude → answer

```


## Known limitations and future improvements
- **Metadata filtering** — all chunks are searched regardless of relevance. Trip-level metadata (country, city, year, themes) extracted at ingest time would allow ChromaDB to pre-filter before running similarity search, improving both precision and latency.
- **Query rewriting** — open-ended questions ("recommend cities I'd enjoy") have weak embedding signal and retrieve poorly. A query rewriting step would significantly improve retrieval on broad questions.
- **Retrieval diversity** - The current implementation lacks a retrieval diversity strategy, meaning results can be dominated by chunks from a single document, which limits the breadth of context passed to Claude.


## Sample Run

<img width="1366" height="716" alt="Screenshot 2026-03-16 at 8 19 13 PM" src="https://github.com/user-attachments/assets/6c8a3ced-2826-4a3c-bb35-0b6287755e90" />



## Project structure
```
Travel-rag/
├── chroma_db/          # ChromaDB vector store for embeddings
├── data/               # Raw travel data and documents
├── venv/               # Python virtual environment
├── cli.py              # Command-line interface entry point
├── main.py             # Core application logic and RAG pipeline
├── test_main.py        # Unit and integration tests
├── requirements.txt    # Python dependencies
└── Travel-rag.iml      # IntelliJ/PyCharm project module file
```

## Running locally
```bash
pip install -r requirements.txt
Terminal 1: uvicorn main:app --reload
Terminal 2: python3 cli.py
```
