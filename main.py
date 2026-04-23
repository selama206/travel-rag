import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "travel_itineraries"

# ── App setup ─────────────────────────────────────────────────────────────────

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="mixedbread-ai/mxbai-embed-large-v1"
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def chunk_text(text: str, overlap: int = 2) -> list[str]:
    """
    Splits text into sentence-level chunks with overlapping context.
    Each chunk includes the next 2 sentences so retrieved
    text always has surrounding context.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(len(sentences)):
        chunk = " ".join(sentences[i: i + 1 + overlap])
        chunks.append(chunk)

    return chunks


def ingest():
    """
    reads all .txt files from the data/ folder, chunks them,
    embeds each chunk, and stores them in chromadb.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError("data/ folder not found — create it and add your .txt files")

    txt_files = list(DATA_DIR.glob("*.txt"))

    if not txt_files:
        raise ValueError("no .txt files found in data/ folder")

    total_chunks = 0

    for file_path in txt_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.stem}_{i}"
            collection.upsert(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[{"source": file_path.name, "chunk_index": i}],
            )
            total_chunks += 1

    return {
        "status": "ingested",
        "files_processed": len(txt_files),
        "chunks_stored": total_chunks,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: ingest .txt files into chromadb before the app begins accepting requests
    ingest()
    yield


app = FastAPI(title="Travel RAG API", lifespan=lifespan)


def query(question: str, n_results: int = 5, distance_threshold: float = 0.5):
    # Query the vector DB for the most similar chunks to the question
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Unpack the results
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    relevant = []
    filtered = []

    # Sort chunks into relevant/filtered based on similarity to the question
    # Lower distance = more similar; filter out anything above the threshold
    for i in range(len(chunks)):
        chunk = chunks[i]
        source = sources[i]
        distance = distances[i]

        if distance < distance_threshold:
            relevant.append((chunk, source, distance))
        else:
            filtered.append((chunk, source, distance))

    # Print filtered chunks so you can see what was deemed not relevant enough
    for i in range(len(filtered)):
        chunk, source, distance = filtered[i]
        print(f"\nFILTERED (distance={distance:.3f}) [{source}]:\n{chunk.split('.')[0]}...\n")

    if not relevant:
        raise Exception("no relevant chunks found, have you ingested your files?")

    for i in range(len(relevant)):
        chunk, source, distance = relevant[i]
        print(f"\nRELEVANT (distance={distance:.3f}) [{source}]:\n{chunk.split('.')[0]}...\n")

    chunks = [chunk for chunk, source, distance in relevant]
    sources = [source for chunk, source, distance in relevant]

    # Join chunks into a single context block, separated by dividers
    context = "\n\n---\n\n".join(chunks)

    # Build the prompt with the relevant context and the user's question
    prompt = f"""You are a helpful travel assistant. 
The context below contains the user's past travel itineraries.
Use this context to understand their travel history and preferences, 
then answer their question using both the context and your own travel knowledge.
If the context doesn't contain enough information just use your own knowledge.

Past travel itineraries:
{context}

Question: {question}
"""

    # Send the prompt to Claude and get a response
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise Exception("invalid Anthropic API key — check your ANTHROPIC_API_KEY environment variable")
    except anthropic.RateLimitError:
        raise Exception("Anthropic rate limit reached — please wait and try again")
    except anthropic.APIConnectionError:
        raise Exception("could not connect to Anthropic API — check your internet connection")
    except anthropic.APIStatusError as e:
        raise Exception(f"Anthropic API error {e.status_code}: {e.message}")

    u = message.usage
    print(f"Token usage input: {u.input_tokens}, output: {u.output_tokens}")

    return {
        "answer": message.content[0].text,
        # Deduplicate sources in case multiple chunks came from the same file
        "sources": list(set(sources)),
    }


@app.get("/status")
def status():
    count = collection.count()

    return {
        "ingested": count > 0,
        "chunk_count": count,
    }
