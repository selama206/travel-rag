import pytest
from unittest.mock import MagicMock, patch

from main import chunk_text

def test_chunk_text_basic():
    text = "I went to Paris. It was amazing. The food was great."
    chunks = chunk_text(text)
    assert len(chunks) == 3
    assert chunks[0] == "I went to Paris. It was amazing. The food was great."

def test_chunk_text_overlap():
    """Each chunk should include the next 2 sentences for context."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text(text)
    # First chunk should include sentences 1, 2, 3
    assert "Sentence one." in chunks[0]
    assert "Sentence two." in chunks[0]
    assert "Sentence three." in chunks[0]
    assert "Sentence four." not in chunks[0]

def test_chunk_text_single_sentence():
    text = "Just one sentence."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Just one sentence."

def test_chunk_text_empty_string():
    chunks = chunk_text("")
    assert chunks == []


# ── query tests ───────────────────────────────────────────────────────────────

from main import query


@patch("main.collection")
@patch("main.anthropic_client")
def test_query_returns_answer_and_sources(mock_anthropic, mock_collection):
    mock_collection.query.return_value = {
        "documents": [["I visited Tokyo in April. The cherry blossoms were beautiful."]],
        "metadatas": [[{"source": "japan_trip.txt"}]],
        "distances": [[0.2]],
    }

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="You should visit Tokyo in spring.")]
    mock_anthropic.messages.create.return_value = mock_message

    result = query("Where should I travel?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "You should visit Tokyo in spring."
    assert "japan_trip.txt" in result["sources"]


@patch("main.collection")
@patch("main.anthropic_client")
def test_query_filters_high_distance_chunks(mock_anthropic, mock_collection):
    mock_collection.query.return_value = {
        "documents": [["Relevant chunk.", "Irrelevant chunk."]],
        "metadatas": [[{"source": "trip1.txt"}, {"source": "trip2.txt"}]],
        "distances": [[0.2, 0.8]],  # second chunk should be filtered out
    }

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="Here is an answer.")]
    mock_anthropic.messages.create.return_value = mock_message

    result = query("What did I do?")

    # Only trip1.txt should be in sources (trip2.txt was filtered)
    assert result["sources"] == ["trip1.txt"]


@patch("main.collection")
def test_query_raises_when_no_relevant_chunks(mock_collection):
    mock_collection.query.return_value = {
        "documents": [["Not relevant at all."]],
        "metadatas": [[{"source": "trip.txt"}]],
        "distances": [[0.99]],  # above threshold, will be filtered
    }

    with pytest.raises(Exception, match="no relevant chunks found"):
        query("Where did I go?")


@patch("main.collection")
@patch("main.anthropic_client")
def test_query_raises_on_auth_error(mock_anthropic, mock_collection):
    import anthropic as anthropic_lib

    mock_collection.query.return_value = {
        "documents": [["Some relevant chunk."]],
        "metadatas": [[{"source": "trip.txt"}]],
        "distances": [[0.1]],
    }

    mock_anthropic.messages.create.side_effect = anthropic_lib.AuthenticationError(
        message="invalid key", response=MagicMock(status_code=401), body={}
    )

    with pytest.raises(Exception, match="invalid Anthropic API key"):
        query("Where did I go?")


@patch("main.collection")
@patch("main.anthropic_client")
def test_query_raises_on_rate_limit(mock_anthropic, mock_collection):
    import anthropic as anthropic_lib

    mock_collection.query.return_value = {
        "documents": [["Some relevant chunk."]],
        "metadatas": [[{"source": "trip.txt"}]],
        "distances": [[0.1]],
    }

    mock_anthropic.messages.create.side_effect = anthropic_lib.RateLimitError(
        message="rate limit", response=MagicMock(status_code=429), body={}
    )

    with pytest.raises(Exception, match="rate limit"):
        query("Where did I go?")