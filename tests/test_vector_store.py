import pytest

from codebase_agent.rag.chunker import Chunk
from codebase_agent.rag.vector_store import InMemoryVectorStore, cosine_similarity, search_code


class FakeEmbeddingProvider:
    def __init__(self, vectors_by_text):
        self.vectors_by_text = vectors_by_text
        self.calls = []

    def embed_texts(self, texts):
        self.calls.append(list(texts))
        return [self.vectors_by_text[text] for text in texts]


def make_chunk(chunk_id, text, symbol_name="handler"):
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        file_path="app.py",
        start_line=1,
        end_line=3,
        symbol_name=symbol_name,
    )


def test_cosine_similarity_scores_direction_similarity():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vector():
    assert cosine_similarity([0, 0], [1, 0]) == 0.0


def test_cosine_similarity_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="维度必须一致"):
        cosine_similarity([1, 0], [1])

    with pytest.raises(ValueError, match="不能为空"):
        cosine_similarity([], [1])

    with pytest.raises(ValueError, match="只能包含数字"):
        cosine_similarity([1, "bad"], [1, 0])


def test_store_returns_top_k_sorted_by_score_and_keeps_metadata():
    chunks = [
        make_chunk("chunk-api", "@app.get('/health')\ndef health(): pass", "health"),
        make_chunk("chunk-db", "def save_user(): pass", "save_user"),
        make_chunk("chunk-agent", "def route_tool(): pass", "route_tool"),
    ]
    store = InMemoryVectorStore()
    store.add_chunks(
        chunks,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
        ],
    )

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

    assert [result.metadata["chunk_id"] for result in results] == ["chunk-api", "chunk-agent"]
    assert results[0].text.startswith("@app.get")
    assert results[0].metadata["file_path"] == "app.py"
    assert results[0].metadata["symbol_name"] == "health"
    assert results[0].score >= results[1].score


def test_store_boundaries():
    store = InMemoryVectorStore()

    assert store.search([1.0, 0.0], top_k=3) == []

    chunk = make_chunk("chunk-api", "@app.post('/items')\ndef create_item(): pass")
    store.add_chunks([chunk], [[1.0, 0.0]])

    assert len(store) == 1
    assert store.search([1.0, 0.0], top_k=0) == []
    assert len(store.search([1.0, 0.0], top_k=99)) == 1


def test_store_rejects_mismatched_chunks_and_vectors():
    store = InMemoryVectorStore()

    with pytest.raises(ValueError, match="数量必须一致"):
        store.add_chunks([make_chunk("a", "def a(): pass")], [])


def test_store_rejects_empty_chunk_text():
    store = InMemoryVectorStore()
    chunk = make_chunk("empty", "   ")

    with pytest.raises(ValueError, match="chunk 文本"):
        store.add_chunks([chunk], [[1.0]])


def test_search_code_embeds_chunks_and_query_then_returns_ranked_results():
    chunks = [
        make_chunk("chunk-route", "@app.get('/users')\ndef list_users(): pass", "list_users"),
        make_chunk("chunk-db", "def save_user_to_database(): pass", "save_user_to_database"),
    ]
    provider = FakeEmbeddingProvider(
        {
            chunks[0].text: [1.0, 0.0],
            chunks[1].text: [0.0, 1.0],
            "FastAPI route": [1.0, 0.0],
        }
    )

    results = search_code(
        query="FastAPI route",
        chunks=chunks,
        top_k=1,
        embedding_provider=provider,
    )

    assert provider.calls == [[chunks[0].text, chunks[1].text], ["FastAPI route"]]
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "chunk-route"
    assert results[0].score == pytest.approx(1.0)


def test_search_code_rejects_empty_query():
    provider = FakeEmbeddingProvider({})

    with pytest.raises(ValueError, match="query 不能为空"):
        search_code("   ", [], 3, provider)


def test_search_code_returns_empty_without_embedding_when_no_chunks_or_top_k_zero():
    provider = FakeEmbeddingProvider({})

    assert search_code("query", [], 3, provider) == []
    assert search_code("query", [make_chunk("a", "def a(): pass")], 0, provider) == []
    assert provider.calls == []
