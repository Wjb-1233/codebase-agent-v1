from codebase_agent.rag.chunker import CHUNK_SIZE, chunk_file


def test_chunk_file_splits_top_level_symbols():
    code = "\n".join(
        [
            "import os",
            "",
            "def health():",
            "    return 'ok'",
            "",
            "class Service:",
            "    pass",
        ]
    )

    chunks = chunk_file("app.py", code)

    assert [chunk.symbol_name for chunk in chunks] == ["health", "Service"]
    assert chunks[0].chunk_id == "app.py:def:health:2"
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 5
    assert chunks[1].chunk_id == "app.py:class:Service:5"


def test_chunk_file_ignores_nested_symbols():
    code = "\n".join(
        [
            "def outer():",
            "    def inner():",
            "        return 1",
            "    return inner()",
            "",
            "class NextStep:",
            "    pass",
        ]
    )

    chunks = chunk_file("nested.py", code)

    assert [chunk.symbol_name for chunk in chunks] == ["outer", "NextStep"]


def test_chunk_file_splits_long_symbol_into_child_chunks():
    body = [f"    value_{index} = {index}" for index in range(CHUNK_SIZE * 2 + 5)]
    code = "\n".join(["def big_function():", *body])

    chunks = chunk_file("big.py", code)

    assert len(chunks) == 3
    assert all(chunk.symbol_name == "big_function" for chunk in chunks)
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == CHUNK_SIZE + 1
    assert chunks[2].text.endswith(f"value_{CHUNK_SIZE * 2 + 4} = {CHUNK_SIZE * 2 + 4}")


def test_chunk_file_keeps_module_level_file_without_symbols():
    chunks = chunk_file("settings.py", "API_URL = 'http://example.com'\nTIMEOUT = 5")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "settings.py:module:module:0"
    assert chunks[0].symbol_name == "module"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 2


def test_chunk_file_returns_empty_for_blank_file():
    assert chunk_file("empty.py", "   \n\n") == []