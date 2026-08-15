from fastapi.testclient import TestClient

from codebase_agent.backend.main import app


client = TestClient(app)


def test_code_graph_api_returns_nodes_edges_and_summary():
    response = client.post(
        "/code-graph",
        json={
            "files": [
                {
                    "file_path": "pkg/main.py",
                    "content": "from pkg.worker import run\n\ndef start():\n    return run()\n",
                },
                {
                    "file_path": "pkg/worker.py",
                    "content": "def run():\n    return 'ok'\n",
                },
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["files"] == 2
    assert data["summary"]["supported_language_files"] == 2
    assert data["summary"]["internal_imports"] == 1
    assert data["summary"]["external_imports"] == 0
    assert any(node["label"] == "start" for node in data["nodes"])
    assert any(
        edge["type"] == "imports" and edge["target"] == "file:pkg/worker.py"
        for edge in data["edges"]
    )


def test_code_graph_api_rejects_empty_files():
    response = client.post("/code-graph", json={"files": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "请提供要分析的代码文件"
