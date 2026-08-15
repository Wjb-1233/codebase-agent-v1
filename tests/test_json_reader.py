"""Tests for loading repository summary JSON files."""
import json

import pytest

from scripts.json_reader import load_repo_summary


def test_load_repo_summary_normal(tmp_path):
    """Return parsed repository summary data for a valid JSON file."""
    data = {
        "repo_url": "https://github.com/test/repo",
        "total_files": 3,
        "files": [
            {"path": "main.py", "type": ".py"},
            {"path": "README.md", "type": ".md"},
            {"path": "utils.py", "type": ".py"},
        ],
    }
    json_file = tmp_path / "test_repo.json"

    with open(json_file, "w") as f:
        json.dump(data, f)

    result = load_repo_summary(str(json_file))

    assert result["repo_url"] == data["repo_url"]
    assert result["total_files"] == data["total_files"]
    py_files = [f for f in result["files"] if f["type"] == ".py"]
    md_files = [f for f in result["files"] if f["type"] == ".md"]
    assert len(py_files) == 2
    assert len(md_files) == 1


def test_load_repo_summary_file_not_found():
    """Raise FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_repo_summary("nonexistent.json")


def test_load_repo_summary_bad_json(tmp_path):
    """Raise JSONDecodeError for invalid JSON content."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{this is not json")

    with pytest.raises(json.JSONDecodeError):
        load_repo_summary(str(bad_file))


def test_load_repo_summary_zero_files(tmp_path):
    """Handle empty repositories without crashing."""
    data = {
        "repo_url": "https://github.com/empty/repo",
        "total_files": 0,
        "files": [],
    }
    json_file = tmp_path / "empty.json"

    with open(json_file, "w") as f:
        json.dump(data, f)

    result = load_repo_summary(str(json_file))

    assert result["total_files"] == 0
    assert result["files"] == []
