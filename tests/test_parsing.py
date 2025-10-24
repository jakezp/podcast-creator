import json
import pytest

from podcast_creator.core import clean_thinking_content, extract_json_text


def test_clean_thinking_content_removes_block():
    raw = "<think>internal chain</think> {\n  \"ok\": true\n}"
    cleaned = clean_thinking_content(raw)
    assert "<think>" not in cleaned
    assert cleaned.strip().startswith("{")


def test_extract_json_text_from_fenced_json():
    raw = """
Thoughts first
```json
{"a": 1, "b": [2,3]}
```
Trailing words
"""
    text = extract_json_text(raw)
    data = json.loads(text)
    assert data["a"] == 1
    assert data["b"] == [2, 3]


def test_extract_json_text_from_unfenced_with_think():
    raw = """
<think>reasoning...</think>
{"transcript": [{"speaker": "Alice", "dialogue": "Hi"}]}
extra trailing text
"""
    text = extract_json_text(raw)
    obj = json.loads(text)
    assert "transcript" in obj
    assert obj["transcript"][0]["speaker"] == "Alice"


def test_extract_json_text_handles_unclosed_think_prefix():
    raw = """
<think>reasoning without close tag
{"segments": [{"name": "Intro", "description": "d", "size": "short"}]}
"""
    text = extract_json_text(raw)
    obj = json.loads(text)
    assert "segments" in obj
    assert obj["segments"][0]["name"] == "Intro"
