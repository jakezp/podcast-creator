import os
import pytest

from podcast_creator.core import normalize_tts_text


@pytest.fixture(autouse=True)
def _ensure_defaults(monkeypatch):
    # Ensure all toggles are on by default for these tests
    monkeypatch.setenv("PODCAST_CREATOR_TTS_NORMALIZE", "1")
    monkeypatch.setenv("PODCAST_CREATOR_TTS_ENSURE_PUNCT", "1")
    monkeypatch.setenv("PODCAST_CREATOR_TTS_LONGER_COMMA_PAUSE", "1")
    monkeypatch.setenv("PODCAST_CREATOR_TTS_EMPHASIS", "1")


def test_dash_and_smart_punct_normalization():
    s = "Hello — YouTube - really--yes\u2013maybe."
    out = normalize_tts_text(s)
    # Em/en dashes and spaced hyphens should become commas/semicolons (since comma -> semicolon)
    assert ";" in out  # at least one comma became semicolon
    assert "—" not in out and "–" not in out and "--" not in out


def test_quotes_and_ellipsis_and_controls():
    s = "“Smart” ‘quotes’ and ellipsis…\u200b"
    out = normalize_tts_text(s)
    assert '"Smart"' in out
    assert "'quotes'" in out
    assert "..." in out
    assert "\u200b" not in out


def test_ensure_terminal_punct():
    s = "This needs punctuation"
    out = normalize_tts_text(s)
    assert out.endswith(".")


def test_season_episode_zero_stripping():
    s = "Season 02, Episode 05"
    out = normalize_tts_text(s)
    assert "Season 2, Episode 5" in out


def test_sxxexx_expansion_and_protected_hyphens():
    s = "We cover S02E05 and also S2E5 and even S 03 E 07."
    out = normalize_tts_text(s)
    # Expect S-0-2-E-0-5 etc.
    assert "S-0-2-E-0-5" in out
    assert "S-2-E-5" in out
    assert "S-0-3-E-0-7" in out


def test_longer_comma_pause_not_in_numbers(monkeypatch):
    monkeypatch.setenv("PODCAST_CREATOR_TTS_LONGER_COMMA_PAUSE", "1")
    s = "Values are 1,000 and 2, 3, and 4"
    out = normalize_tts_text(s)
    # The thousands separator should remain a comma, but list commas become semicolons
    assert "1,000" in out
    assert "; 3; and 4" in out or "; 3 and 4" in out or "; 3; and 4." in out


def test_emphasis_doubling():
    s = "Wow! Really? OK."
    out = normalize_tts_text(s)
    assert "Wow!!" in out
    assert "Really??" in out
