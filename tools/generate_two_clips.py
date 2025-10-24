#!/usr/bin/env python
"""
Generate two TTS clips from a transcript JSON using a chosen provider/model.

Usage:
    python tools/generate_two_clips.py \
            --file test_clips/transcript.json \
            --indices 3 4 \
            --out-dir test_clips/generated \
            [--voice1 am_onyx --voice2 af_heart]

Environment (defaults if unset):
    TTS_PROVIDER=openai | elevenlabs | google | speaches
    TTS_MODEL=tts-1 (or provider-specific)
    SPEAKER_VOICE_DEFAULT=alloy
    SPEAKER_VOICE_<SANITIZED_NAME>=<voice_id>  # e.g., SPEAKER_VOICE_DR_ALEX_CHEN

    # Speaches (local server) specific:
    SPEACHES_BASE_URL=http://localhost:8969
    SPEACHES_TTS_PATH=/v1/audio/speech   # OpenAI-compatible default
    SPEACHES_FORMAT=mp3                  # mp3 or wav

Notes:
    - Cloud providers require proper API keys (e.g., OPENAI_API_KEY).
    - For speaches (local), no API key is required; the server must be running.
    - Saves outputs as 0001.<ext> and 0002.<ext> in the given out dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

from loguru import logger
from esperanto.factory import AIFactory
import requests


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").upper()


def pick_voice_for_speaker(speaker: str, default_voice: str) -> str:
    env_key = f"SPEAKER_VOICE_{sanitize_name(speaker)}"
    return os.getenv(env_key, default_voice)


async def generate_clip(text: str, voice: str, out_path: Path, provider: str, model: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating clip → {out_path.name} | provider={provider} model={model} voice={voice}")

    if provider == "speaches":
        base_url = os.getenv("SPEACHES_BASE_URL", "http://localhost:8969")
        tts_path = os.getenv("SPEACHES_TTS_PATH", "/v1/audio/speech")
        fmt = os.getenv("SPEACHES_FORMAT", out_path.suffix.lstrip(".") or "mp3").lower()

        url = urljoin(base_url.rstrip("/") + "/", tts_path.lstrip("/"))
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "format": fmt,
        }
        headers = {"Accept": "audio/mpeg" if fmt == "mp3" else "*/*"}
        try:
            resp = requests.post(url, json=payload, timeout=120)
        except Exception as e:
            raise RuntimeError(f"Failed to reach speaches server at {url}: {e}")

        if resp.status_code != 200:
            raise RuntimeError(
                f"Speaches TTS request failed: HTTP {resp.status_code} | {resp.text[:300]}"
            )

        # Save audio bytes
        out_path.write_bytes(resp.content)
        logger.info(f"Wrote clip: {out_path}")
        return

    # Default path via Esperanto providers
    tts = AIFactory.create_text_to_speech(provider, model)
    await tts.agenerate_speech(text=text, voice=voice, output_file=out_path)
    logger.info(f"Wrote clip: {out_path}")


async def amain(args: argparse.Namespace) -> int:
    # Load transcript
    transcript_path = Path(args.file)
    data = json.loads(transcript_path.read_text())

    # Indices to use
    if len(args.indices) != 2:
        logger.error("Please provide exactly two indices.")
        return 2

    idx1, idx2 = args.indices
    try:
        item1 = data[idx1]
        item2 = data[idx2]
    except Exception as e:
        logger.error(f"Index error accessing transcript items: {e}")
        return 2

    speaker1 = item1.get("speaker", "Speaker1")
    speaker2 = item2.get("speaker", "Speaker2")
    text1 = item1.get("dialogue", "")
    text2 = item2.get("dialogue", "")
    if not text1 or not text2:
        logger.error("Missing dialogue text for one or both items.")
        return 2

    # Provider/model defaults
    provider = os.getenv("TTS_PROVIDER", "openai").lower()
    model = os.getenv("TTS_MODEL", "tts-1")

    # Basic provider credential checks to prevent confusing runtime errors
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Export it or set TTS_PROVIDER to a configured provider.")
        return 2
    if provider == "elevenlabs" and not os.getenv("ELEVENLABS_API_KEY"):
        logger.error("ELEVENLABS_API_KEY is not set. Export it or set TTS_PROVIDER to a configured provider.")
        return 2
    # Speaches is local – no key check
    default_voice = os.getenv("SPEAKER_VOICE_DEFAULT", "alloy")

    # Allow CLI overrides for voices (useful for testing different voices quickly)
    if args.voice1:
        voice1 = args.voice1
    else:
        voice1 = pick_voice_for_speaker(speaker1, default_voice)

    if args.voice2:
        voice2 = args.voice2
    else:
        voice2 = pick_voice_for_speaker(speaker2, default_voice)

    out_dir = Path(args.out_dir)
    # Choose extension based on provider/format
    ext = os.getenv("SPEACHES_FORMAT", "mp3") if provider == "speaches" else "mp3"
    out1 = out_dir / f"0001.{ext}"
    out2 = out_dir / f"0002.{ext}"

    # Generate sequentially to keep logs clear
    await generate_clip(text1, voice1, out1, provider, model)
    await generate_clip(text2, voice2, out2, provider, model)

    # Summary
    logger.info("Generation complete.")
    print(json.dumps({
        "clips": [str(out1), str(out2)],
        "provider": provider,
        "model": model,
        "voices": {speaker1: voice1, speaker2: voice2}
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate two TTS clips from transcript")
    p.add_argument("--file", type=str, default="test_clips/transcript.json", help="Transcript JSON path")
    p.add_argument("--indices", nargs=2, type=int, default=[3, 4], help="Two item indices to generate")
    p.add_argument("--out-dir", type=str, default="test_clips/generated", help="Output directory")
    p.add_argument("--voice1", type=str, default="", help="Override voice for the first selected item")
    p.add_argument("--voice2", type=str, default="", help="Override voice for the second selected item")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
