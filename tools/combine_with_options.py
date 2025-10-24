#!/usr/bin/env python
"""
Combine two audio clips with controllable padding, fades, fps, and optional micro-crossfade.

Usage:
  python tools/combine_with_options.py in1.mp3 in2.mp3 out.mp3 \
      --pad-sec 0.2 --fade-ms 10 --fps 44100 --crossfade-ms 0

Defaults mirror the library's environment defaults.
This script uses moviepy (which leverages imageio-ffmpeg) and does not
require a system ffmpeg installation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger
from typing import cast
from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
from moviepy.audio.fx.AudioFadeOut import AudioFadeOut


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Combine two audio clips with options")
    parser.add_argument("clip1", type=str)
    parser.add_argument("clip2", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--pad-sec", type=float, default=float(os.getenv("PODCAST_CREATOR_CLIP_PAD_SEC", "0.2")))
    parser.add_argument("--fade-ms", type=float, default=float(os.getenv("PODCAST_CREATOR_FADE_MS", "10")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("PODCAST_CREATOR_AUDIO_FPS", "44100")))
    parser.add_argument("--crossfade-ms", type=float, default=float(os.getenv("PODCAST_CREATOR_CROSSFADE_MS", "0")))
    parser.add_argument("--normalize", action="store_true", help="Decode inputs to WAV via ffmpeg before combining (avoids bad MP3/VBR durations)")
    args = parser.parse_args(argv)

    pad_sec = max(args.pad_sec, 0.0)
    fade_sec = max(args.fade_ms, 0.0) / 1000.0
    crossfade_sec = max(args.crossfade_ms, 0.0) / 1000.0
    fps = args.fps

    p1 = Path(args.clip1)
    p2 = Path(args.clip2)
    if not p1.exists() or not p2.exists():
        logger.error("One or both input files do not exist")
        return 2

    logger.info(f"Combine settings: pad={pad_sec:.3f}s, fade={fade_sec:.3f}s, crossfade={crossfade_sec:.3f}s, fps={fps}")

    # Optional normalization to WAV for reliable duration
    # Prefer bundled ffmpeg from imageio-ffmpeg; fallback to system ffmpeg
    ffmpeg_path = None
    if args.normalize:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = shutil.which("ffmpeg")
    tmpdir_ctx = tempfile.TemporaryDirectory() if ffmpeg_path else None
    tmpdir = Path(tmpdir_ctx.name) if tmpdir_ctx else None

    if args.normalize:
        if ffmpeg_path:
            logger.info(f"Normalization enabled; using ffmpeg at {ffmpeg_path}")
        else:
            logger.warning("--normalize was set but no ffmpeg executable found. Proceeding without normalization.")

    def normalize(path: Path, idx: int) -> Path:
        if not tmpdir or not ffmpeg_path:
            return path
        out = tmpdir / f"norm_{idx:02d}.wav"
        try:
            ffmpeg_exe = cast(str, ffmpeg_path)
            subprocess.run([
                ffmpeg_exe, "-y", "-i", str(path),
                "-c:a", "pcm_s16le", "-ar", str(fps), str(out)
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"Normalized {path.name} -> {out.name}")
            return out
        except Exception as e:
            logger.warning(f"Normalization failed for {path}: {e}. Using original.")
            return path

    src1 = normalize(p1, 1) if args.normalize else p1
    src2 = normalize(p2, 2) if args.normalize else p2

    clip1 = AudioFileClip(str(src1)).with_fps(fps)
    clip2 = AudioFileClip(str(src2)).with_fps(fps)

    # Boundary fades: fade-out clip1 (if needed), fade-in clip2 (if needed)
    if fade_sec > 0:
        clip1 = clip1.with_effects([AudioFadeOut(fade_sec)])
        clip2 = clip2.with_effects([AudioFadeIn(fade_sec)])

    # Placement
    t = 0.0
    placed = []
    c1 = clip1.with_start(t)
    placed.append(c1)
    end1 = t + c1.duration

    if crossfade_sec > 0:
        # Overlap by crossfade duration
        c2_start = max(end1 - crossfade_sec, 0.0)
    else:
        # No overlap; optional pad between
        c2_start = end1 + pad_sec

    c2 = clip2.with_start(c2_start)
    placed.append(c2)

    final_end = max(end1, c2_start + c2.duration)
    final = CompositeAudioClip(placed).with_duration(final_end)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        if out.suffix.lower() == ".wav":
            # Write WAV (PCM)
            final.write_audiofile(str(out), fps=fps, codec="pcm_s16le")
        else:
            # Default to mp3
            final.write_audiofile(str(out), fps=fps, codec="libmp3lame")
        logger.info(f"Wrote combined audio: {out}")
    finally:
        final.close()
        clip1.close()
        clip2.close()

    # Cleanup temp dir if used
    if tmpdir_ctx:
        try:
            tmpdir_ctx.cleanup()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
