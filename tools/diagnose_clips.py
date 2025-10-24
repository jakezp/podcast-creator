#!/usr/bin/env python
"""
Quick diagnostics for two audio clips.

Reports duration and tail-window RMS to help detect truncated endings.

Usage:
  python tools/diagnose_clips.py path/to/clip1.mp3 path/to/clip2.mp3 \
      --tail-ms 200

This script prefers pydub for RMS and duration, and will auto-configure
ffmpeg using imageio-ffmpeg if a system ffmpeg is not installed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from loguru import logger
import numpy as np
from moviepy import AudioFileClip


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    # Ensure float64
    s = samples.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(s))))


def analyze_clip(path: Path, tail_ms: int = 200) -> dict:
    clip = AudioFileClip(str(path))
    try:
        dur_s = float(clip.duration or 0.0)
        dur_ms = int(round(dur_s * 1000))
        window_ms = max(min(tail_ms, dur_ms), 1)

        # Read whole array (clips are typically short)
        arr = clip.to_soundarray(fps=clip.fps)
        if arr.ndim == 2:
            # mixdown to mono for RMS
            arr_mono = arr.mean(axis=1)
        else:
            arr_mono = arr
        overall = _rms(arr_mono)

        # Tail window via array slicing
        sr = int(clip.fps)
        total_samples = arr_mono.shape[0]
        tail_samples = max(int(round((window_ms / 1000.0) * sr)), 1)
        tail_start = max(total_samples - tail_samples, 0)
        tail_arr = arr_mono[tail_start:]
        tail_r = _rms(tail_arr)

        # Fine 20ms window
        fine_ms = min(20, dur_ms)
        fine_samples = max(int(round((fine_ms / 1000.0) * sr)), 1)
        fine_start = max(total_samples - fine_samples, 0)
        fine_arr = arr_mono[fine_start:]
        fine_r = _rms(fine_arr)

        overall_safe = overall if overall > 0 else 1.0
        return {
            "duration_ms": dur_ms,
            "tail_ms": window_ms,
            "overall_rms": overall,
            "tail_rms": tail_r,
            "tail_ratio": (tail_r / overall_safe),
            "fine_ms": fine_ms,
            "fine_rms": fine_r,
            "fine_ratio": (fine_r / overall_safe),
        }
    finally:
        try:
            clip.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose two audio clips")
    parser.add_argument("clip1", type=str, help="Path to first clip")
    parser.add_argument("clip2", type=str, help="Path to second clip")
    parser.add_argument(
        "--tail-ms",
        type=int,
        default=int(os.getenv("PODCAST_CREATOR_TAIL_DIAG_MS", "200")),
        help="Tail window (ms) for RMS analysis (default: 200)",
    )
    args = parser.parse_args(argv)

    p1 = Path(args.clip1)
    p2 = Path(args.clip2)
    if not p1.exists() or not p2.exists():
        logger.error("One or both clip paths do not exist.")
        return 2

    r1 = analyze_clip(p1, args.tail_ms)
    r2 = analyze_clip(p2, args.tail_ms)

    def fmt(label: str, r: dict) -> None:
        dur_s = r["duration_ms"] / 1000.0
        print(
            f"{label}: duration={dur_s:.3f}s, tail_ms={r['tail_ms']} | "
            f"tail_rms={r['tail_rms']} overall_rms={r['overall_rms']} "
            f"tail_ratio={r['tail_ratio']:.3f} | fine_ms={r['fine_ms']} fine_ratio={r['fine_ratio']:.3f}"
        )

    fmt("Clip 1", r1)
    fmt("Clip 2", r2)

    # Simple heuristic: a very low tail_ratio could indicate truncation/sudden silence
    warn_thresh = 0.05
    for idx, rr in enumerate([r1, r2], start=1):
        if rr["tail_ratio"] < warn_thresh and rr["fine_ratio"] < warn_thresh:
            logger.warning(
                f"Clip {idx} tail energy is very low; potential truncation or early silence."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
