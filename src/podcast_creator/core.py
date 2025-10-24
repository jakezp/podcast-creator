import os
import re
import uuid
from pathlib import Path
import tempfile
import subprocess
import shutil
from typing import Any, Dict, List, Literal, Tuple, Union

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from loguru import logger
from moviepy import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.AudioFadeIn import AudioFadeIn
from moviepy.audio.fx.AudioFadeOut import AudioFadeOut
from pydantic import BaseModel, Field, field_validator

# Compile regex patterns once for better performance
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
JSON_FENCE_PATTERN = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
ANY_FENCE_PATTERN = re.compile(r"```\s*([\s\S]*?)```")


def parse_thinking_content(content: str) -> Tuple[str, str]:
    """
    Parse message content to extract thinking content from <think> tags.

    Args:
        content (str): The original message content

    Returns:
        Tuple[str, str]: (thinking_content, cleaned_content)
            - thinking_content: Content from within <think> tags
            - cleaned_content: Original content with <think> blocks removed

    Example:
        >>> content = "<think>Let me analyze this</think>Here's my answer"
        >>> thinking, cleaned = parse_thinking_content(content)
        >>> print(thinking)
        "Let me analyze this"
        >>> print(cleaned)
        "Here's my answer"
    """
    # Input validation
    if not isinstance(content, str):
        return "", str(content) if content is not None else ""

    # Limit processing for very large content (100KB limit)
    if len(content) > 100000:
        return "", content

    # Find all thinking blocks
    thinking_matches = THINK_PATTERN.findall(content)

    if not thinking_matches:
        return "", content

    # Join all thinking content with double newlines
    thinking_content = "\n\n".join(match.strip() for match in thinking_matches)

    # Remove all <think>...</think> blocks from the original content
    cleaned_content = THINK_PATTERN.sub("", content)

    # Clean up extra whitespace
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content).strip()

    return thinking_content, cleaned_content


def clean_thinking_content(content: str) -> str:
    """
    Remove thinking content from AI responses, returning only the cleaned content.

    This is a convenience function for cases where you only need the cleaned
    content and don't need access to the thinking process.

    Args:
        content (str): The original message content with potential <think> tags

    Returns:
        str: Content with <think> blocks removed and whitespace cleaned

    Example:
        >>> content = "<think>Let me think...</think>Here's the answer"
        >>> clean_thinking_content(content)
        "Here's the answer"
    """
    _, cleaned_content = parse_thinking_content(content)
    return cleaned_content


def extract_json_text(content: str) -> str:
    """
    Extract the most likely JSON payload from an LLM response.

    Handles cases where providers prepend reasoning traces like <think>...</think>,
    or wrap JSON in code fences. Falls back to slicing from the first JSON delimiter
    to the last closing bracket if necessary.

    Args:
        content: Raw model output (potentially with reasoning or markdown)

    Returns:
        A string that should contain only JSON (object or array) content.
    """
    if not isinstance(content, str):
        return str(content) if content is not None else ""

    text = content

    # 1) Remove any complete <think>...</think> blocks (case-insensitive)
    text = THINK_PATTERN.sub("", text)

    # 2) Prefer JSON fenced blocks if present (use the last one if multiple)
    fence_matches = list(JSON_FENCE_PATTERN.finditer(text))
    if fence_matches:
        return fence_matches[-1].group(1).strip()

    # 3) Fall back to any code fence content (often JSON without explicit language)
    any_fence_matches = list(ANY_FENCE_PATTERN.finditer(text))
    if any_fence_matches:
        candidate = any_fence_matches[-1].group(1).strip()
        if candidate:
            return candidate

    # 4) If residual standalone <think> tag exists (unclosed), trim everything before
    # the first likely JSON start delimiter
    # Also robust against leading commentary before JSON
    first_obj = text.find("{")
    first_arr = text.find("[")

    # Choose earliest positive index among object/array starts
    starts = [i for i in [first_obj, first_arr] if i >= 0]
    if starts:
        start = min(starts)
        trimmed = text[start:]
        # 5) Trim trailing noise after the last plausible JSON closer
        last_obj = trimmed.rfind("}")
        last_arr = trimmed.rfind("]")
        end_candidates = [i for i in [last_obj, last_arr] if i >= 0]
        if end_candidates:
            end = max(end_candidates) + 1
            return trimmed[:end].strip()
        return trimmed.strip()

    # Nothing better found; return original (parser may still handle markdown JSON)
    return text.strip()


class Segment(BaseModel):
    name: str = Field(..., description="Name of the segment")
    description: str = Field(..., description="Description of the segment")
    size: Literal["short", "medium", "long"] = Field(
        ..., description="Size of the segment"
    )


class Outline(BaseModel):
    segments: list[Segment] = Field(..., description="List of segments")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        return {"segments": [segment.model_dump(**kwargs) for segment in self.segments]}


class Dialogue(BaseModel):
    speaker: str = Field(..., description="Speaker name")
    dialogue: str = Field(..., description="Dialogue")

    @field_validator("speaker")
    @classmethod
    def validate_speaker_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("Speaker name cannot be empty")
        return v.strip()


class Transcript(BaseModel):
    transcript: list[Dialogue] = Field(..., description="Transcript")

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        # Custom serialization: convert list of Dialogue models to list of dicts
        return {
            "transcript": [
                dialogue.model_dump(**kwargs) for dialogue in self.transcript
            ]
        }


def create_validated_transcript_parser(valid_speaker_names: List[str]):
    """
    Create a transcript parser that validates speaker names against a list of valid names

    Args:
        valid_speaker_names: List of valid speaker names

    Returns:
        PydanticOutputParser: Parser with speaker validation
    """

    class ValidatedDialogue(BaseModel):
        speaker: str = Field(..., description="Speaker name")
        dialogue: str = Field(..., description="Dialogue")

        @field_validator("speaker")
        @classmethod
        def validate_speaker_name(cls, v):
            if not v or len(v.strip()) == 0:
                raise ValueError("Speaker name cannot be empty")

            cleaned_name = v.strip()
            if cleaned_name not in valid_speaker_names:
                raise ValueError(
                    f"Invalid speaker name '{cleaned_name}'. Must be one of: {', '.join(valid_speaker_names)}"
                )

            return cleaned_name

    class ValidatedTranscript(BaseModel):
        transcript: list[ValidatedDialogue] = Field(..., description="Transcript")

        def model_dump(self, **kwargs) -> Dict[str, Any]:
            return {
                "transcript": [
                    dialogue.model_dump(**kwargs) for dialogue in self.transcript
                ]
            }

    return PydanticOutputParser(pydantic_object=ValidatedTranscript)


outline_parser = PydanticOutputParser(pydantic_object=Outline)
transcript_parser = PydanticOutputParser(pydantic_object=Transcript)

def sanitize_jsonish_backslashes(text: str) -> str:
    r"""
    Make loosely JSON-like strings parseable by doubling any single backslashes
    that are not starting a valid JSON escape sequence.

    Example: "Plex\ Media\ Server" (invalid JSON escapes) becomes
    "Plex\\ Media\\ Server" which is valid JSON and preserves the intended
    backslash characters.

    Valid JSON escape starters are: \" \/ \\ \b \f \n \r \t \u
    We only rewrite backslashes not followed by one of these letters/symbols.
    """
    if not isinstance(text, str):
        return text
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)

def _int_or_zero(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 0

def _hyphenate_digits(d: str, placeholder: str) -> str:
    return placeholder.join(list(d)) if d else ""

def normalize_tts_text(text: str) -> str:
    """
    Normalize transcript text before sending to TTS engines.

    Behavior is gated by environment variables (all enabled by default):
    - PODCAST_CREATOR_TTS_NORMALIZE: main normalization pass (default 1)
    - PODCAST_CREATOR_TTS_ENSURE_PUNCT: ensure closing punctuation (default 1)
    - PODCAST_CREATOR_TTS_LONGER_COMMA_PAUSE: lengthen comma pause by using semicolons (default 1)
    - PODCAST_CREATOR_TTS_EMPHASIS: increase emphasis for ! and ? by doubling them (default 1)

    Transformations (when enabled):
    - Replace em/en dashes and spaced hyphens with commas
    - Convert smart quotes and ellipsis to ASCII
    - Remove stray control characters and zero-width spaces
    - Collapse whitespace to single spaces
    - Season/Episode: remove leading zeros (e.g., "Season 02, Episode 05" -> "Season 2, Episode 5")
    - SxxExx patterns (e.g., S02E05) expanded to S-0-2-E-0-5 to avoid "sexex" pronunciations
    - Make comma pauses slightly longer (comma -> semicolon) where safe (not between digits)
    - Add a terminal punctuation mark (., !, ?) if missing
    - Add slight emphasis for ! and ? by doubling them ("!" -> "!!", "?" -> "??")
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    ensure_punct = os.getenv("PODCAST_CREATOR_TTS_ENSURE_PUNCT", "1").lower() not in ("0", "false")
    do_norm = os.getenv("PODCAST_CREATOR_TTS_NORMALIZE", "1").lower() not in ("0", "false")
    longer_comma = os.getenv("PODCAST_CREATOR_TTS_LONGER_COMMA_PAUSE", "1").lower() not in ("0", "false")
    emphasis = os.getenv("PODCAST_CREATOR_TTS_EMPHASIS", "1").lower() not in ("0", "false")

    s = text

    # Protect SxxExx expansions so subsequent dash cleanup doesn't remove the helpful hyphens
    H = "\uFFF0"  # placeholder unlikely to appear in user text

    def expand_sxxexx(match: re.Match) -> str:
        s_num = match.group(1)
        e_num = match.group(2)
        s_digits = _hyphenate_digits(s_num, H)
        e_digits = _hyphenate_digits(e_num, H)
        return f"S{H}{s_digits}{H}E{H}{e_digits}"

    if do_norm:
        # Season 02, Episode 05 -> Season 2, Episode 5
        def season_ep_repl(m: re.Match) -> str:
            s_no = _int_or_zero(m.group(1))
            e_no = _int_or_zero(m.group(2))
            return f"Season {s_no}, Episode {e_no}"

        s = re.sub(r"\bSeason\s+0*(\d+)\s*,\s*Episode\s+0*(\d+)\b", season_ep_repl, s, flags=re.IGNORECASE)

        # Expand SxxExx (e.g., S02E05, S2E5, S 02 E 05)
        s = re.sub(r"\bS\s*0*(\d{1,3})\s*E\s*0*(\d{1,3})\b", expand_sxxexx, s, flags=re.IGNORECASE)
        s = re.sub(r"\bS\s*0*(\d{1,3})\s*[xX]\s*0*(\d{1,3})\b", expand_sxxexx, s)
        s = re.sub(r"\bS0*(\d{1,3})E0*(\d{1,3})\b", expand_sxxexx, s)

        # Normalize smart punctuation -> ASCII
        replacements = {
            "\u201C": '"',  # left double quote
            "\u201D": '"',  # right double quote
            "\u2018": "'",  # left single quote
            "\u2019": "'",  # right single quote
            "\u2026": "...",  # ellipsis
            "\u00A0": " ",  # non-breaking space
            "\u2009": " ",  # thin space
            "\u200A": " ",  # hair space
            "\u200B": "",    # zero-width space
            "\u200C": "",
            "\u200D": "",
            "\u2060": "",
            "\uFEFF": "",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)

        # Replace em/en/figure dashes or spaced hyphens with comma + space
        s = re.sub(r"\s*[\u2014\u2013\u2012]+\s*", ", ", s)  # em/en/figure dash
        s = re.sub(r"\s+-\s+", ", ", s)  # spaced ASCII hyphen
        s = re.sub(r"--+", ", ", s)  # double hyphens

        # For intra-word hyphens, prefer a soft separation (space) instead of comma
        s = re.sub(r"(?<=\w)-(?!\s|$)(?=\w)", " ", s)

        # Remove stray C0/C1 control chars except tab/newline (we collapse later anyway)
        s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]", "", s)

        # Collapse repeated commas/spaces to a single ", "
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r",\s*,+", ", ", s)

    # Lengthen comma pause (not between digits like 1,000)
    if do_norm and longer_comma:
        s = re.sub(r"(?<!\d),(?!\d)", ";", s)

    # Emphasis on ! and ? by doubling single occurrences
    if do_norm and emphasis:
        s = re.sub(r"!+", lambda m: "!!" if len(m.group(0)) == 1 else m.group(0), s)
        s = re.sub(r"\?+", lambda m: "??" if len(m.group(0)) == 1 else m.group(0), s)

    # Collapse all whitespace to single spaces
    if do_norm:
        s = re.sub(r"\s+", " ", s).strip()

    # Ensure terminal punctuation
    if ensure_punct and s and s[-1] not in ".!?":
        s = s + "."

    # Restore protected hyphens in S-*-E-* expansions
    s = s.replace(H, "-")

    return s


def get_outline_prompter():
    """Get outline prompter with configuration support."""
    from .config import ConfigurationManager

    config_manager = ConfigurationManager()
    return config_manager.get_template_prompter("outline", parser=outline_parser)


def get_transcript_prompter():
    """Get transcript prompter with configuration support."""
    from .config import ConfigurationManager

    config_manager = ConfigurationManager()
    return config_manager.get_template_prompter("transcript", parser=transcript_parser)


# Legacy exports for backward compatibility
outline_prompt = get_outline_prompter()
transcript_prompt = get_transcript_prompter()

# Legacy functions removed - use create_podcast from graph.py instead


async def combine_audio_files(
    audio_dir: Union[Path, str], final_filename: str, final_output_dir: Union[Path, str]
):
    """
    Combines multiple audio files into a single MP3 file using moviepy with timeline placement.
    
    Uses CompositeAudioClip to place each clip at explicit start times with configurable padding
    and fade effects to prevent audio truncation at clip boundaries.
    
    Configuration (via environment variables):
    - PODCAST_CREATOR_CLIP_PAD_SEC: Padding between clips in seconds (default: 0.2)
    - PODCAST_CREATOR_FADE_MS: Fade in/out duration in milliseconds (default: 10)
    - PODCAST_CREATOR_AUDIO_FPS: Target sample rate/fps (default: 44100)
    
    Example input: {
        "audio_segments_data": ["path/to/audio1.mp3", "path/to/audio2.mp3"],
        "final_filename": "my_podcast.mp3"
    }
    Output: {"combined_audio_path": "output/audio/my_podcast.mp3"}
    """
    logger.info("[Core Function] combine_audio_files called.")
    
    # Load .env if present so users can configure without exporting
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
    except Exception as e:
        logger.debug(f".env loading skipped or failed: {e}")

    # Read configuration from environment variables with defaults
    pad_sec = float(os.getenv("PODCAST_CREATOR_CLIP_PAD_SEC", "0.2"))
    pad_sec = max(pad_sec, 0.0)
    fade_ms = float(os.getenv("PODCAST_CREATOR_FADE_MS", "10"))
    target_fps = int(os.getenv("PODCAST_CREATOR_AUDIO_FPS", "44100"))
    
    fade_sec = max(fade_ms, 0.0) / 1000.0  # Convert milliseconds to seconds, clamp >= 0

    # Log at INFO so users can see effective settings without enabling debug
    logger.info(
        f"Audio combination settings: pad={pad_sec:.3f}s, fade={fade_sec:.3f}s, fps={target_fps}"
    )
    
    if isinstance(audio_dir, str):
        audio_dir = Path(audio_dir)
    if isinstance(final_output_dir, str):
        final_output_dir = Path(final_output_dir)
    list_of_audio_paths = sorted(audio_dir.glob("*.mp3"))
    output_filename_from_input = final_filename

    logger.debug(list_of_audio_paths)

    if not list_of_audio_paths:
        logger.warning(
            "combine_audio_files: No audio segment data (list of paths) provided."
        )
        return {"combined_audio_path": "ERROR: No audio segment data"}

    if not isinstance(list_of_audio_paths, list):
        logger.error(
            f"combine_audio_files: 'audio_segments_data' is not a list. Received: {type(list_of_audio_paths)}"
        )
        return {
            "combined_audio_path": "ERROR: audio_segments_data must be a list of file paths"
        }

    clips = []
    timeline = []
    t = 0.0
    last_end = 0.0
    total_files = len(list_of_audio_paths)

    # Normalize problematic compressed inputs (e.g., MP3 with bad VBR headers)
    # to WAV via ffmpeg to avoid truncated durations during composition.
    # Prefer the bundled ffmpeg from imageio-ffmpeg (no system install required),
    # falling back to a system ffmpeg if available.
    normalize_inputs = os.getenv("PODCAST_CREATOR_NORMALIZE_INPUT", "1") not in ("0", "false", "False")
    ffmpeg_path = None
    if normalize_inputs:
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = shutil.which("ffmpeg")

    tmpdir_ctx = tempfile.TemporaryDirectory() if normalize_inputs and ffmpeg_path else None
    tmpdir = Path(tmpdir_ctx.name) if tmpdir_ctx else None

    # Log normalization configuration
    logger.info(
        f"Normalization: {'enabled' if normalize_inputs else 'disabled'}"
    )
    if normalize_inputs:
        if ffmpeg_path:
            logger.info(f"Normalization ffmpeg: {ffmpeg_path}")
        else:
            logger.warning(
                "Normalization requested but no ffmpeg executable found. Proceeding without normalization."
            )
    
    for i, file_path in enumerate(list_of_audio_paths):
        if not isinstance(file_path, Path):
            logger.warning(
                f"combine_audio_files: Item {i} in audio_segments_data is not a string path: {file_path}. Skipping."
            )
            continue

        try:
            if file_path.exists() and file_path.is_file():
                # Optional: pre-normalize to WAV for reliable duration
                src_path = file_path
                ext = file_path.suffix.lower()
                if normalize_inputs and ffmpeg_path and ext in {".mp3", ".m4a", ".aac", ".ogg"}:
                    try:
                        norm_path = (tmpdir / f"{file_path.stem}_{i:02d}.wav") if tmpdir else None
                        if norm_path:
                            cmd = [
                                ffmpeg_path,
                                "-y",
                                "-i",
                                str(file_path),
                                "-c:a",
                                "pcm_s16le",
                                "-ar",
                                str(target_fps),
                                str(norm_path),
                            ]
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            src_path = norm_path
                            logger.info(
                                f"Normalized clip {i+1}/{total_files}: {file_path.name} ({ext}) -> {norm_path.name}"
                            )
                    except Exception as norm_exc:
                        logger.warning(f"Normalization to WAV failed for {file_path}: {norm_exc}. Using original file.")
                else:
                    if normalize_inputs and ext not in {".mp3", ".m4a", ".aac", ".ogg"}:
                        logger.debug(f"Skipping normalization for {file_path.name} (ext {ext})")

                # Load clip and set fps
                clip = AudioFileClip(str(src_path))
                clip = clip.with_fps(target_fps)

                # Apply boundary fades conservatively to avoid audible cuts
                # - fade in on non-first clips
                # - fade out on non-last clips
                if fade_sec > 0:
                    effects = []
                    if i > 0:
                        effects.append(AudioFadeIn(fade_sec))
                    if i < total_files - 1:
                        effects.append(AudioFadeOut(fade_sec))
                    if effects:
                        clip = clip.with_effects(effects)

                # Place on timeline at current start time
                clip_start = t
                clip = clip.with_start(clip_start)
                timeline.append(clip)
                clips.append(clip)

                # Compute end based on actual placed duration
                clip_end = clip_start + clip.duration
                last_end = max(last_end, clip_end)

                # Advance timeline position, only add padding if not the last clip
                if i < total_files - 1:
                    t = clip_end + pad_sec
                else:
                    t = clip_end

                logger.info(
                    f"Added clip {i+1}/{total_files}: start={clip_start:.3f}s, "
                    f"dur={clip.duration:.3f}s, end={clip_end:.3f}s"
                )
            else:
                logger.error(
                    f"combine_audio_files: File not found or not a file: {file_path}"
                )
        except Exception as e:
            logger.error(
                f"combine_audio_files: Error loading audio clip {file_path}: {e}"
            )

    if not timeline:
        logger.error("combine_audio_files: No valid audio clips could be loaded.")
        return {"combined_audio_path": "ERROR: No valid clips"}

    try:
        # Use the computed end of the last placed clip for precise duration
        final_duration = max(last_end, 0)

        # Create composite clip with timeline
        final_clip = CompositeAudioClip(timeline).with_duration(final_duration)
        logger.info(
            f"Final composition duration: {final_duration:.3f}s from {len(timeline)} clip(s)"
        )
    except Exception as e:
        logger.error(f"Error during CompositeAudioClip creation: {e}")
        for clip_obj in clips:
            try:
                clip_obj.close()
            except Exception as close_exc:
                logger.debug(f"Error closing clip during error handling: {close_exc}")
        return {"combined_audio_path": f"ERROR: Composition failed - {e}"}

    output_dir = final_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the filename from input if provided, otherwise generate one.
    if output_filename_from_input and isinstance(output_filename_from_input, str):
        # Basic sanitization for filename (optional, depending on how robust it needs to be)
        # For now, assume it's a simple filename like 'episode.mp3'
        output_filename = Path(
            output_filename_from_input
        ).name  # Use only the filename part
        if not output_filename.endswith(".mp3"):
            output_filename += ".mp3"  # Ensure .mp3 extension
    else:
        output_filename = f"combined_{uuid.uuid4().hex}.mp3"
        logger.warning(
            f"'final_filename' not provided or invalid in inputs. Using generated name: {output_filename}"
        )

    output_path = output_dir / output_filename

    try:
        final_clip.write_audiofile(str(output_path), fps=target_fps, codec="libmp3lame")
        logger.info(f"Successfully combined audio to: {output_path.resolve()}")
        return {
            "combined_audio_path": str(output_path.resolve()),
            "original_segments_count": len(clips),
            "total_duration_seconds": final_clip.duration,
        }
    except Exception as e:
        logger.error(f"Error writing final audio file {output_path}: {e}")
        return {"combined_audio_path": f"ERROR: Failed to write output audio - {e}"}
    finally:
        final_clip.close()  # Close the final composite clip
        for clip_obj in clips:  # Ensure all source clips are closed
            try:
                clip_obj.close()
            except Exception as close_exc:
                logger.debug(f"Error closing source clip: {close_exc}")
        if tmpdir_ctx:
            try:
                tmpdir_ctx.cleanup()
            except Exception:
                pass
