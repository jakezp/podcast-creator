import os
import re
import uuid
from pathlib import Path
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
    
    for i, file_path in enumerate(list_of_audio_paths):
        if not isinstance(file_path, Path):
            logger.warning(
                f"combine_audio_files: Item {i} in audio_segments_data is not a string path: {file_path}. Skipping."
            )
            continue

        try:
            if file_path.exists() and file_path.is_file():
                # Load clip and set fps
                clip = AudioFileClip(str(file_path))
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
