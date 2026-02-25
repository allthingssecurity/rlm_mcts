"""
YouTube video transcription: yt-dlp captions → Whisper API fallback.
"""

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TranscriptSegment:
    start: float  # seconds
    end: float    # seconds
    text: str


async def transcribe(url: str) -> list[TranscriptSegment]:
    """Transcribe a YouTube video: try captions first, then Whisper API."""
    segments = _try_captions(url)
    if segments:
        return segments
    return await _whisper_fallback(url)


def _try_captions(url: str) -> list[TranscriptSegment] | None:
    """Try extracting captions via yt-dlp (manual first, then auto-generated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "subs")

        # Try manual captions first
        for auto in (False, True):
            opts = {
                "skip_download": True,
                "outtmpl": out_path,
                "quiet": True,
                "no_warnings": True,
            }
            if auto:
                opts["writeautomaticsub"] = True
            else:
                opts["writesubtitles"] = True
            opts["subtitleslangs"] = ["en"]
            opts["subtitlesformat"] = "vtt"

            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    ydl.download([url])
            except Exception:
                continue

            # Find the subtitle file
            for f in Path(tmpdir).glob("*.vtt"):
                text = f.read_text(encoding="utf-8", errors="replace")
                segments = _parse_vtt(text)
                if segments:
                    return _deduplicate(segments)

            for f in Path(tmpdir).glob("*.srt"):
                text = f.read_text(encoding="utf-8", errors="replace")
                segments = _parse_srt(text)
                if segments:
                    return _deduplicate(segments)

    return None


async def _whisper_fallback(url: str) -> list[TranscriptSegment]:
    """Download audio and transcribe with OpenAI Whisper API."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        opts = {
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": audio_path,
            "quiet": True,
            "no_warnings": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }

        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

        # Find the actual audio file (extension may vary)
        audio_file = None
        for ext in (".mp3", ".m4a", ".webm", ".ogg"):
            candidate = audio_path.replace(".mp3", ext)
            if os.path.exists(candidate):
                audio_file = candidate
                break
        if not audio_file:
            # Try the exact path
            audio_file = audio_path

        with open(audio_file, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments = []
        for seg in response.segments:
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))
        return segments


def _parse_vtt(text: str) -> list[TranscriptSegment]:
    """Parse WebVTT subtitle content."""
    segments = []
    # Remove VTT header and style blocks
    text = re.sub(r"WEBVTT.*?\n\n", "", text, flags=re.DOTALL)
    text = re.sub(r"STYLE.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"NOTE.*?(?=\n\n|\Z)", "", text, flags=re.DOTALL)

    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find the timestamp line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                # Remove VTT tags like <c>, </c>, etc.
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if clean:
                    text_lines.append(clean)

        if ts_line and text_lines:
            match = re.search(
                r"(\d+:)?(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+:)?(\d+):(\d+)[.,](\d+)",
                ts_line,
            )
            if match:
                g = match.groups()
                start = _ts_to_seconds(g[0], g[1], g[2], g[3])
                end = _ts_to_seconds(g[4], g[5], g[6], g[7])
                segments.append(TranscriptSegment(
                    start=start, end=end, text=" ".join(text_lines),
                ))

    return segments


def _parse_srt(text: str) -> list[TranscriptSegment]:
    """Parse SRT subtitle content."""
    segments = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue
        # Skip the index line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif ts_line is not None:
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if clean:
                    text_lines.append(clean)

        if ts_line and text_lines:
            match = re.search(
                r"(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)",
                ts_line,
            )
            if match:
                g = match.groups()
                start = _ts_to_seconds(g[0], g[1], g[2], g[3])
                end = _ts_to_seconds(g[4], g[5], g[6], g[7])
                segments.append(TranscriptSegment(
                    start=start, end=end, text=" ".join(text_lines),
                ))

    return segments


def _ts_to_seconds(h: str | None, m: str, s: str, ms: str) -> float:
    """Convert timestamp parts to seconds."""
    hours = int(h.rstrip(":")) if h else 0
    return hours * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def _deduplicate(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """Deduplicate overlapping auto-caption lines."""
    if not segments:
        return segments

    deduped = [segments[0]]
    for seg in segments[1:]:
        prev = deduped[-1]
        # Skip if the text is identical to the previous segment
        if seg.text == prev.text:
            # Extend the end time
            prev.end = max(prev.end, seg.end)
            continue
        # Skip if the current text is a substring of the previous
        if seg.text in prev.text:
            continue
        # If previous is a substring of current (progressive reveal), replace
        if prev.text in seg.text:
            prev.text = seg.text
            prev.end = seg.end
            continue
        deduped.append(seg)

    return deduped


def get_video_info(url: str) -> dict:
    """Get basic video metadata."""
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return {
            "id": info.get("id", ""),
            "title": info.get("title", ""),
            "duration": info.get("duration", 0),
            "channel": info.get("channel", ""),
            "url": url,
        }
