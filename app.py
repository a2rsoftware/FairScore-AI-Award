# Youtube Video summarization

import os, json, subprocess, tempfile, math, time
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from faster_whisper import WhisperModel

# ---------------------------
# Config (tweak as needed)
# ---------------------------
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")  # or "qwen2.5:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ASR_MODEL_SIZE = os.environ.get("ASR_MODEL_SIZE", "base")   # "base" or "small"
ASR_COMPUTE_TYPE = os.environ.get("ASR_COMPUTE", "int8")    # int8 is RAM-friendly

CHUNK_MAX_CHARS = 5000  # ~1.5–2k tokens depending on language
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# Helpers
# ---------------------------
def youtube_id_from_url(url: str) -> str:
    # crude but sufficient for most URLs
    import re
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_\-]{11})', url)
    if not m:
        raise ValueError("Could not parse YouTube video ID from URL")
    return m.group(1)

def _parse_vtt_to_segments(vtt_path: Path) -> List[Dict]:
    """
    Minimal WebVTT parser -> [{text, start, duration}]
    """
    def _ts_to_seconds(ts: str) -> float:
        # 00:01:23.456  or  00:01:23.45
        h, m, s = ts.split(":")
        sec = float(s.replace(",", "."))
        return int(h) * 3600 + int(m) * 60 + sec

    segments = []
    with open(vtt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip("\n") for ln in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "-->" in line:
            # time range line
            try:
                left, right = [x.strip() for x in line.split("-->")]
                start = _ts_to_seconds(left)
                end = _ts_to_seconds(right.split(" ")[0])  # strip any trailing settings
            except Exception:
                i += 1
                continue
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() != "":
                # skip possible cue identifiers like "00:00.000 --> ..." already handled
                if not lines[i].startswith("NOTE") and "-->" not in lines[i]:
                    text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(t for t in text_lines if t)
            if text:
                segments.append({"text": text, "start": start, "duration": max(0.0, end - start)})
        i += 1
    return segments

def _fetch_subs_with_ytdlp(video_id: str) -> List[Dict]:
    """
    Fallback: use yt-dlp to download English subtitles (auto or manual) as VTT and parse them.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        # try manual subs first, then auto
        for flags in (["--write-sub"], ["--write-auto-sub"]):
            cmd = [
                "yt-dlp", "--skip-download",
                *flags,
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "-o", str(tdir / "%(id)s.%(ext)s"),
                url,
            ]
            try:
                subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue

            # find any VTT file produced
            vtts = list(tdir.glob("*.vtt"))
            if vtts:
                return _parse_vtt_to_segments(vtts[0])
    return []

def try_fetch_captions(video_id: str, languages=("en", "en-US", "en-GB")) -> List[Dict]:
    """
    Try multiple strategies so it works regardless of youtube-transcript-api version:
    1) Use get_transcript if available.
    2) Else use list_transcripts if available.
    3) Else fallback to yt-dlp subtitles (VTT) and parse.
    """
    # 1) get_transcript (most versions)
    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            return YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        print(f"⚠️ get_transcript failed: {e}")

    # 2) list_transcripts (newer API)
    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            # Try preferred languages
            for pref in languages:
                try:
                    t = tlist.find_transcript([pref])
                    return t.fetch()
                except Exception:
                    continue
            # Otherwise take the first available
            for t in tlist:
                try:
                    return t.fetch()
                except Exception:
                    continue
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        print(f"⚠️ list_transcripts failed: {e}")

    # 3) yt-dlp fallback
    try:
        subs = _fetch_subs_with_ytdlp(video_id)
        if subs:
            print("✅ Got captions via yt-dlp.")
            return subs
    except Exception as e:
        print(f"⚠️ yt-dlp subtitles fallback failed: {e}")

    return []

def download_audio(url: str, out_path: Path) -> Path:
    # requires ffmpeg in PATH
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", str(out_path / "%(id)s.%(ext)s"),
        url
    ]
    subprocess.check_call(cmd)
    # find resulting file
    files = list(out_path.glob("*.mp3"))
    if not files:
        raise RuntimeError("Audio download failed.")
    return files[0]

def transcribe_local(audio_path: Path) -> List[Dict]:
    model = WhisperModel(ASR_MODEL_SIZE, compute_type=ASR_COMPUTE_TYPE)
    segments, info = model.transcribe(str(audio_path), vad_filter=True)
    out = []
    for seg in segments:
        out.append({"text": seg.text, "start": seg.start, "duration": seg.end - seg.start})
    return out

def flatten_captions(caps: List[Dict]) -> List[Dict]:
    # Ensure {text, start, duration}
    flat = []
    for c in caps:
        flat.append({
            "text": c["text"].strip(),
            "start": float(c["start"]),
            "duration": float(c["duration"])
        })
    return flat

def group_into_chunks(caps: List[Dict], max_chars=CHUNK_MAX_CHARS) -> List[Dict]:
    chunks = []
    buf, cur_chars = [], 0
    for c in caps:
        t = c["text"]
        if not t:
            continue
        if cur_chars + len(t) > max_chars and buf:
            # flush
            start = buf[0]["start"]
            end = buf[-1]["start"] + buf[-1]["duration"]
            chunks.append({
                "text": " ".join(x["text"] for x in buf),
                "start": start,
                "end": end
            })
            buf, cur_chars = [], 0
        buf.append(c)
        cur_chars += len(t) + 1
    if buf:
        start = buf[0]["start"]
        end = buf[-1]["start"] + buf[-1]["duration"]
        chunks.append({"text": " ".join(x["text"] for x in buf), "start": start, "end": end})
    return chunks

import requests, time

def _ping_ollama():
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False

def call_ollama(prompt: str, model=OLLAMA_MODEL, connect_timeout=5, read_timeout=None) -> str:
    """
    Call Ollama /api/generate with better timeouts.
    - connect_timeout: seconds to establish TCP connection (small).
    - read_timeout: seconds to wait for a response body. None = no limit.
    """
    if not _ping_ollama():
        raise RuntimeError(
            "Cannot reach Ollama at "
            f"{OLLAMA_URL}. Start it with 'ollama serve' or ensure the Windows service is running."
        )

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }

    # If you want a finite but generous read timeout, set read_timeout=900 (15 min)
    try:
        r = requests.post(url, json=payload, timeout=(connect_timeout, read_timeout))
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            "Timed out waiting for Ollama to respond. "
            "This often happens on the first run while the model is downloading/loading. "
            "Run 'ollama pull {0}' first and retry, or increase the read timeout."
            .format(model)
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            "Is it running? Try 'ollama serve' in another terminal."
        ) from e


def sec_to_mmss(s: float) -> str:
    m = int(s // 60)
    s = int(s % 60)
    return f"{m:02d}:{s:02d}"

# ---------------------------
# Prompts
# ---------------------------
MAP_PROMPT_TMPL = """You are an expert note-taker.
Summarize the following transcript chunk in 5–8 concise bullets.
Include the chunk time window: {start}-{end}.
Keep only the most important facts.

Transcript:
{body}
"""

REDUCE_PROMPT_TMPL = """Combine these chunk summaries into one clear summary.

Sections:
- TL;DR (3 bullets)
- Key Points (6–10 bullets)
- Notable Quotes (<=3, if any)
- Actionable Takeaways (3–5)

Avoid repetition. Keep it crisp.
Chunk summaries:
{body}
"""

# ---------------------------
# Main
# ---------------------------
def summarize_youtube(url: str) -> Path:
    video_id = youtube_id_from_url(url)
    title = f"youtube_{video_id}"
    workdir = OUT_DIR / video_id
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) captions or ASR
    caps = try_fetch_captions(video_id)
    if caps:
        print("✅ Found captions.")
        caps = flatten_captions(caps)
    else:
        print("ℹ️ No captions available; downloading audio + transcribing locally (this may take a few minutes).")
        audio = download_audio(url, workdir)
        caps = transcribe_local(audio)

    # 2) chunk
    chunks = group_into_chunks(caps, CHUNK_MAX_CHARS)
    print(f"Chunking into {len(chunks)} chunks...")

    # 3) map
    map_summaries = []
    for i, ch in enumerate(tqdm(chunks, desc="Map")):
        start = sec_to_mmss(ch["start"])
        end = sec_to_mmss(ch["end"])
        prompt = MAP_PROMPT_TMPL.format(start=start, end=end, body=ch["text"])
        resp = call_ollama(prompt)
        map_summaries.append(f"[{i+1}] ({start}-{end})\n{resp}")

    # 4) reduce
    reduce_prompt = REDUCE_PROMPT_TMPL.format(body="\n\n".join(map_summaries))
    final = call_ollama(reduce_prompt)

    # 5) write output
    out_md = workdir / f"{title}_summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Summary for {url}\n\n")
        f.write(final)
        f.write("\n")
    return out_md

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python app.py <YouTube URL>")
        sys.exit(1)
    url = sys.argv[1]
    out = summarize_youtube(url)
    print(f"\n✅ Done. Summary saved to: {out}")

