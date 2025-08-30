import os, math, textwrap, subprocess, tempfile, uuid
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont
import pyttsx3
from rich import print

# -----------------------------
# Config (edit to taste)
# -----------------------------
W, H = 1920, 1080          # video size (YouTube/LinkedIn 16:9)
FPS = 30
BG_COLOR = (18, 24, 38)    # deep blue/gray
FG_COLOR = (240, 244, 255) # off-white
TITLE_COLOR = (133, 180, 255)
MARGIN = 120               # px around edges
MAX_LINES = 14             # safety for text overflow
VOICE_RATE_WPM = 175       # TTS speaking rate
VOICE_NAME_PREF = ""       # e.g., "Zira", "David" (leave "" to auto-pick)
FONT_PATHS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
]
TITLE_FONT_SIZE = 70
BODY_FONT_SIZE = 54

# NEW: output sizing / grouping
MAX_OUTPUT_VIDEOS = 5            # set 1 for one big video, or 5 for five big videos
MIN_SCENE_SEC = 4.0              # pad each scene to at least this many seconds
X264_CRF = 20                    # lower = bigger/better (18–23 typical)
X264_PRESET = "medium"           # slower = better compression; "fast" on low-end PCs
AUDIO_BITRATE = "160k"           # AAC audio bitrate

# -----------------------------
# Utilities
# -----------------------------
def find_font(fallback=True) -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    for p in FONT_PATHS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, TITLE_FONT_SIZE), ImageFont.truetype(p, BODY_FONT_SIZE)
            except Exception:
                pass
    if fallback:
        return ImageFont.load_default(), ImageFont.load_default()
    raise RuntimeError("No usable font found.")

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    words = text.strip().split()
    lines, line = [], []
    draw = ImageDraw.Draw(Image.new("RGB", (W, H)))
    for w in words:
        test = " ".join(line + [w])
        if draw.textlength(test, font=font) <= max_width:
            line.append(w)
        else:
            if line:
                lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return lines

def pick_voice(engine: pyttsx3.Engine, pref: str="") -> str:
    voices = engine.getProperty("voices")
    if pref:
        for v in voices:
            if pref.lower() in (v.name or "").lower():
                return v.id
    for v in voices:
        name = (v.name or "").lower()
        if any(k in name for k in ["zira","david","hedi","mark","english","en-us","en_gb"]):
            return v.id
    return voices[0].id if voices else ""

def synth_tts(text: str, wav_path: Path, voice_name_pref: str=""):
    engine = pyttsx3.init()
    voice_id = pick_voice(engine, voice_name_pref)
    if voice_id:
        engine.setProperty("voice", voice_id)
    engine.setProperty("rate", VOICE_RATE_WPM)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()

def duration_s(wav_path: Path) -> float:
    """
    Get WAV duration using the Python standard library.
    Works on Windows, no pydub/audioop needed.
    """
    import wave, contextlib
    with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0

def make_slide_image(title: str, body: str, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)

    title_font, body_font = find_font()
    # Title at top
    title_lines = wrap_text(title, title_font, W - 2*MARGIN)
    y = MARGIN
    for line in title_lines:
        w_line = draw.textlength(line, font=title_font)
        draw.text(((W - w_line)//2, y), line, fill=TITLE_COLOR, font=title_font)
        y += TITLE_FONT_SIZE + 10

    # Body, centered block
    body_lines = wrap_text(body, body_font, W - 2*MARGIN)
    if len(body_lines) > MAX_LINES:
        body_lines = body_lines[:MAX_LINES-1] + ["[...]"]

    # estimate line height
    def line_h(s: str) -> int:
        # PIL getbbox returns (l,t,r,b); height = b
        return body_font.getbbox(s or "Ag")[3]

    total_h = sum(line_h(l) for l in body_lines) + (len(body_lines)-1)*10
    start_y = max(y + 40, (H - total_h)//2)

    for line in body_lines:
        w_line = draw.textlength(line, font=body_font)
        draw.text(((W - w_line)//2, start_y), line, fill=FG_COLOR, font=body_font)
        start_y += line_h(line) + 10

    img.save(out_png)

def run_ffmpeg(cmd: list):
    print("[dim]FFMPEG>[/] " + " ".join(cmd))
    subprocess.check_call(cmd)

# -----------------------------
# Core pipeline
# -----------------------------
@dataclass
class Scene:
    title: str
    text: str
    wav: Path
    png: Path
    dur: float
    seg: Path

def split_into_scenes(script_text: str) -> List[Tuple[str, str]]:
    # Split by blank lines into scenes; first sentence = title, rest = body
    blocks = [b.strip() for b in script_text.split("\n\n") if b.strip()]
    scenes = []
    for b in blocks:
        parts = b.split(". ")
        if len(parts) > 1:
            title = parts[0].strip()
            body = ". ".join(parts[1:]).strip()
        else:
            title, body = "Scene", b
        scenes.append((title, body))
    return scenes

def chunk(lst, n):
    """Split list into n nearly equal chunks."""
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        if size == 0:
            break
        chunks.append(lst[start:start+size])
        start += size
    return chunks

def build_video(script_path: Path, out_mp4: Path, bgm_path: Path|None=None):
    work = Path("build"); work.mkdir(exist_ok=True)
    audio_dir = work/"audio"; audio_dir.mkdir(exist_ok=True, parents=True)
    img_dir = work/"img"; img_dir.mkdir(exist_ok=True, parents=True)
    seg_dir = work/"seg"; seg_dir.mkdir(exist_ok=True, parents=True)

    script = script_path.read_text(encoding="utf-8")
    scene_defs = split_into_scenes(script)

    scenes: List[Scene] = []
    t_cursor = 0.0
    srt_lines = []

    # 1) TTS + slides + per-segment render
    for idx, (title, body) in enumerate(scene_defs, start=1):
        base = f"s{idx:03d}"
        wav = audio_dir/f"{base}.wav"
        png = img_dir/f"{base}.png"
        seg = seg_dir/f"{base}.mp4"

        print(f"[bold cyan]Scene {idx}[/] — TTS")
        synth_tts(f"{title}. {body}", wav, VOICE_NAME_PREF)
        dur = max(0.1, duration_s(wav))  # safety
        seg_dur = max(dur, MIN_SCENE_SEC)

        print(f"[bold cyan]Scene {idx}[/] — slide")
        make_slide_image(title, body, png)

        print(f"[bold cyan]Scene {idx}[/] — video segment ({seg_dur:.2f}s)")
        # - apad pads audio with silence; -t sets output length; no -shortest
        run_ffmpeg([
            "ffmpeg", "-y",
            "-loop", "1", "-framerate", str(FPS), "-i", str(png),
            "-i", str(wav),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", f"scale={W}:{H}",
            "-r", str(FPS),
            "-c:a", "aac", "-b:a", AUDIO_BITRATE,
            "-af", "apad",
            "-t", f"{seg_dur:.3f}",
            "-preset", X264_PRESET,
            "-crf", str(X264_CRF),
            str(seg)
        ])

        # SRT entry per scene (one block = full text)
        start = t_cursor
        end = t_cursor + seg_dur
        srt_lines.append(format_srt(len(srt_lines)+1, start, end, f"{title}\n{body}"))
        t_cursor = end

        scenes.append(Scene(title, body, wav, png, seg_dur, seg))

    # 2) Group scenes into MAX_OUTPUT_VIDEOS parts and concat
    groups = chunk(scenes, MAX_OUTPUT_VIDEOS) if MAX_OUTPUT_VIDEOS > 1 else [scenes]
    outputs = []

    for part_idx, group in enumerate(groups, start=1):
        concat_list = work / f"segments_part{part_idx:02d}.txt"
        lines = []
        for s in group:
            p = s.seg.resolve()
            lines.append(f"file '{p.as_posix()}'")
        concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out_part = out_mp4 if len(groups) == 1 else out_mp4.with_stem(f"{out_mp4.stem}_part{part_idx:02d}")
        print(f"[bold green]Concat → {out_part.name}[/]")
        run_ffmpeg([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy", str(out_part)
        ])
        outputs.append(out_part)

    # 3) Single SRT covering the whole script (only if one output)
    if len(outputs) == 1:
        srt_path = outputs[0].with_suffix(".srt")
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        print(f"[bold green]✅ Done[/] Video: {outputs[0]}   Subtitles: {srt_path}")
    else:
        print(f"[bold green]✅ Done[/] Created {len(outputs)} videos:")
        for p in outputs:
            print(" -", p)

def format_srt(idx: int, start_s: float, end_s: float, text: str) -> str:
    return f"{idx}\n{to_ts(start_s)} --> {to_ts(end_s)}\n{text}\n"

def to_ts(s: float) -> str:
    h = int(s//3600); m = int((s%3600)//60); sec = int(s%60); ms = int((s - int(s))*1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Text to Video (slides + TTS) for Windows")
    p.add_argument("--script", type=str, default="script.txt")
    p.add_argument("--out", type=str, default="output.mp4")
    args = p.parse_args()

    build_video(Path(args.script), Path(args.out))