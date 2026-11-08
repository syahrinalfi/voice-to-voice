"""Decode uploaded audio into the mono WAV format the engine expects."""
import re
import subprocess
from pathlib import Path

import soundfile as sf

MODEL_SR = 22050  # sample rate of the Seed-VC v1 offline model
MIN_PIECE_SECONDS = 1.0  # Seed-VC's fine-tuning dataset skips shorter clips
MAX_SOURCE_SECONDS = 600  # a 10-min source already takes several minutes to convert on a 6 GB laptop GPU


def normalize(src, dst, max_seconds=None):
    """Decode `src` (any ffmpeg-readable format) into a loudness-normalized mono WAV at MODEL_SR.

    With `max_seconds`, only the first `max_seconds` of audio are kept.
    """
    dst = Path(dst)
    trim = ["-t", str(max_seconds)] if max_seconds else []
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *trim,
         "-ac", "1", "-af", "loudnorm", "-ar", str(MODEL_SR), str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"Could not read {Path(src).name} as audio: {result.stderr.strip()}")
    return dst


def normalize_source(src, dst):
    """normalize() for speech to convert, rejecting recordings longer than MAX_SOURCE_SECONDS."""
    dst = normalize(src, dst)
    seconds = sf.info(dst).duration
    if seconds > MAX_SOURCE_SECONDS:
        raise ValueError(
            f"The speech to convert is {seconds / 60:.1f} min long; the limit is {MAX_SOURCE_SECONDS / 60:g} min."
        )
    return dst


def split_at_pauses(src, out_dir, stem, max_seconds):
    """Cut WAV `src` into `<stem>_NN.wav` pieces of at most `max_seconds`, cutting in pauses where possible.

    Pieces shorter than MIN_PIECE_SECONDS are dropped. Returns the written paths.
    """
    total = sf.info(src).duration
    pauses = _pause_midpoints(src)
    cuts = [0.0]
    while total - cuts[-1] > max_seconds:
        start = cuts[-1]
        inside = [m for m in pauses if start + MIN_PIECE_SECONDS <= m <= start + max_seconds]
        cuts.append(inside[-1] if inside else start + max_seconds)
    cuts.append(total)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    for i, (start, end) in enumerate(zip(cuts, cuts[1:])):
        if end - start < MIN_PIECE_SECONDS:
            continue
        dst = out_dir / f"{stem}_{i:02d}.wav"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-ss", f"{start:.3f}", "-to", f"{end:.3f}", str(dst)],
            check=True,
        )
        pieces.append(dst)
    return pieces


def _pause_midpoints(src):
    log = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(src),
         "-af", "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return [(s + e) / 2 for s, e in zip(starts, ends)]
