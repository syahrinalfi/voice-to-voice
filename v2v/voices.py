"""Saved voices: one folder per voice under data/voices/<slug>/.

Each folder holds reference.wav and meta.json, plus train/ (extra recordings) and
finetune/ (ft_model.pth + config.yml) once the voice has been fine-tuned.
"""
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf

from v2v import audio

VOICES_DIR = Path(__file__).resolve().parent.parent / "data" / "voices"
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 30.0  # longer references are trimmed; Seed-VC uses only the first 25 s


@dataclass
class Voice:
    slug: str
    name: str
    created: str
    path: Path

    @property
    def reference(self):
        return self.path / "reference.wav"

    @property
    def train_dir(self):
        """Extra recordings added for fine-tuning."""
        return self.path / "train"

    @property
    def checkpoint(self):
        return self.path / "finetune" / "ft_model.pth"

    @property
    def config(self):
        return self.path / "finetune" / "config.yml"

    @property
    def is_finetuned(self):
        return self.checkpoint.exists()


def create(name, reference_src, consent):
    """Save a new voice from a reference recording and return it."""
    name = name.strip()
    if not name:
        raise ValueError("Give the voice a name.")
    if not consent:
        raise ValueError("Confirm you have the speaker's consent to clone this voice.")
    path = VOICES_DIR / _unique_slug(name)
    path.mkdir(parents=True)
    try:
        reference = audio.normalize(reference_src, path / "reference.wav", max_seconds=MAX_REFERENCE_SECONDS)
        seconds = sf.info(reference).duration
        if seconds < MIN_REFERENCE_SECONDS:
            raise ValueError(f"The reference must be at least 1 s of speech; this one is {seconds:.1f} s.")
    except ValueError:
        shutil.rmtree(path)
        raise
    meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), "consent": True}
    (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return load(path.name)


def load(slug):
    path = VOICES_DIR / slug
    meta = json.loads((path / "meta.json").read_text())
    return Voice(slug=slug, name=meta["name"], created=meta["created"], path=path)


def list_voices():
    if not VOICES_DIR.exists():
        return []
    found = [load(p.name) for p in VOICES_DIR.iterdir() if (p / "meta.json").exists()]
    return sorted(found, key=lambda v: v.name.lower())


def delete(slug):
    shutil.rmtree(VOICES_DIR / slug)


def _unique_slug(name):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
    slug, n = base, 2
    while (VOICES_DIR / slug).exists():
        slug, n = f"{base}-{n}", n + 1
    return slug
