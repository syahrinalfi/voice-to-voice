"""Fine-tune Seed-VC on a saved voice through its own train.py (never its internals)."""
import shutil
import subprocess
import sys
import threading
import uuid

from v2v import audio, voices
from v2v.engine import SEED_VC_DIR

RUNS_DIR = SEED_VC_DIR / "runs"  # train.py writes to ./runs/<run-name> (log_dir in the preset)
PRESET = SEED_VC_DIR / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"
DEFAULT_STEPS = 400  # best-sounding setting in Task 2 (2 min of audio)
BATCH_SIZE = 2
# Batches are padded to their longest clip; clips up to 11 s peaked at 5.9 of 6 GB in Task 2.
MAX_CLIP_SECONDS = 6.0
LOG_TAIL_LINES = 20

_lock = threading.Lock()


class FinetuneError(RuntimeError):
    pass


def is_training():
    return _lock.locked()


def add_clips(slug, paths):
    """Store extra recordings of a voice, normalized, for this and later training runs."""
    train_dir = voices.load(slug).train_dir
    train_dir.mkdir(exist_ok=True)
    for path in paths:
        audio.normalize(path, train_dir / f"{uuid.uuid4().hex[:8]}.wav")


def train(slug, steps=DEFAULT_STEPS):
    """Fine-tune on the voice's reference plus its extra recordings, yielding log lines.

    On success the voice's fine-tuned model is replaced; on failure it is left untouched.
    """
    if not _lock.acquire(blocking=False):
        raise FinetuneError("A training run is already in progress. Wait for it to finish.")
    try:
        voice = voices.load(slug)
        dataset = voice.path / "dataset"
        shutil.rmtree(dataset, ignore_errors=True)
        for i, src in enumerate([voice.reference, *sorted(voice.train_dir.glob("*.wav"))]):
            audio.split_at_pauses(src, dataset, f"{i:03d}", MAX_CLIP_SECONDS)
        yield f"Prepared {len(list(dataset.glob('*.wav')))} training clips; starting {steps} steps."

        run_dir = RUNS_DIR / f"v2v-{slug}"
        shutil.rmtree(run_dir, ignore_errors=True)  # train.py would otherwise resume from old checkpoints
        cmd = [
            sys.executable, "train.py",
            "--config", str(PRESET),
            "--dataset-dir", str(dataset),
            "--run-name", run_dir.name,
            "--batch-size", str(BATCH_SIZE),
            "--max-steps", str(steps),
            "--max-epochs", "1000",
            "--save-every", str(steps + 1),  # skip 1.2 GB intermediate checkpoints
            "--num-workers", "0",
        ]
        proc = subprocess.Popen(cmd, cwd=SEED_VC_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for raw in proc.stdout:
            line = raw.rstrip().split("\r")[-1].strip()
            tail = (tail + [line])[-LOG_TAIL_LINES:]
            if line and "%|" not in line:  # skip tqdm progress bars
                yield line
        code = proc.wait()
        if code != 0:
            raise FinetuneError(f"Seed-VC training failed (exit {code}):\n" + "\n".join(tail))

        model = run_dir / "ft_model.pth"
        if not model.exists():
            raise FinetuneError("Training finished but wrote no model file.")
        voice.checkpoint.parent.mkdir(exist_ok=True)
        shutil.move(model, voice.checkpoint)
        shutil.move(run_dir / PRESET.name, voice.config)
        shutil.rmtree(run_dir)
        yield "Done. This voice now converts with its fine-tuned model."
    finally:
        _lock.release()
