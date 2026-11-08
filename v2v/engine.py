"""Run Seed-VC voice conversion through its own CLI (never its internals)."""
import subprocess
import sys
from pathlib import Path

SEED_VC_DIR = Path(__file__).resolve().parent.parent / "third_party" / "seed-vc"
STDERR_TAIL_LINES = 20


class EngineError(RuntimeError):
    pass


def convert(source, reference, out_dir, diffusion_steps=50, cfg_rate=1.0, checkpoint=None, config=None):
    """Speak `source`'s words in `reference`'s voice and return the output WAV path.

    Defaults are the settings that sounded best in the Task 1 listening tests (variant A).
    Pass a fine-tuned `checkpoint` with its `config` to use it instead of the pretrained model.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "inference.py",
        "--source", str(Path(source).resolve()),
        "--target", str(Path(reference).resolve()),
        "--output", str(out_dir),
        "--diffusion-steps", str(diffusion_steps),
        "--length-adjust", "1.0",
        "--inference-cfg-rate", str(cfg_rate),
        "--f0-condition", "False",
        "--auto-f0-adjust", "False",
        "--semi-tone-shift", "0",
        "--fp16", "True",
    ]
    if checkpoint:
        cmd += ["--checkpoint", str(Path(checkpoint).resolve()), "--config", str(Path(config).resolve())]
    result = subprocess.run(cmd, cwd=SEED_VC_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-STDERR_TAIL_LINES:])
        raise EngineError(f"Seed-VC failed (exit {result.returncode}):\n{tail}")
    outputs = sorted(out_dir.glob("vc_*.wav"))
    if not outputs:
        raise EngineError("Seed-VC finished but wrote no output file")
    return outputs[-1]
