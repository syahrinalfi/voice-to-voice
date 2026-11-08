import subprocess
import sys
from pathlib import Path

import pytest
import soundfile as sf

from v2v import finetune, voices


@pytest.fixture
def voice(tmp_path, make_tone, voices_dir):
    return voices.create("Target", make_tone(tmp_path / "ref.wav", seconds=8.0), consent=True)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(finetune, "RUNS_DIR", root)
    return root


class FakePopen:
    """Stands in for train.py: prints log lines and writes its outputs into the run dir."""
    calls = []
    returncode = 0

    def __init__(self, cmd, **kwargs):
        FakePopen.calls.append((cmd, kwargs))
        self.args = dict(zip(cmd[2::2], cmd[3::2]))
        self.stdout = iter(["loading\n", "epoch 0, step 0, loss: 0.6\r\n", "epoch 9, step 10, loss: 0.5\n"])
        if FakePopen.returncode == 0:
            run_dir = finetune.RUNS_DIR / self.args["--run-name"]
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "ft_model.pth").write_bytes(b"model")
            (run_dir / finetune.PRESET.name).write_text("config")

    def wait(self):
        return FakePopen.returncode


@pytest.fixture
def fake_popen(monkeypatch):
    """Fake only train.py; ffmpeg still runs for real (subprocess.run also goes through Popen)."""
    FakePopen.calls, FakePopen.returncode = [], 0
    real_popen = subprocess.Popen

    def popen(cmd, **kwargs):
        return FakePopen(cmd, **kwargs) if cmd[1:2] == ["train.py"] else real_popen(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    return FakePopen


def test_add_clips_normalizes_into_voice_train_dir(tmp_path, make_tone, voice):
    finetune.add_clips(voice.slug, [make_tone(tmp_path / "extra.mp3", seconds=3.0)])

    saved = sorted((voice.path / "train").glob("*.wav"))
    assert len(saved) == 1
    assert sf.info(saved[0]).channels == 1


def test_train_runs_seed_vc_and_installs_checkpoint(tmp_path, make_tone, voice, runs_dir, fake_popen):
    finetune.add_clips(voice.slug, [make_tone(tmp_path / "extra.wav", seconds=13.0)])
    stale = runs_dir / f"v2v-{voice.slug}" / "DiT_epoch_00001_step_00100.pth"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    lines = list(finetune.train(voice.slug, steps=300))

    cmd, kwargs = fake_popen.calls[0]
    args = dict(zip(cmd[2::2], cmd[3::2]))
    assert cmd[:2] == [sys.executable, "train.py"]
    assert kwargs["cwd"] == finetune.SEED_VC_DIR
    assert args["--max-steps"] == "300"
    assert args["--batch-size"] == "2"
    assert args["--run-name"] == f"v2v-{voice.slug}"
    dataset = Path(args["--dataset-dir"])
    clips = sorted(dataset.glob("*.wav"))
    assert len(clips) >= 4  # 8 s reference + 13 s extra, split into pieces of at most 6 s
    assert all(sf.info(c).duration <= 6.01 for c in clips)
    assert "epoch 9, step 10, loss: 0.5" in lines
    reloaded = voices.load(voice.slug)
    assert reloaded.is_finetuned
    assert reloaded.checkpoint.read_bytes() == b"model"
    assert reloaded.config.read_text() == "config"
    assert not (runs_dir / f"v2v-{voice.slug}").exists()


def test_train_failure_raises_and_keeps_previous_state(voice, runs_dir, fake_popen):
    fake_popen.returncode = 1

    with pytest.raises(finetune.FinetuneError, match="exit 1"):
        list(finetune.train(voice.slug, steps=100))

    assert not voices.load(voice.slug).is_finetuned


def test_train_refuses_while_another_run_is_active(voice, runs_dir, fake_popen):
    with finetune._lock:
        assert finetune.is_training()
        with pytest.raises(finetune.FinetuneError, match="already"):
            list(finetune.train(voice.slug, steps=100))

    assert not finetune.is_training()
