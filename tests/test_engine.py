import subprocess
import sys

import pytest

from v2v import engine


def fake_run(returncode=0, stderr="", write_output=True):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if write_output:
            out_dir = cmd[cmd.index("--output") + 1]
            (engine.Path(out_dir) / "vc_result.wav").write_bytes(b"RIFF")
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=stderr)

    return run, calls


def test_convert_runs_seed_vc_and_returns_output(tmp_path, monkeypatch):
    run, calls = fake_run()
    monkeypatch.setattr(engine.subprocess, "run", run)
    src, ref = tmp_path / "src.wav", tmp_path / "ref.wav"

    out = engine.convert(src, ref, tmp_path / "job")

    assert out.name == "vc_result.wav"
    assert out.parent == tmp_path / "job"
    cmd, kwargs = calls[0]
    assert cmd[:2] == [sys.executable, "inference.py"]
    assert kwargs["cwd"] == engine.SEED_VC_DIR
    args = dict(zip(cmd[2::2], cmd[3::2]))
    assert args["--source"] == str(src.resolve())
    assert args["--target"] == str(ref.resolve())
    assert args["--diffusion-steps"] == "50"
    assert args["--inference-cfg-rate"] == "1.0"
    assert args["--f0-condition"] == "False"


def test_convert_passes_diffusion_steps(tmp_path, monkeypatch):
    run, calls = fake_run()
    monkeypatch.setattr(engine.subprocess, "run", run)

    engine.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "job", diffusion_steps=10)

    cmd = calls[0][0]
    assert cmd[cmd.index("--diffusion-steps") + 1] == "10"


def test_convert_uses_finetuned_checkpoint_when_given(tmp_path, monkeypatch):
    run, calls = fake_run()
    monkeypatch.setattr(engine.subprocess, "run", run)
    ckpt, cfg = tmp_path / "ft_model.pth", tmp_path / "config.yml"

    engine.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "job", checkpoint=ckpt, config=cfg)

    cmd = calls[0][0]
    assert cmd[cmd.index("--checkpoint") + 1] == str(ckpt.resolve())
    assert cmd[cmd.index("--config") + 1] == str(cfg.resolve())


def test_convert_without_checkpoint_uses_default_model(tmp_path, monkeypatch):
    run, calls = fake_run()
    monkeypatch.setattr(engine.subprocess, "run", run)

    engine.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "job")

    assert "--checkpoint" not in calls[0][0]


def test_convert_raises_with_stderr_tail_on_failure(tmp_path, monkeypatch):
    stderr = "\n".join(f"line {i}" for i in range(50)) + "\nCUDA out of memory"
    run, _ = fake_run(returncode=1, stderr=stderr, write_output=False)
    monkeypatch.setattr(engine.subprocess, "run", run)

    with pytest.raises(engine.EngineError) as exc:
        engine.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "job")

    assert "CUDA out of memory" in str(exc.value)
    assert "line 0\n" not in str(exc.value)


def test_convert_raises_when_no_output_written(tmp_path, monkeypatch):
    run, _ = fake_run(write_output=False)
    monkeypatch.setattr(engine.subprocess, "run", run)

    with pytest.raises(engine.EngineError, match="no output"):
        engine.convert(tmp_path / "s.wav", tmp_path / "r.wav", tmp_path / "job")


@pytest.mark.gpu
def test_convert_real_seed_vc(tmp_path):
    examples = engine.SEED_VC_DIR / "examples"

    out = engine.convert(examples / "source/source_s1.wav", examples / "reference/s1p1.wav",
                         tmp_path / "job", diffusion_steps=10)

    assert out.exists() and out.stat().st_size > 0
