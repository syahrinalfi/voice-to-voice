import subprocess

import pytest

from v2v import voices


@pytest.fixture
def make_tone():
    """Write a sine-tone audio file; the format follows the path's extension."""
    def make(path, seconds=2.0, sr=44100, channels=2):
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"sine=frequency=220:sample_rate={sr}:duration={seconds}",
             "-ac", str(channels), str(path)],
            check=True,
        )
        return path

    return make


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    root = tmp_path / "voices"
    monkeypatch.setattr(voices, "VOICES_DIR", root)
    return root
