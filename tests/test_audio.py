import subprocess

import pytest
import soundfile as sf

from v2v import audio
from v2v.audio import MODEL_SR, normalize, normalize_source, split_at_pauses


@pytest.mark.parametrize("ext", ["wav", "mp3", "ogg"])
def test_normalize_outputs_mono_wav_at_model_rate(tmp_path, make_tone, ext):
    src = make_tone(tmp_path / f"in.{ext}")

    out = normalize(src, tmp_path / "out.wav")

    info = sf.info(out)
    assert info.channels == 1
    assert info.samplerate == MODEL_SR
    assert info.duration == pytest.approx(2.0, abs=0.1)


def test_normalize_trims_to_max_seconds(tmp_path, make_tone):
    src = make_tone(tmp_path / "long.wav", seconds=5.0)

    out = normalize(src, tmp_path / "out.wav", max_seconds=3.0)

    assert sf.info(out).duration == pytest.approx(3.0, abs=0.05)


def test_normalize_source_accepts_audio_within_limit(tmp_path, make_tone, monkeypatch):
    monkeypatch.setattr(audio, "MAX_SOURCE_SECONDS", 3)

    out = normalize_source(make_tone(tmp_path / "ok.wav", seconds=2.0), tmp_path / "out.wav")

    assert sf.info(out).duration == pytest.approx(2.0, abs=0.1)


def test_normalize_source_rejects_audio_over_limit(tmp_path, make_tone, monkeypatch):
    monkeypatch.setattr(audio, "MAX_SOURCE_SECONDS", 3)

    with pytest.raises(ValueError, match="limit"):
        normalize_source(make_tone(tmp_path / "long.wav", seconds=5.0), tmp_path / "out.wav")


def durations(paths):
    return [sf.info(p).duration for p in paths]


def test_split_cuts_long_audio_without_pauses_into_max_length_pieces(tmp_path, make_tone):
    src = make_tone(tmp_path / "long.wav", seconds=20.0, channels=1)

    pieces = split_at_pauses(src, tmp_path / "out", "long", max_seconds=6.0)

    assert all(1.0 <= d <= 6.0 + 0.01 for d in durations(pieces))
    assert sum(durations(pieces)) == pytest.approx(20.0, abs=0.1)
    assert [p.name for p in pieces][:2] == ["long_00.wav", "long_01.wav"]


def test_split_prefers_cutting_at_pauses(tmp_path):
    src = tmp_path / "speechy.wav"
    tone = "sine=frequency=220:sample_rate=22050:duration=4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", tone,
         "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono:d=1", "-f", "lavfi", "-i", tone,
         "-filter_complex", "[0][1][2]concat=n=3:v=0:a=1", str(src)],
        check=True,
    )

    pieces = split_at_pauses(src, tmp_path / "out", "speechy", max_seconds=6.0)

    assert durations(pieces) == [pytest.approx(4.5, abs=0.1), pytest.approx(4.5, abs=0.1)]


def test_split_drops_trailing_piece_under_one_second(tmp_path, make_tone):
    src = make_tone(tmp_path / "a.wav", seconds=6.5, channels=1)

    pieces = split_at_pauses(src, tmp_path / "out", "a", max_seconds=6.0)

    assert durations(pieces) == [pytest.approx(6.0, abs=0.01)]


def test_normalize_raises_on_non_audio(tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("not audio")

    with pytest.raises(ValueError, match="notes.txt"):
        normalize(src, tmp_path / "out.wav")
