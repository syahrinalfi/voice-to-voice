import gradio as gr
import pytest

import app
from v2v import voices


@pytest.fixture
def voice(tmp_path, make_tone, voices_dir, monkeypatch):
    monkeypatch.setattr(app, "OUTPUTS_DIR", tmp_path / "outputs")
    return voices.create("Target", make_tone(tmp_path / "ref.wav"), consent=True)


@pytest.fixture
def engine_calls(monkeypatch):
    calls = []

    def convert(source, reference, out_dir, **kwargs):
        calls.append(kwargs)
        return out_dir / "vc_result.wav"

    monkeypatch.setattr(app.engine, "convert", convert)
    return calls


def test_convert_passes_quality_steps_to_engine(tmp_path, make_tone, voice, engine_calls):
    app.convert(voice.slug, str(make_tone(tmp_path / "src.wav")), 10)

    assert engine_calls == [{"diffusion_steps": 10}]


def test_convert_uses_finetuned_model_when_present(tmp_path, make_tone, voice, engine_calls):
    voice.checkpoint.parent.mkdir()
    voice.checkpoint.write_bytes(b"model")

    app.convert(voice.slug, str(make_tone(tmp_path / "src.wav")), 50)

    assert engine_calls[0]["checkpoint"] == voice.checkpoint
    assert engine_calls[0]["config"] == voice.config


def test_convert_reports_unreadable_source(tmp_path, voice, engine_calls):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not really an mp3")

    with pytest.raises(gr.Error, match="broken.mp3"):
        app.convert(voice.slug, str(bad), 50)

    assert engine_calls == []
