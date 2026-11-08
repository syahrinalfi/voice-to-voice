import json

import pytest
import soundfile as sf

from v2v import voices


pytestmark = pytest.mark.usefixtures("voices_dir")


def test_create_saves_reference_and_meta(tmp_path, make_tone, voices_dir):
    voice = voices.create("My Voice", make_tone(tmp_path / "ref.mp3"), consent=True)

    assert voice.slug == "my-voice"
    assert voice.name == "My Voice"
    assert voice.reference == voices_dir / "my-voice" / "reference.wav"
    assert sf.info(voice.reference).channels == 1
    meta = json.loads((voices_dir / "my-voice" / "meta.json").read_text())
    assert meta["name"] == "My Voice"
    assert meta["consent"] is True
    assert meta["created"]
    assert not voice.is_finetuned


def test_list_and_load_read_back_from_disk(tmp_path, make_tone):
    voices.create("Bravo", make_tone(tmp_path / "b.wav"), consent=True)
    voices.create("Alpha", make_tone(tmp_path / "a.wav"), consent=True)

    assert [v.name for v in voices.list_voices()] == ["Alpha", "Bravo"]
    assert voices.load("bravo").name == "Bravo"


def test_list_is_empty_before_any_voice():
    assert voices.list_voices() == []


def test_same_name_gets_unique_slug(tmp_path, make_tone):
    first = voices.create("My Voice", make_tone(tmp_path / "1.wav"), consent=True)
    second = voices.create("My Voice", make_tone(tmp_path / "2.wav"), consent=True)

    assert (first.slug, second.slug) == ("my-voice", "my-voice-2")


def test_non_ascii_name_keeps_name_with_fallback_slug(tmp_path, make_tone):
    voice = voices.create("さくら", make_tone(tmp_path / "ref.wav"), consent=True)

    assert voice.slug == "voice"
    assert voice.name == "さくら"


def test_create_requires_consent(tmp_path, make_tone, voices_dir):
    with pytest.raises(ValueError, match="consent"):
        voices.create("Someone", make_tone(tmp_path / "ref.wav"), consent=False)

    assert not (voices_dir / "someone").exists()


def test_create_requires_name(tmp_path, make_tone):
    with pytest.raises(ValueError, match="name"):
        voices.create("   ", make_tone(tmp_path / "ref.wav"), consent=True)


def test_create_rejects_too_short_reference(tmp_path, make_tone, voices_dir):
    with pytest.raises(ValueError, match="at least 1 s"):
        voices.create("Short", make_tone(tmp_path / "ref.wav", seconds=0.5), consent=True)

    assert not (voices_dir / "short").exists()


def test_create_trims_long_reference_to_30s(tmp_path, make_tone):
    voice = voices.create("Long", make_tone(tmp_path / "ref.wav", seconds=35), consent=True)

    assert sf.info(voice.reference).duration == pytest.approx(30.0, abs=0.05)


def test_create_rejects_non_audio(tmp_path, voices_dir):
    src = tmp_path / "notes.txt"
    src.write_text("not audio")

    with pytest.raises(ValueError):
        voices.create("Text", src, consent=True)

    assert not (voices_dir / "text").exists()


def test_delete_removes_voice(tmp_path, make_tone, voices_dir):
    voices.create("Gone", make_tone(tmp_path / "ref.wav"), consent=True)

    voices.delete("gone")

    assert not (voices_dir / "gone").exists()
    assert voices.list_voices() == []
