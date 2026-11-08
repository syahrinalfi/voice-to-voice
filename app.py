"""Local web UI: re-speak an uploaded recording in a saved voice.

Run with `uv run python app.py`, then open http://127.0.0.1:7860.
"""
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr
import soundfile as sf

from v2v import audio, engine, finetune, voices

OUTPUTS_DIR = Path(__file__).resolve().parent / "data" / "outputs"
LOG_LINES_SHOWN = 15


def voice_choices():
    return [(f"{v.name} · fine-tuned" if v.is_finetuned else v.name, v.slug) for v in voices.list_voices()]


def pickers(**kwargs):
    """Updated choices for the three voice dropdowns (convert, delete, train)."""
    choices = voice_choices()
    return tuple(gr.Dropdown(choices=choices, **kwargs) for _ in range(3))


def convert(slug, source_path, steps):
    if not slug:
        raise gr.Error("Pick a voice. Create one in the Voices tab first.")
    if not source_path:
        raise gr.Error("Upload or record the speech to convert.")
    if finetune.is_training():
        raise gr.Error("A voice is training and using the GPU. Convert after it finishes.")
    voice = voices.load(slug)
    job = OUTPUTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    job.mkdir(parents=True)
    try:
        source = audio.normalize_source(source_path, job / "source.wav")
        model = {"checkpoint": voice.checkpoint, "config": voice.config} if voice.is_finetuned else {}
        return str(engine.convert(source, voice.reference, job, diffusion_steps=int(steps), **model))
    except (ValueError, engine.EngineError) as e:
        raise gr.Error(str(e)) from e


def save_voice(name, reference_path, consent):
    if not reference_path:
        raise gr.Error("Upload a reference recording of the voice.")
    try:
        voice = voices.create(name, reference_path, consent)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    seconds = sf.info(voice.reference).duration
    choices = voice_choices()
    return (
        f"Saved **{voice.name}** ({seconds:.1f} s of reference audio).",
        gr.Dropdown(choices=choices, value=voice.slug), gr.Dropdown(choices=choices), gr.Dropdown(choices=choices),
    )


def delete_voice(slug):
    if not slug:
        raise gr.Error("Pick a voice to delete.")
    if finetune.is_training():
        raise gr.Error("Wait for the training run to finish before deleting voices.")
    name = voices.load(slug).name
    voices.delete(slug)
    return (f"Deleted **{name}**.", *pickers(value=None))


def train_voice(slug, files, steps):
    if not slug:
        raise gr.Error("Pick a voice to train.")
    if finetune.is_training():  # checked before add_clips so a refused run stores nothing
        raise gr.Error("A training run is already in progress. Wait for it to finish.")
    log = []
    try:
        if files:
            finetune.add_clips(slug, files)
        for line in finetune.train(slug, int(steps)):
            log.append(line)
            yield "\n".join(log[-LOG_LINES_SHOWN:]), gr.skip(), gr.skip(), gr.skip()
    except (ValueError, finetune.FinetuneError) as e:
        raise gr.Error(str(e)) from e
    yield ("\n".join(log[-LOG_LINES_SHOWN:]), *pickers())


with gr.Blocks(title="Voice-to-Voice") as demo:
    gr.Markdown("# Voice-to-Voice")
    with gr.Tab("Convert"):
        with gr.Row():
            voice_pick = gr.Dropdown(label="Voice", choices=voice_choices())
            source_in = gr.Audio(label="Speech to convert (up to 10 min)", sources=["upload", "microphone"],
                                 type="filepath")
        steps_pick = gr.Slider(label="Quality steps (10 = fastest, 50 = best)", minimum=10, maximum=50, step=5,
                               value=50)
        convert_btn = gr.Button("Convert", variant="primary")
        result_out = gr.Audio(label="Converted", type="filepath", interactive=False)
    with gr.Tab("Voices"):
        name_in = gr.Textbox(label="Voice name")
        reference_in = gr.Audio(
            label="Reference recording (clean speech, 1–30 s; longer clips are trimmed to 30 s)",
            sources=["upload"], type="filepath",
        )
        consent_in = gr.Checkbox(label="I have the speaker's consent to clone this voice")
        save_btn = gr.Button("Save voice", variant="primary")
        voice_status = gr.Markdown()
        with gr.Row():
            delete_pick = gr.Dropdown(label="Delete a voice", choices=voice_choices())
            delete_btn = gr.Button("Delete", variant="stop")
    with gr.Tab("Train"):
        gr.Markdown(
            "Fine-tune a saved voice so conversions sound closer to it. Training uses the voice's "
            "reference plus any extra recordings you add here (kept for later runs). More clean audio "
            "of the same person helps most. 400 steps take about 6 minutes."
        )
        train_pick = gr.Dropdown(label="Voice", choices=voice_choices())
        clips_in = gr.File(label="Extra recordings of this voice (optional)", file_count="multiple",
                           file_types=["audio"], type="filepath")
        steps_in = gr.Slider(label="Training steps", minimum=100, maximum=1000, step=50,
                             value=finetune.DEFAULT_STEPS)
        train_btn = gr.Button("Start training", variant="primary")
        train_log = gr.Textbox(label="Training log", lines=LOG_LINES_SHOWN, interactive=False)

    convert_btn.click(convert, inputs=[voice_pick, source_in, steps_pick], outputs=result_out)
    save_btn.click(save_voice, inputs=[name_in, reference_in, consent_in],
                   outputs=[voice_status, voice_pick, delete_pick, train_pick])
    delete_btn.click(delete_voice, inputs=delete_pick, outputs=[voice_status, voice_pick, delete_pick, train_pick])
    # No queue limit: a second click must reach finetune's lock and be refused, not wait in line.
    train_btn.click(train_voice, inputs=[train_pick, clips_in, steps_in],
                    outputs=[train_log, voice_pick, delete_pick, train_pick], concurrency_limit=None)
    demo.load(pickers, outputs=[voice_pick, delete_pick, train_pick])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
