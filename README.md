# Voice-to-Voice

A local web app that re-speaks a recording in another voice. You save a voice from a short clean recording, can fine-tune the model on more audio of that person, then upload speech and get it back in that voice.

It wraps [Seed-VC](https://github.com/Plachtaa/seed-vc) (v1 offline speech model) and runs entirely on your machine. Nothing is uploaded anywhere.

## What this adds over Seed-VC

Seed-VC provides the model, command-line scripts and demo pages. Used alone, you upload a reference every run, prepare training clips by hand and run `train.py` yourself. This app turns that into one guided workflow:

- **Voice library:** save named voices once and reuse them, with a required consent check for each voice.
- **Fine-tuning from the UI:** add recordings of any length. They are split at pauses into clips Seed-VC can train on, and the trained model is stored with the voice and used automatically.
- **Audio prep:** any format ffmpeg can read is converted for you, and references are trimmed to the length Seed-VC uses.
- **History:** each conversion is kept in its own folder.
- **Safety on one GPU:** only one job runs at a time, and a voice can't be deleted while it is training.
- **Reproducible setup:** Seed-VC is pinned to one commit, dependencies are locked, and it is tested on a 6 GB laptop GPU.

It keeps only offline speech conversion. Seed-VC's real-time, singing and v2 modes are left out. The v1 model with 50 diffusion steps sounded best in listening tests against v2 on the voices tried here.

## Requirements

- Linux or WSL2 with an NVIDIA GPU with **6 GB VRAM or more** and a recent driver (tested on an RTX 3060 Laptop, 6 GB)
- `ffmpeg` and `git` on the `PATH`
- [`uv`](https://docs.astral.sh/uv/). It installs Python 3.10 and all dependencies for you
- About 10 GB of free disk: about 6 GB of Python packages, plus about 2 GB of models downloaded on first use

## Setup

From the project folder:

1. Fetch Seed-VC at the pinned commit. The upstream repository is archived, so this commit is final:

   ```bash
   git clone https://github.com/Plachtaa/seed-vc.git third_party/seed-vc
   ```

   ```bash
   git -C third_party/seed-vc checkout 51383efd921027683c89e5348211d93ff12ac2a8
   ```

2. Install Python 3.10 and the dependencies (PyTorch 2.4 with CUDA 12.4):

   ```bash
   uv sync
   ```

3. Check that the GPU is visible:

   ```bash
   uv run pytest -q tests/test_env.py
   ```

## Run

```bash
uv run python app.py
```

Open <http://127.0.0.1:7860>. The app listens on your machine only. The first conversion downloads the models into `third_party/seed-vc/checkpoints/` and takes several minutes. Later conversions start in seconds.

## Usage

1. **Voices tab:** enter a name, upload a reference recording of the target voice, tick the consent box, and click **Save voice**.
   - The reference should be 1–30 s of clean speech. Longer clips are trimmed to 30 s, and Seed-VC uses only the first 25 s.
2. **Train tab (optional, recommended):** pick the voice, add more recordings of the same person, and click **Start training**.
   - Recordings can be any length. They are split at pauses into clips of 6 s or less.
   - 400 steps take about 6 minutes on a 6 GB GPU.
   - Afterwards the voice shows as "· fine-tuned", and conversions use the trained model automatically.
   - Extra recordings are kept, so retraining later uses them too.
3. **Convert tab:** pick a voice, then upload or record the speech to convert (up to 10 minutes). Click **Convert**, then play or download the result.
   - The quality slider trades speed for quality: 10 is fastest, 50 is best and the default.

Saved voices live in `data/voices/<name>/`. Each conversion is kept in its own folder in `data/outputs/`.

## Getting good results

- **More clean audio of the target voice helps most.** Two minutes plus fine-tuning sounded clearly better than 30 s with no training.
- **Use clean recordings:** one speaker, no music or background noise, little room echo, and natural speech. The reference language doesn't need to match the speech you convert. A Japanese reference with English speech works.
- **Pick source speech close to the target.** Similar pitch and a calm delivery convert best. Very expressive or high-pitched sources leave more of the original speaker audible.
- **Expect these speeds:** about 30–40 s for a 12 s clip, and about 4.5 min for a 3 min recording, at 50 steps with a 25 s reference.

## Limits

- Offline file conversion only. There is no real-time microphone mode and no singing (Seed-VC supports both, but this app doesn't expose them).
- The model reloads for every conversion, which adds about 15–20 s.
- One GPU job at a time. Converting and deleting voices are blocked while a voice is training.

## Tests

```bash
uv run pytest -q
```

Add `-m "not gpu"` to skip the tests that need the GPU and models.

## Consent and license

Clone only your own voice, or voices whose owners have agreed. The app asks you to confirm consent for every saved voice.

This project's code is under the [MIT License](LICENSE). Seed-VC is a separate program under GPL-3.0: you download it yourself during setup, and this app only runs its command-line scripts. If you redistribute Seed-VC together with this app, follow GPL-3.0 for Seed-VC.
