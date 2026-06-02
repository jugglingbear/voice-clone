# voice-clone

A small Poetry + Python CLI that synthesizes text in a voice **cloned from a short reference
clip**. It is built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI)
— a state-of-the-art, MIT-licensed, zero-shot voice cloning model.

You can clone a voice on the fly from a clip, or **enroll** a voice once into a small reusable
file and synthesize from it later. A **batch** mode loads the model a single time and renders
many lines in one go.

A reference clip of roughly **7–15 seconds of clean speech** gives the best clone.

---

## Requirements

- **Python** 3.10–3.12
- **[Poetry](https://python-poetry.org/)**
- **[ffmpeg](https://ffmpeg.org/)** on your `PATH` — required only for non-`.wav` references
  (`.mp3`, `.mp4`, `.ogg`, `.m4a`, `.amr`, `.opus`, `.3gp`, `.aac`, `.caf`, `.flac`, `.aiff`, and
  most other audio/video files phones and messaging apps produce), whose audio is
  extracted/converted to 24 kHz mono WAV. `.wav` references skip ffmpeg entirely.

  ```shell
  # macOS
  brew install ffmpeg

  # Debian / Ubuntu / WSL
  sudo apt-get install -y ffmpeg

  # Windows
  winget install ffmpeg
  ```

- A **GPU (CUDA)** or **Apple Silicon (MPS)** is optional but much faster. The tool auto-selects
  CUDA → MPS → CPU.

### Platform support

| Platform | Status |
|----------|--------|
| macOS (Apple Silicon / Intel) | Verified. Uses MPS when available. |
| Linux (native) | Expected to work (CUDA or CPU). |
| Windows (native) | Expected to work (CUDA or CPU). `sounddevice` bundles PortAudio, so playback works. |
| Windows (WSL2) | Works for file output (`-o`). Audio **playback** needs an audio bridge — WSLg on Windows 11 provides this; otherwise use `-o` to write a WAV. |

> All dependencies (`torch`, `torchaudio`, `sounddevice`, `ffmpeg`) are cross-platform. Only the
> macOS path has been exercised directly; the Windows/Linux notes above are based on dependency
> support, not a test run on those platforms.

---

## Install

```shell
cd voice-clone
make install        # verifies python3/poetry/ffmpeg, then runs `poetry install`
```

`make install` is a thin wrapper; `poetry install` works too. Run `make help` to see all targets.

The first synthesis run downloads the Chatterbox model weights (cached afterward).

---

## Usage

The CLI has three subcommands. Run `voice-clone <command> --help` for the full option list.

### `say` — synthesize one line

Provide the voice via either `-r/--reference` (a clip) or `--voice` (a saved file).

```shell
# Clone from a reference clip and write to a file
voice-clone say -r jane.wav "Hello, this is my cloned voice." -o out.wav

# Non-.wav references (.mp3, .mp4, .m4a, .amr, .opus, .ogg, ...) convert automatically via ffmpeg
voice-clone say -r clip.mp4 "Now I sound like the person in that video." -o out.wav

# No -o -> play the synthesized speech aloud immediately
voice-clone say -r jane.wav "Speak this out loud right now."

# Too fast? Slow it down (pitch-preserving). <1.0 is slower, >1.0 is faster.
voice-clone say --voice jane.voice --speed 0.85 "This reads at a calmer pace." -o out.wav
```

### `enroll` — save a reusable voice

Sampling a reference is a one-time step. Save it to a small `.voice` file and reuse it:

```shell
voice-clone enroll jane.mp4 -o jane.voice
voice-clone say --voice jane.voice "No reference clip needed this time."
```

> A `.voice` file skips re-reading the reference, but every invocation still reloads the model
> (the dominant cost). To render many lines quickly, use `batch`.

### `batch` — many lines, one model load

Loads the model **once**, then synthesizes every line. Lines come from `--input` (one per line;
blank lines and `#` comments are skipped) or, if omitted, are read interactively until EOF
(Ctrl-D).

```shell
# From a file, writing NNN.wav into a directory
voice-clone batch --voice jane.voice -i lines.txt -d out/

# Interactive, playing each line aloud
voice-clone batch -r jane.mp4
```

---

## Notes

- `--exaggeration` (0.0–1.0) controls the emotional intensity of the cloned voice on every
  command.
- `--speed` (0.25–4.0, default 1.0) adjusts the playback speed of `say` and `batch` output.
  Values below 1.0 slow the speech down, above 1.0 speed it up; pitch is preserved. This is a
  post-process time-stretch via ffmpeg's `atempo` filter — Chatterbox itself has no speech-rate
  parameter — so `--speed` requires ffmpeg on your `PATH`. Try `0.8`–`0.85` for recordings that
  read too fast.
- Chatterbox embeds an imperceptible Perth neural watermark in all generated audio so
  synthesized speech remains identifiable. Use responsibly and only clone voices you have
  permission to use.
