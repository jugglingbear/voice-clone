# voice-clone

A small Poetry + Python CLI that synthesizes a line of text in a voice **cloned from a short
reference clip**. It is built on [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
(Resemble AI) — a state-of-the-art, MIT-licensed, zero-shot voice cloning model.

```shell
voice-clone reference.ext "Text to mimic in cloned voice" [-o /path/to/output.wav]
```

- `reference.ext` may be **`.wav`**, **`.mp3`**, or **`.mp4`**.
- With `-o/--output`, the result is written to that WAV file.
- **Without** `-o`, the audio is **played aloud** through your default output device.

A reference clip of roughly **7–15 seconds of clean speech** gives the best clone.

---

## Requirements

- **Python** 3.10–3.12
- **[ffmpeg](https://ffmpeg.org/)** on your `PATH` — required only for `.mp3` and `.mp4`
  references (used to extract/convert their audio to 24 kHz mono WAV). `.wav` references skip
  ffmpeg entirely.

  ```shell
  # macOS
  brew install ffmpeg

  # Debian / Ubuntu
  sudo apt-get install -y ffmpeg
  ```

- A **GPU (CUDA)** or **Apple Silicon (MPS)** is optional but much faster. The tool auto-selects
  CUDA → MPS → CPU.

---

## Install

```shell
cd temp/voice-clone
poetry install
```

The first synthesis run downloads the Chatterbox model weights (cached afterward).

---

## Usage

```shell
# Clone from a WAV reference, write the result to a file
voice-clone samples/jane.wav "Hello, this is my cloned voice." -o out.wav

# Clone from an MP3 reference (ffmpeg extracts the audio automatically)
voice-clone samples/jane.mp3 "Reading this sentence aloud."  -o out.wav

# Clone from an MP4 video (audio track is extracted via ffmpeg)
voice-clone samples/clip.mp4 "Now I sound like the person in that video." -o out.wav

# No output path -> play the synthesized speech aloud immediately
voice-clone samples/jane.wav "Speak this out loud right now."
```

Run `voice-clone --help` for the full option list.

---

## How it works

1. If the reference is `.mp3`/`.mp4`, ffmpeg converts it to a temporary 24 kHz mono WAV
   (`ffmpeg -i ref -vn -ac 1 -ar 24000 out.wav`). `.wav` files are used directly.
2. Chatterbox loads on the best available device and clones the reference voice.
3. The generated waveform is either saved with `torchaudio` (when `-o` is given) or played
   through [`sounddevice`](https://python-sounddevice.readthedocs.io/) (when it is not).

> **Note:** Chatterbox embeds an imperceptible Perth neural watermark in all generated audio so
> synthesized speech remains identifiable. Use responsibly and only clone voices you have
> permission to use.
