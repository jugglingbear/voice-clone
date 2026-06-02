"""Voice cloning CLI built on Chatterbox TTS.

Synthesize a line of text in a voice cloned from a short reference clip.

External dependency
-------------------
``ffmpeg`` must be installed and on your PATH. It is used to extract / convert
audio from ``.mp3``, ``.mp4``, and ``.ogg`` reference files into the 24 kHz mono
WAV that the model expects. ``.wav`` references are used directly (no ffmpeg
needed).

    macOS:          brew install ffmpeg
    Debian/Ubuntu:  sudo apt-get install -y ffmpeg

Usage
-----
    voice-clone say reference.ext "Text to mimic in cloned voice" [-o output.wav]

``reference.ext`` may be a ``.wav`` or any common audio/video format ffmpeg can
decode (``.mp3``, ``.mp4``, ``.ogg``, ``.m4a``, ``.amr``, ``.opus``, ``.3gp``,
``.caf``, ``.aac``, ``.flac``, ``.aiff``, ...). If ``-o/--output`` is omitted,
the synthesized audio is played aloud through the default output device instead
of being written to disk.

Reusable voices
---------------
Extracting the speaker characteristics from a reference clip is a one-time step.
``voice-clone enroll`` saves those characteristics to a small ``.voice`` file so
later runs can skip re-reading the reference::

    voice-clone enroll reference.mp4 -o sean.voice
    voice-clone say --voice sean.voice "Hello there"

Note: a ``.voice`` file avoids re-processing the reference, but every command
still reloads the ~2-3 GB model from scratch (the dominant cost). To synthesize
many lines quickly, use ``voice-clone batch``, which loads the model once and
then generates every line in a single process::

    voice-clone batch --voice sean.voice --input lines.txt --output-dir out/
    voice-clone batch reference.mp4            # interactive REPL
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import click

# Silence noisy (but harmless) third-party warnings before the heavy ML imports
# fire. These are all known deprecations / informational notices from our pinned
# dependencies; real warnings from any other source still surface. perth imports
# the legacy pkg_resources API (the reason we pin setuptools<81), and
# huggingface_hub warns about anonymous Hub requests even when weights are
# already cached locally.
warnings.filterwarnings("ignore", message=r".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=r".*LoRACompatibleLinear.*deprecated.*")
warnings.filterwarnings("ignore", message=r".*sdp_kernel\(\).*deprecated.*")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# Reference formats accepted for cloning. ``.wav`` is used directly; everything
# else is decoded to 24 kHz mono WAV by ffmpeg, so this list mirrors the common
# audio/video formats phones and messaging apps produce (iOS Voice Memos, Android
# recorders, WhatsApp/Telegram/Signal voice notes, MMS, etc.). ffmpeg decodes
# many more; add an extension here if you need it.
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".mp4",
    ".ogg",
    ".amr",  # SMS/MMS voice, older Android (AMR-NB)
    ".3gp",  # Android MMS audio/video (3GPP)
    ".3g2",  # 3GPP2
    ".m4a",  # iOS Voice Memos, Android recorders (AAC)
    ".aac",  # raw AAC
    ".caf",  # Apple Core Audio Format
    ".opus",  # WhatsApp / Telegram / Signal voice notes
    ".flac",  # lossless
    ".aiff",  # Apple uncompressed
    ".aif",  # Apple uncompressed (short suffix)
}
TARGET_SAMPLE_RATE = 24_000
VOICE_SUFFIX = ".voice"


def _select_device() -> str:
    """Pick the best available torch device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_audio_to_wav(reference: Path, dest_dir: Path) -> Path:
    """Convert a non-WAV reference into a 24 kHz mono WAV using ffmpeg.

    Returns the path to the extracted WAV. ``.wav`` inputs are returned as-is.
    """
    if reference.suffix.lower() == ".wav":
        return reference

    if shutil.which("ffmpeg") is None:
        raise click.ClickException(
            "ffmpeg is required to read non-WAV references but was not found on PATH. "
            "Install it (macOS: 'brew install ffmpeg', Debian/Ubuntu: "
            "'sudo apt-get install -y ffmpeg') or pass a .wav file instead."
        )

    out_wav = dest_dir / "reference_extracted.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(reference),
        "-vn",  # drop any video stream
        "-ac",
        "1",  # mono
        "-ar",
        str(TARGET_SAMPLE_RATE),
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not out_wav.exists():
        raise click.ClickException(f"ffmpeg failed to extract audio from {reference}:\n{result.stderr}")
    return out_wav


def _play_audio(wav, sample_rate: int) -> None:
    """Play a numpy waveform aloud through the default output device."""
    import sounddevice as sd

    sd.play(wav, sample_rate)
    sd.wait()


def _validate_reference(reference: Path) -> None:
    """Raise a ClickException if the reference has an unsupported extension."""
    ext = reference.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise click.ClickException(
            f"Unsupported reference extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _silence_sdpa_attention_log() -> None:
    """Drop transformers' harmless 'sdpa ... output_attentions' log message.

    This notice is emitted through transformers' logging system (not the
    ``warnings`` module), so a logging filter is required. Only this one message
    is suppressed; all other transformers log records still propagate.
    """
    import logging

    class _SdpaFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "does not support `output_attentions=True`" not in record.getMessage()

    logger = logging.getLogger("transformers")
    sdpa_filter = _SdpaFilter()
    logger.addFilter(sdpa_filter)
    for handler in logger.handlers:
        handler.addFilter(sdpa_filter)


def _load_model():
    """Load the Chatterbox model onto the best available device.

    Returns the loaded model. The heavy imports live here so ``--help`` and
    argument validation stay fast.
    """
    from chatterbox.tts import ChatterboxTTS

    _silence_sdpa_attention_log()

    device = _select_device()
    click.echo(f"Loading Chatterbox model on '{device}' (first run downloads weights)...", err=True)
    return ChatterboxTTS.from_pretrained(device=device)


def _apply_voice(model, *, reference: Path | None, voice: Path | None, exaggeration: float, tmp_dir: Path) -> None:
    """Condition ``model`` on either a saved ``.voice`` file or a reference clip.

    Exactly one of ``reference`` / ``voice`` must be provided. After this call
    ``model.conds`` is populated and ``model.generate(text)`` can be invoked
    without an ``audio_prompt_path``.
    """
    if voice is not None:
        from chatterbox.tts import Conditionals

        model.conds = Conditionals.load(voice, map_location=model.device).to(model.device)
        return

    assert reference is not None
    ref_wav = _extract_audio_to_wav(reference, tmp_dir)
    model.prepare_conditionals(str(ref_wav), exaggeration=exaggeration)


def _save_wav(wav, sample_rate: int, output: Path) -> None:
    """Write a generated waveform tensor to ``output`` as a WAV file."""
    import torchaudio as ta

    output.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output), wav, sample_rate)


def _play_wav(wav, sample_rate: int) -> None:
    """Play a generated waveform tensor aloud."""
    samples = wav.squeeze(0).detach().cpu().numpy()
    _play_audio(samples, sample_rate)


# Shared options for the source of the voice (a reference clip or a saved voice).
_reference_option = click.option(
    "-r",
    "--reference",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Reference clip to clone the voice from (.wav, or any audio/video ffmpeg can decode).",
)
_voice_option = click.option(
    "--voice",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=f"Use a saved {VOICE_SUFFIX} file (from 'enroll') instead of a reference clip.",
)
_exaggeration_option = click.option(
    "--exaggeration",
    type=float,
    default=0.5,
    show_default=True,
    help="Emotional intensity of the cloned voice (0.0-1.0).",
)


@click.group(context_settings={"max_content_width": 120})
def main() -> None:
    """Voice cloning CLI built on Chatterbox TTS.

    Clone a voice from a short reference clip (.wav, .mp3, .m4a, .amr, .opus, and
    most other audio/video formats) and synthesize text in that voice. Use
    'enroll' to save a reusable voice and 'batch' to synthesize many lines in one
    model-load.
    """


@main.command("enroll")
@click.argument("reference", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help=f"Where to write the {VOICE_SUFFIX} file.",
)
@_exaggeration_option
def enroll(reference: Path, output: Path, exaggeration: float) -> None:
    """Sample REFERENCE and save a reusable voice file.

    \b
    Example:
        voice-clone enroll sean.mp4 -o sean.voice
    """
    _validate_reference(reference)

    with tempfile.TemporaryDirectory() as tmp:
        model = _load_model()
        click.echo("Sampling reference clip...", err=True)
        _apply_voice(model, reference=reference, voice=None, exaggeration=exaggeration, tmp_dir=Path(tmp))

        output.parent.mkdir(parents=True, exist_ok=True)
        model.conds.save(output)
        click.echo(f"Wrote voice {output}")


@main.command("say")
@click.argument("text", type=str)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the synthesized WAV. If omitted, the audio is played aloud instead.",
)
@_reference_option
@_voice_option
@_exaggeration_option
def say(text: str, output: Path | None, reference: Path | None, voice: Path | None, exaggeration: float) -> None:
    """Synthesize TEXT in a cloned voice.

    Provide the voice via either --reference (a clip) or --voice (a saved file).

    \b
    Examples:
        voice-clone say -r sean.mp4 "Hello there" -o hello.wav
        voice-clone say --voice sean.voice "Hello there"
    """
    if (reference is None) == (voice is None):
        raise click.UsageError("Provide exactly one of --reference or --voice.")
    if reference is not None:
        _validate_reference(reference)

    with tempfile.TemporaryDirectory() as tmp:
        model = _load_model()
        _apply_voice(model, reference=reference, voice=voice, exaggeration=exaggeration, tmp_dir=Path(tmp))

        click.echo("Synthesizing...", err=True)
        wav = model.generate(text)

        if output is not None:
            _save_wav(wav, model.sr, output)
            click.echo(f"Wrote {output}")
        else:
            click.echo("Playing audio aloud...", err=True)
            _play_wav(wav, model.sr)


@main.command("batch")
@click.option(
    "-i",
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Text file with one line to synthesize per line. If omitted, read lines interactively.",
)
@click.option(
    "-d",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory to write NNN.wav files into. If omitted, each line is played aloud.",
)
@_reference_option
@_voice_option
@_exaggeration_option
def batch(
    input_file: Path | None,
    output_dir: Path | None,
    reference: Path | None,
    voice: Path | None,
    exaggeration: float,
) -> None:
    """Synthesize many lines, loading the model only once.

    Provide the voice via either --reference (a clip) or --voice (a saved file).
    Lines come from --input (one per line; blank lines and '#' comments are
    skipped) or, if omitted, are read interactively until EOF (Ctrl-D).

    \b
    Examples:
        voice-clone batch --voice sean.voice -i lines.txt -d out/
        voice-clone batch -r sean.mp4            # interactive, plays each line
    """
    if (reference is None) == (voice is None):
        raise click.UsageError("Provide exactly one of --reference or --voice.")
    if reference is not None:
        _validate_reference(reference)

    lines: list[str] | None = None
    if input_file is not None:
        lines = [
            stripped
            for raw in input_file.read_text(encoding="utf-8").splitlines()
            if (stripped := raw.strip()) and not stripped.startswith("#")
        ]
        if not lines:
            raise click.ClickException(f"No non-empty, non-comment lines found in {input_file}.")

    with tempfile.TemporaryDirectory() as tmp:
        model = _load_model()
        _apply_voice(model, reference=reference, voice=voice, exaggeration=exaggeration, tmp_dir=Path(tmp))

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        if lines is not None:
            _batch_from_lines(model, lines, output_dir)
        else:
            _batch_interactive(model, output_dir)


def _batch_from_lines(model, lines: list[str], output_dir: Path | None) -> None:
    """Synthesize a fixed list of lines, saving or playing each."""
    width = len(str(len(lines)))
    for index, text in enumerate(lines, start=1):
        click.echo(f"[{index}/{len(lines)}] {text}", err=True)
        wav = model.generate(text)
        if output_dir is not None:
            out = output_dir / f"{index:0{width}d}.wav"
            _save_wav(wav, model.sr, out)
            click.echo(f"  wrote {out}")
        else:
            _play_wav(wav, model.sr)


def _batch_interactive(model, output_dir: Path | None) -> None:
    """Read lines from the prompt until EOF, synthesizing each."""
    click.echo("Enter text to synthesize, one line at a time (Ctrl-D to finish).", err=True)
    index = 0
    while True:
        try:
            text = click.prompt("text", prompt_suffix="> ")
        except (EOFError, click.Abort):
            click.echo("", err=True)
            break
        text = text.strip()
        if not text:
            continue
        index += 1
        wav = model.generate(text)
        if output_dir is not None:
            out = output_dir / f"{index:03d}.wav"
            _save_wav(wav, model.sr, out)
            click.echo(f"  wrote {out}", err=True)
        else:
            _play_wav(wav, model.sr)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
