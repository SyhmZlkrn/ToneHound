"""Pull the guitar out of a full mix.

Step 1 of the pipeline, and the whole wall clock: htdemucs_6s runs at roughly
realtime, everything after it is seconds. It is the same model StemRoller
bundles -- StemRoller spawns the `demucs` executable with `-n <model>`, this
drives the same weights through `audio_separator`, which is already a
dependency and already has htdemucs_6s in `.cache/uvr_models`. The `demucs` CLI
is kept as a fallback so a machine with StemRoller's toolchain and without
`audio_separator` still works.

**htdemucs_6s is the only choice here.** It is the one Demucs variant with a
`guitar` output at all; the four-stem models emit `other`, and UVR's stronger
models are vocal/instrumental splitters with nothing to say about a guitar.

**The fallback chain matters more than it looks.** On some material the guitar
stem comes back empty -- a clean tone buried under a synth, a mix where the
model hears the guitar as keys. The old behaviour was to fall back to the raw
mix, which hands the matcher drums and vocals and calls it a guitar tone.
`other` is tried first instead: it is what htdemucs_6s puts guitar into when it
does not recognise it as one, and it has at least had the drums, bass and
vocals removed. Which stem was used is reported, never hidden, because a
ranking from `other` deserves a different amount of trust than one from
`guitar`.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .audioio import to_rate
from .config import SAMPLE_RATE

GUITAR_MODEL = "htdemucs_6s"
GUITAR_MODEL_FILE = "htdemucs_6s.yaml"
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")

FALLBACK_ORDER = ("guitar", "other")
"""Stems tried, in order, when asked for the guitar."""

MIN_STEM_RMS = 1e-4
"""Below this a stem is silence with rounding on top, and counts as absent."""


class SeparationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class Separated:
    """The result of one separation, and how much to trust it."""

    samples: np.ndarray
    stem: str                  # which stem this actually is
    requested: str = "guitar"
    stems: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        """True when the separator gave us the stem that was asked for."""
        return self.stem == self.requested

    @property
    def note(self) -> str:
        """One sentence for the user about what they are about to hear."""
        if self.ok:
            return "isolated guitar"
        if self.stem == "other":
            return ("no guitar stem was produced, so this is the 'other' stem -- "
                    "drums, bass and vocals removed, but it may not be only guitar")
        return ("no guitar stem was produced, so this is the raw mix -- "
                "the ranking below is measuring the whole song, not a guitar")


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class Separator:
    """A loaded htdemucs_6s, reusable across calls.

    Loading costs seconds, so the server keeps one of these on a single-worker
    pool rather than paying for it per match.
    """

    def __init__(self, model_dir: str | pathlib.Path,
                 work_dir: str | pathlib.Path | None = None,
                 model: str = GUITAR_MODEL_FILE, *,
                 autocast: bool = False, shifts: int | None = None) -> None:
        self.model_dir = pathlib.Path(model_dir)
        self.work_dir = pathlib.Path(work_dir or tempfile.mkdtemp(prefix="tonehound_sep_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.autocast = autocast
        self.shifts = shifts
        self._impl: Any = None

    def _load(self) -> Any:
        """Import `audio_separator` lazily: it pulls in onnxruntime and costs
        seconds, which nothing that only ranks should have to pay.

        There is no device argument to pass: `audio_separator` calls
        `torch.cuda.is_available()` itself and moves htdemucs onto the GPU when
        the installed torch has CUDA. That means the only thing that decides
        where separation runs is which torch build is installed, which is worth
        knowing when it is unexpectedly slow.

        `shifts` is demucs's own quality/time dial -- it separates the track
        several times at different offsets and averages. The library defaults to
        2; dropping to 1 roughly halves the wall clock, so it is exposed rather
        than buried.
        """
        if self._impl is None:
            from audio_separator.separator import Separator as UVRSeparator

            kwargs: dict[str, Any] = {}
            if self.shifts is not None:
                kwargs["demucs_params"] = {"segment_size": "Default",
                                           "shifts": self.shifts,
                                           "overlap": 0.25,
                                           "segments_enabled": True}
            impl = UVRSeparator(log_level=40, output_dir=str(self.work_dir),
                                model_file_dir=str(self.model_dir),
                                use_soundfile=True,
                                use_autocast=self.autocast, **kwargs)
            impl.load_model(model_filename=self.model)
            self._impl = impl
        return self._impl

    @property
    def device(self) -> str:
        """Where separation will actually run. Loads the model to find out."""
        return str(getattr(self._load(), "torch_device", "unknown"))

    def split(self, x: np.ndarray, sr: int = SAMPLE_RATE,
              tag: str = "mix") -> dict[str, np.ndarray]:
        """Every stem the model produced, keyed by name, at ``sr``."""
        import soundfile as sf

        self.work_dir.mkdir(parents=True, exist_ok=True)
        src = self.work_dir / f"{tag}.wav"
        peak = float(np.abs(x).max()) + 1e-12
        sf.write(src, np.asarray(x, dtype=np.float64) / peak * 0.98, sr)

        outputs = self._load().separate(str(src))
        paths = [p if (p := pathlib.Path(o)).is_absolute() else self.work_dir / o
                 for o in outputs]
        try:
            return _read_stems(paths, sr, peak)
        finally:
            # htdemucs_6s writes all six stems per call and nothing deletes
            # them. A long session would otherwise leave gigabytes behind, and
            # a full disk shows up as a silently truncated write, not an error.
            for p in [src, *paths]:
                p.unlink(missing_ok=True)


def _read_stems(paths: Sequence[pathlib.Path], sr: int,
                scale: float) -> dict[str, np.ndarray]:
    import soundfile as sf

    out: dict[str, np.ndarray] = {}
    for path in paths:
        if not path.exists():
            continue
        name = next((s for s in STEMS if s in path.stem.lower()), None)
        if name is None:
            continue
        y, file_sr = sf.read(path, always_2d=True)
        y = to_rate(y, int(file_sr), sr)
        out[name] = np.asarray(y * scale, dtype=np.float32)
    return out


def _split_with_demucs_cli(x: np.ndarray, sr: int, work_dir: pathlib.Path,
                           model: str = GUITAR_MODEL) -> dict[str, np.ndarray]:
    """StemRoller's path: `demucs <file> -n htdemucs_6s`.

    Only reached when `audio_separator` will not import. Demucs writes into
    ``<out>/<model>/<track>/<stem>.wav``, which is why this globs rather than
    reading a returned list.
    """
    import soundfile as sf

    exe = shutil.which("demucs") or shutil.which("demucs-cxfreeze")
    if exe is None:
        raise SeparationError(
            "separator_missing",
            "separation needs either the `audio-separator` package or the "
            "`demucs` executable on PATH")

    work_dir.mkdir(parents=True, exist_ok=True)
    src = work_dir / "mix.wav"
    peak = float(np.abs(x).max()) + 1e-12
    sf.write(src, np.asarray(x, dtype=np.float64) / peak * 0.98, sr)
    out_dir = work_dir / "separated"

    args = [exe, str(src), "-n", model, "-o", str(out_dir)]
    proc = subprocess.run(args, capture_output=True, text=True, check=False,
                          cwd=str(work_dir))
    if proc.returncode != 0:
        raise SeparationError("separation_failed",
                              f"demucs failed: {proc.stderr.strip()[-300:]}")
    return _read_stems(sorted(out_dir.rglob("*.wav")), sr, peak)


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def isolate(x: np.ndarray, sr: int = SAMPLE_RATE, *,
            separator: Separator | None = None,
            model_dir: str | pathlib.Path | None = None,
            want: str = "guitar",
            fallbacks: Sequence[str] = FALLBACK_ORDER,
            tag: str = "mix", keep_stems: bool = False,
            preserve_stereo: bool = False) -> Separated:
    """Separate ``x`` and return the guitar, or the best stand-in for it.

    Never raises on "no guitar in this song": that is a result, and `Separated`
    says which stem came back. It raises only when separation itself failed.
    """
    x = np.asarray(x, dtype=np.float64)
    # Demucs receives the original stereo image. Folding before separation
    # discards panning/phase cues and can cancel wide guitars.

    if separator is not None:
        stems = separator.split(x, sr, tag)
    elif model_dir is not None:
        try:
            stems = Separator(model_dir).split(x, sr, tag)
        except ImportError:
            stems = _split_with_demucs_cli(
                x, sr, pathlib.Path(tempfile.mkdtemp(prefix="tonehound_sep_")))
    else:
        raise SeparationError("no_separator",
                              "isolate() needs either a Separator or a model_dir")

    order = [want, *(s for s in fallbacks if s != want)]
    for name in order:
        stem = stems.get(name)
        if stem is not None and float(np.sqrt(np.mean(np.square(stem)))) > MIN_STEM_RMS:
            samples = stem if preserve_stereo or stem.ndim == 1 else stem.mean(axis=1)
            return Separated(samples=np.asarray(samples, dtype=np.float32), stem=name,
                             requested=want,
                             stems=stems if keep_stems else {})

    samples = x if preserve_stereo or x.ndim == 1 else x.mean(axis=1)
    return Separated(samples=np.asarray(samples, dtype=np.float32), stem="mix",
                     requested=want, stems=stems if keep_stems else {})
