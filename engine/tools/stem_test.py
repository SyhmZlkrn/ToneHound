"""Profile-to-stem test, stage 1: does the fingerprint survive a real song?

Gate B measured profile-to-profile distances. The product does something
different -- it matches a *separated stem from a mixed, mastered, lossy
recording* against the index. Every Gate B number is optimistic by however much
that domain gap costs, and nothing so far has measured it.

Gate B's strongest features were the four LTAS bands above 10 kHz. They also
identified the *capturer* nearly as well as the amp, which is the signature of a
shortcut rather than a signal. This script is the test: mp3 lowpasses hard in
exactly that region, so if those bands are load-bearing, the matcher is resting
on the first thing a real recording destroys.

Ground truth is exact and free: degrade profile i's own render and ask whether
it still retrieves profile i. Self-retrieval rank is the honest metric -- if a
profile cannot find *itself* through the degradation, matching a real song is
hopeless.

Stage 2 (UVR separation) is a separate step; this isolates mix/master/codec so
their cost is not confounded with separation artefacts.

    PYTHONPATH=engine python engine/tools/stem_test.py --cache .cache/renders_v2
    PYTHONPATH=engine python engine/tools/stem_test.py --cache .cache/renders_di \
        --di "assets/user_di/Djent DI.wav"
"""

import argparse
import pathlib
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_source
from tonehound import match
from tonehound.config import GUITAR, SAMPLE_RATE, InstrumentConfig
from tonehound.features import (band_centers, fingerprint, third_octave_edges)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]
BEDS_DIR = pathlib.Path(
    r"C:\ProgramData\Ableton\.Live 11 Suite_updated\Resources\Core Library\Samples\Loops\Drums\Full"
)


# --------------------------------------------------------------------------
# The degradation chain a real recording imposes
# --------------------------------------------------------------------------


def mastering_eq(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A plausible mastering curve: gentle tilt plus low/high shelves."""
    sos = []
    # Broad tilt, +/- 3 dB across the band.
    tilt_db = rng.uniform(-3.0, 3.0)
    sos.append(signal.butter(1, 700.0, btype="low", fs=sr, output="sos"))
    y = x + (10 ** (tilt_db / 20.0) - 1.0) * signal.sosfilt(sos[0], x)

    # High shelf: mastering nearly always touches the top end.
    shelf_db = rng.uniform(-4.0, 4.0)
    hs = signal.butter(2, 6000.0, btype="high", fs=sr, output="sos")
    y = y + (10 ** (shelf_db / 20.0) - 1.0) * signal.sosfilt(hs, y)

    # Low shelf.
    ls_db = rng.uniform(-3.0, 3.0)
    ls = signal.butter(2, 120.0, btype="low", fs=sr, output="sos")
    y = y + (10 ** (ls_db / 20.0) - 1.0) * signal.sosfilt(ls, y)
    return y


def load_beds(n: int, length: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A backing bed of drum loops, tiled to `length`.

    Ableton Core Library material, used locally as test signal only -- it is not
    redistributable and must never end up in a shipped asset.
    """
    paths = sorted(BEDS_DIR.glob("*.wav"))
    if not paths:
        return np.zeros(length)
    bed = np.zeros(length)
    for path in rng.choice(paths, size=min(n, len(paths)), replace=False):
        audio, file_sr = sf.read(path, always_2d=True)
        audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = signal.resample_poly(audio, sr, file_sr)
        if len(audio) < length:
            audio = np.tile(audio, int(np.ceil(length / len(audio))))
        bed += audio[:length]
    peak = np.abs(bed).max()
    return bed / peak if peak > 0 else bed


def add_backing(x: np.ndarray, bed: np.ndarray, guitar_db: float) -> np.ndarray:
    """Mix guitar against the bed at a given relative level."""
    gx = x / (np.abs(x).max() + 1e-12)
    return gx * 10 ** (guitar_db / 20.0) + bed * 0.5


def limit(x: np.ndarray, ceiling: float = 0.95) -> np.ndarray:
    """Brickwall-ish limiting, as a master bus would apply."""
    peak = np.abs(x).max() + 1e-12
    y = x / peak * 1.6           # drive into the limiter
    return np.tanh(y) * ceiling


def codec_roundtrip(x: np.ndarray, sr: int, subtype: str | None) -> np.ndarray:
    """Real MP3 encode/decode through libsndfile.

    An approximation of codec damage would beg the question -- mp3's lowpass and
    high-band quantisation are precisely what is under test here.
    """
    peak = np.abs(x).max() + 1e-12
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "clip.mp3"
        kwargs = {"format": "MP3"}
        if subtype:
            kwargs["subtype"] = subtype
        sf.write(path, x / peak * 0.98, sr, **kwargs)
        y, out_sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)
    if out_sr != sr:
        y = signal.resample_poly(y, sr, out_sr)
    return y[: len(x)] * peak


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


# These four moved to ``tonehound.match`` when the live server needed them: the
# server and this script must rank with the *same* arithmetic, or "the app
# disagrees with the tool" becomes a bug nobody can localise. The old names are
# kept because a dozen scratch scripts import them.
zscore_ref = match.zscore_reference
vectors = match.whiten
cosine = match.cosine_distance
self_retrieval = match.self_retrieval


# --------------------------------------------------------------------------


def band_mask(cfg: InstrumentConfig, lo: float, hi: float) -> np.ndarray:
    c = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
    return (c >= lo) & (c < hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guitar-db", type=float, default=-6.0)
    probe_source.add_probe_args(ap)
    args = ap.parse_args()

    cfg, sr = GUITAR, SAMPLE_RATE
    # This stage reads renders rather than making them, so the probe is here to
    # be named in the report and to bind the cache -- a synthetic-probe cache
    # read by a --di run would answer a question nobody asked.
    src = probe_source.probe_from_args(args, sr)
    print(src.banner())
    probe_source.check_cache(args.cache, src, write=False)

    paths = sorted(args.cache.glob("*.npy"))
    if len(paths) < 2:
        print(f"FAIL: need cached renders in {args.cache}; run gate_b.py --cache first.")
        return 1

    names = [p.stem for p in paths]
    clean = [np.load(p) for p in paths]
    print(f"{len(clean)} cached renders, {len(clean[0]) / sr:.1f} s each\n")

    subtypes = sf.available_subtypes("MP3")
    subtype = "MPEG_LAYER_III" if "MPEG_LAYER_III" in subtypes else None
    print(f"MP3 subtypes available: {list(subtypes)} -> using {subtype}\n")

    # Clean index and the whitening statistics a real matcher would ship with.
    fps = [fingerprint(y, sr, cfg) for y in clean]
    idx_shape = np.array([f.shape_vector for f in fps])
    idx_scal = np.array([f.scalar_vector for f in fps])
    ref = (zscore_ref(idx_shape), zscore_ref(idx_scal))
    index = vectors(idx_shape, idx_scal, ref)

    rng = np.random.default_rng(args.seed)
    bed = load_beds(3, len(clean[0]), sr, rng)
    print(f"backing bed: {'built from Ableton drum loops' if bed.any() else 'NONE FOUND'}\n")

    # Ablation: each stage alone, then the full chain, so the damage is
    # attributable rather than a single lump.
    stages = {
        "clean (control)": lambda y, r: y,
        "mastering EQ": lambda y, r: mastering_eq(y, sr, r),
        "limiting": lambda y, r: limit(y),
        "mp3 128k": lambda y, r: codec_roundtrip(y, sr, subtype),
        "+ drums/bass bed": lambda y, r: add_backing(y, bed, args.guitar_db),
        "FULL CHAIN": lambda y, r: codec_roundtrip(
            limit(add_backing(mastering_eq(y, sr, r), bed, args.guitar_db)), sr, subtype),
    }

    hi = band_mask(cfg, 10_000.0, 1e9)
    lo_mid = band_mask(cfg, 80.0, 10_000.0)

    print(f"{'stage':<20}{'top-1':>8}{'top-5':>8}{'med rank':>10}"
          f"{'d>10kHz':>10}{'d 80Hz-10k':>12}")
    print("-" * 68)

    results = {}
    for label, fn in stages.items():
        r = np.random.default_rng(args.seed)
        degraded = [fn(y, r) for y in clean]
        dfps = [fingerprint(y, sr, cfg) for y in degraded]
        q = vectors(np.array([f.shape_vector for f in dfps]),
                    np.array([f.scalar_vector for f in dfps]), ref)
        ranks, top1, top5 = self_retrieval(q, index)

        # How far each band region moved, in dB, clean vs degraded.
        dl = np.array([d.ltas - c.ltas for d, c in zip(dfps, fps)])
        dl = dl - dl.mean(axis=1, keepdims=True)      # level-independent
        results[label] = (top1, top5, ranks, dl)
        print(f"{label:<20}{top1:>8.3f}{top5:>8.3f}{np.median(ranks):>10.1f}"
              f"{np.abs(dl[:, hi]).mean():>10.2f}{np.abs(dl[:, lo_mid]).mean():>12.2f}")

    # The specific question: are the >10 kHz bands load-bearing, and do they
    # survive? Re-run the full chain with them excluded.
    print()
    keep = ~hi
    full = stages["FULL CHAIN"]
    r = np.random.default_rng(args.seed)
    degraded = [full(y, r) for y in clean]
    dfps = [fingerprint(y, sr, cfg) for y in degraded]

    for label, mask in (("all 48 bands", np.ones(cfg.n_bands, bool)),
                        (">10 kHz bands dropped", keep),
                        (">10 kHz bands ONLY", hi)):
        sub_ref = (zscore_ref(idx_shape[:, mask]), zscore_ref(idx_scal))
        i2 = vectors(idx_shape[:, mask], idx_scal, sub_ref)
        q2 = vectors(np.array([f.shape_vector for f in dfps])[:, mask],
                     np.array([f.scalar_vector for f in dfps]), sub_ref)
        _, t1, t5 = self_retrieval(q2, i2)
        print(f"  full chain, {label:<24} top-1={t1:.3f}  top-5={t5:.3f}")

    top1_full = results["FULL CHAIN"][0]
    print(f"\n{'=' * 68}\nSTAGE 1 VERDICT: "
          f"{'PASS' if top1_full >= 0.80 else 'FAIL'} "
          f"(self-retrieval top-1 = {top1_full:.3f}, bar 0.80)")
    print("A profile that cannot retrieve its own degraded render will not match\n"
          "a real song. This is a necessary condition, not a sufficient one --\n"
          "stage 2 adds UVR separation on top.")
    return 0 if top1_full >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
