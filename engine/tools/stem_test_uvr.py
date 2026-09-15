"""Profile-to-stem test, stage 2: add UVR separation.

Stage 1 (``stem_test.py``) measures what mastering, limiting and mp3 cost the
fingerprint. This adds the step the product actually performs: pull the guitar
back out of the mix with UVR, then match the *separated stem* against the clean
index.

Stage 1 measured this, and the framing here was originally wrong: mastering,
limiting and mp3 turned out to be nearly free (top-1 stayed 0.97-1.00), while
mixing in drums and bass alone collapsed top-1 from 1.000 to 0.147. Instrument
bleed IS the domain gap, and separation is not extra damage piled on top -- it
is the remedy for the only thing that actually broke.

So this measures recovery, not further loss:

    1.000  clean render                    ceiling
    0.118  mixed, mastered, mp3, no UVR    floor (stage 1)
      ?    + UVR guitar stem               how far separation climbs back

The caveat is that the guitar stem is the weakest output these models produce
-- a 6-stem extra, not one of the well-trained core four -- so recovery may be
partial.

Ground truth stays exact: degrade and separate profile i's own render, then ask
whether it still retrieves profile i.

    PYTHONPATH=engine python engine/tools/stem_test_uvr.py --limit 12
    PYTHONPATH=engine python engine/tools/stem_test_uvr.py --limit 12 \
        --cache .cache/renders_di --di "assets/user_di/Djent DI.wav"
"""

import argparse
import pathlib
import shutil
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_source
from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import fingerprint

from stem_test import (add_backing, band_mask, codec_roundtrip, limit, load_beds,
                       mastering_eq, self_retrieval, vectors, zscore_ref)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]

# The 6-stem Demucs model is the only one here that emits a guitar stem at all.
# UVR's stronger models (MDX23C etc.) are vocal/instrumental splitters, so on a
# guitar+drums mix with no vocals they have nothing useful to contribute -- the
# cascade's first stage is skipped here deliberately.
GUITAR_MODEL = "htdemucs_6s.yaml"


def separate(mix: np.ndarray, sr: int, separator, workdir: pathlib.Path,
             tag: str) -> np.ndarray | None:
    """Run UVR and return the guitar stem, or None if it produced nothing."""
    src = workdir / f"{tag}.wav"
    peak = np.abs(mix).max() + 1e-12
    sf.write(src, mix / peak * 0.98, sr)

    outputs = separator.separate(str(src))
    guitar = [p for p in outputs if "guitar" in str(p).lower()]
    try:
        if not guitar:
            return None

        path = pathlib.Path(guitar[0])
        if not path.is_absolute():
            path = workdir / path
        y, file_sr = sf.read(path, always_2d=True)
        y = y.mean(axis=1)
        if file_sr != sr:
            y = signal.resample_poly(y, sr, file_sr)
        return y * peak
    finally:
        # htdemucs_6s writes all six stems per call and nothing deletes them, so
        # a 68-separation run leaves ~1.5 GB of temp behind. On a full disk the
        # next write is silently short, which is how a truncated render got into
        # a cache before.
        for p in [src, *(pathlib.Path(o) if pathlib.Path(o).is_absolute()
                         else workdir / o for o in outputs)]:
            p.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_v2")
    ap.add_argument("--limit", type=int, default=12,
                    help="profiles to test; separation is slow, so default to a subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guitar-db", type=float, default=-6.0)
    ap.add_argument("--seconds", type=float, default=9.0,
                    help="clip length; clamped to one probe level pass")
    probe_source.add_probe_args(ap)
    args = ap.parse_args()

    cfg, sr = GUITAR, SAMPLE_RATE
    src = probe_source.probe_from_args(args, sr)
    print(src.banner())
    probe_source.check_cache(args.cache, src, write=False)

    paths = sorted(args.cache.glob("*.npy"))
    if len(paths) < 2:
        print(f"FAIL: need cached renders in {args.cache}; run gate_b.py --cache first.")
        return 1
    paths = paths[: args.limit]

    # Take the clip from the LOUDEST level pass. Slicing from t=0 lands in
    # the -36 dBFS pass, which after mixing sits ~42 dB under the bed -- the
    # separator rightly returns near-silence and the test then measures
    # nothing at all. Only the loudest pass represents a guitar that is
    # actually sitting in a mix.
    loud = src.slices[-1]
    n = min(int(args.seconds * sr), loud.stop - loud.start)
    clean = [np.load(p)[loud.start:loud.start + n] for p in paths]
    print(f"{len(clean)} profiles, {n / sr:.1f} s clips from the loudest "
          f"probe pass (t={loud.start / sr:.1f}s)")
    print()

    # Index built from the SAME clips, so clip length is not a confound.
    fps = [fingerprint(y, sr, cfg) for y in clean]
    idx_shape = np.array([f.shape_vector for f in fps])
    idx_scal = np.array([f.scalar_vector for f in fps])
    ref = (zscore_ref(idx_shape), zscore_ref(idx_scal))
    index = vectors(idx_shape, idx_scal, ref)

    rng = np.random.default_rng(args.seed)
    bed = load_beds(3, len(clean[0]), sr, rng)
    subtypes = sf.available_subtypes("MP3")
    subtype = "MPEG_LAYER_III" if "MPEG_LAYER_III" in subtypes else None

    from audio_separator.separator import Separator

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_uvr_"))
    try:
        separator = Separator(log_level=40, output_dir=str(workdir),
                              model_file_dir=str(ROOT / ".cache" / "uvr_models"),
                              use_soundfile=True)   # no ffmpeg on this machine
        print(f"loading {GUITAR_MODEL} (first run downloads it) ...")
        separator.load_model(model_filename=GUITAR_MODEL)
        print("model ready\n")

        mixes, separated, failed = [], [], []
        for i, y in enumerate(clean):
            r = np.random.default_rng(args.seed)
            mix = codec_roundtrip(
                limit(add_backing(mastering_eq(y, sr, r), bed, args.guitar_db)), sr, subtype)
            mixes.append(mix)
            stem = separate(mix, sr, separator, workdir, f"p{i:03d}")
            if stem is None:
                failed.append(paths[i].stem)
                separated.append(mix)          # fall back so shapes line up
            else:
                n = min(len(stem), len(y))
                separated.append(stem[:n])
            print(f"  [{i + 1}/{len(clean)}] {paths[i].stem[:52]}")

        if failed:
            print(f"\nWARNING: no guitar stem for {len(failed)} clip(s): {failed[:3]}")

        print(f"\n{'condition':<34}{'top-1':>8}{'top-5':>8}{'med rank':>10}")
        print("-" * 60)
        hi = band_mask(cfg, 10_000.0, 1e9)

        for label, sigs in (("mixed, mastered, mp3 (stage 1)", mixes),
                            ("+ UVR guitar stem (stage 2)", separated)):
            dfps = [fingerprint(s, sr, cfg) for s in sigs]
            q = vectors(np.array([f.shape_vector for f in dfps]),
                        np.array([f.scalar_vector for f in dfps]), ref)
            ranks, t1, t5 = self_retrieval(q, index)
            print(f"{label:<34}{t1:>8.3f}{t5:>8.3f}{np.median(ranks):>10.1f}")

            for sub, mask in ((">10 kHz dropped", ~hi), (">10 kHz only", hi)):
                sref = (zscore_ref(idx_shape[:, mask]), zscore_ref(idx_scal))
                i2 = vectors(idx_shape[:, mask], idx_scal, sref)
                q2 = vectors(np.array([f.shape_vector for f in dfps])[:, mask],
                             np.array([f.scalar_vector for f in dfps]), sref)
                _, s1, s5 = self_retrieval(q2, i2)
                print(f"{'    ' + sub:<34}{s1:>8.3f}{s5:>8.3f}")

        print()
        print("Recovery toward the clean ceiling (1.000) from the unseparated"
              " floor (0.118) is the number that matters. Substantial recovery means"
              " the feature space is the binding constraint and the learned embedding"
              " is the next move; little recovery means separation quality caps the"
              " whole product, and no feature work gets past it.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
