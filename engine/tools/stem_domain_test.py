"""Is separation damage systematic or random -- and can domain-matching fix it?

Stage 2 established that UVR separation consistently *worsens* matching
(top-1 0.147 -> 0.029 on an audible guitar, and the same direction in two other
configurations). That is only fatal if the damage is random. It decomposes as

    LTAS(separated) - LTAS(clean render)  =  systematic  +  per-profile residual

A large systematic term is not information loss at all -- it is a domain shift,
and the cure is to fingerprint the index through the same separator so query and
index live in the same domain. That is cheap: the index is built offline once.
A large residual term means separation genuinely destroys per-amp detail, and no
indexing trick recovers it.

The domain-matched retrieval test must avoid being circular. Index and query are
built over *different backing beds*, so a match cannot come from both sides
having seen identical interference.

    PATH=.cache/bin:$PATH PYTHONPATH=engine python engine/tools/stem_domain_test.py
    ... --cache .cache/renders_di --di "assets/user_di/Djent DI.wav"   # real DI probe
"""

import argparse
import pathlib
import shutil
import sys
import tempfile
import warnings

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_source
from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import band_centers, fingerprint, third_octave_edges

from stem_test import (add_backing, codec_roundtrip, limit, load_beds, mastering_eq,
                       self_retrieval, vectors, zscore_ref)
from stem_test_uvr import GUITAR_MODEL, separate

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]


def build(clean, bed, sr, cfg, separator, workdir, tag, guitar_db, subtype,
          eq_base=1000):
    """Mix -> master -> codec -> separate, returning (mixed, separated).

    ``eq_base`` seeds the per-profile mastering curve. Index and query must draw
    from *different* bases: a shared base gives profile i the same random tilt
    and shelves on both sides, which is a per-profile watermark the matcher can
    ride instead of amp character -- and the product can never know the
    mastering of a song it is asked about.
    """
    mixes, stems = [], []
    for i, y in enumerate(clean):
        r = np.random.default_rng(eq_base + i)       # per-profile master settings
        mix = codec_roundtrip(
            limit(add_backing(mastering_eq(y, sr, r), bed, guitar_db)), sr, subtype)
        stem = separate(mix, sr, separator, workdir, f"{tag}{i:03d}")
        mixes.append(mix)
        stems.append(mix if stem is None else stem[: len(y)])
        print(f"  [{tag}] {i + 1}/{len(clean)}", flush=True)
    return mixes, stems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_v2")
    ap.add_argument("--limit", type=int, default=34)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--guitar-db", type=float, default=-6.0)
    ap.add_argument("--query-eq-base", type=int, default=1000,
                    help="mastering-EQ seed base for the query side; 1000 (default) "
                         "matches the index and leaks a per-profile watermark, so "
                         "pass something else for an honest domain-matched number")
    probe_source.add_probe_args(ap)
    args = ap.parse_args()

    cfg, sr = GUITAR, SAMPLE_RATE
    src = probe_source.probe_from_args(args, sr)
    print(src.banner())
    probe_source.check_cache(args.cache, src, write=False)

    paths = sorted(args.cache.glob("*.npy"))[: args.limit]
    loud = src.slices[-1]
    n = min(int(args.seconds * sr), loud.stop - loud.start)
    clean = [np.load(p)[loud.start:loud.start + n] for p in paths]
    print(f"{len(clean)} profiles, {n / sr:.1f}s from the loudest pass\n")

    subtypes = sf.available_subtypes("MP3")
    subtype = "MPEG_LAYER_III" if "MPEG_LAYER_III" in subtypes else None

    # Two different backing beds: index and query must not share interference.
    bed_a = load_beds(3, n, sr, np.random.default_rng(0))
    bed_b = load_beds(3, n, sr, np.random.default_rng(7))
    print(f"beds differ: {not np.allclose(bed_a, bed_b)}")
    shared = args.query_eq_base == 1000
    print(f"mastering EQ shared between index and query: {shared}"
          f"{'   <-- CONFOUNDED' if shared else ''}\n")

    from audio_separator.separator import Separator
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_dom_"))
    try:
        sep = Separator(log_level=40, output_dir=str(workdir),
                        model_file_dir=str(ROOT / ".cache" / "uvr_models"),
                        use_soundfile=True)
        sep.load_model(model_filename=GUITAR_MODEL)

        _, stems_a = build(clean, bed_a, sr, cfg, sep, workdir, "A", args.guitar_db,
                           subtype, eq_base=1000)
        mixes_b, stems_b = build(clean, bed_b, sr, cfg, sep, workdir, "B", args.guitar_db,
                                 subtype, eq_base=args.query_eq_base)

        fp_clean = [fingerprint(y, sr, cfg) for y in clean]
        fp_a = [fingerprint(y, sr, cfg) for y in stems_a]
        fp_b = [fingerprint(y, sr, cfg) for y in stems_b]
        fp_mix_b = [fingerprint(y, sr, cfg) for y in mixes_b]

        # ---- systematic vs residual ------------------------------------
        d = np.array([b.ltas - c.ltas for b, c in zip(fp_b, fp_clean)])
        d = d - d.mean(axis=1, keepdims=True)          # level-independent
        systematic = d.mean(axis=0)
        residual = d - systematic

        centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
        band = (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)
        sys_rms = float(np.sqrt((systematic[band] ** 2).mean()))
        res_rms = float(np.sqrt((residual[:, band] ** 2).mean()))

        print("\n" + "=" * 66)
        print("SEPARATION DAMAGE, decomposed (analysis band)")
        print("=" * 66)
        print(f"  systematic shift (same for every profile) : {sys_rms:6.2f} dB rms")
        print(f"  per-profile residual (amp-specific)       : {res_rms:6.2f} dB rms")
        print(f"  systematic share of total variance        : "
              f"{sys_rms**2 / (sys_rms**2 + res_rms**2):6.1%}")
        print("\n  A large systematic share means a domain shift that indexing can")
        print("  absorb. A large residual means genuinely destroyed detail.")

        worst = np.argsort(-np.abs(systematic))[:6]
        print("\n  biggest systematic shifts:")
        for i in worst:
            print(f"    {centres[i]:8.0f} Hz  {systematic[i]:+6.2f} dB")

        # ---- does domain-matched indexing help? ------------------------
        def retrieval(index_fps, query_fps):
            ish = np.array([f.shape_vector for f in index_fps])
            isc = np.array([f.scalar_vector for f in index_fps])
            ref = (zscore_ref(ish), zscore_ref(isc))
            idx = vectors(ish, isc, ref)
            q = vectors(np.array([f.shape_vector for f in query_fps]),
                        np.array([f.scalar_vector for f in query_fps]), ref)
            ranks, t1, t5 = self_retrieval(q, idx)
            return t1, t5, float(np.median(ranks))

        print("\n" + "=" * 66)
        print("RETRIEVAL: query = separated stem over bed B")
        print("=" * 66)
        print(f"{'index built from':<44}{'top-1':>7}{'top-5':>7}{'med':>6}")
        print("-" * 66)
        for label, idx_fps, q_fps in (
            ("clean renders (current design)", fp_clean, fp_b),
            ("mixed, unseparated (control)", fp_clean, fp_mix_b),
            ("separated stems over bed A (domain-matched)", fp_a, fp_b),
        ):
            t1, t5, med = retrieval(idx_fps, q_fps)
            print(f"{label:<44}{t1:>7.3f}{t5:>7.3f}{med:>6.1f}")

        print("\nIf domain-matched indexing lifts top-1 substantially, separation")
        print("damage is a shift to be absorbed, not information destroyed, and the")
        print("index should be built through the same separator the product uses.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
