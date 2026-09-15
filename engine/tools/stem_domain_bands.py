"""Is the domain-matched retrieval win real amp character, or the >10 kHz artefact?

``stem_domain_test`` showed that indexing through the same separator lifts real-DI
top-1 from 0.206 to 0.588. That number is only trustworthy if it survives the
band that is under suspicion. Two things make the four LTAS bands above 10 kHz
suspect on a real DI:

  * a real guitar DI sits 50 dB down up there, so those bands read whatever the
    NAM model, the mp3 encoder and the separator leave behind, not the amp;
  * they lie *outside* ``analysis_hi_hz`` (10 kHz), so the systematic/residual
    decomposition never sees them even though retrieval scores them.

So this repeats the domain experiment with every retrieval also run with those
bands dropped and with those bands alone, and it prints the absolute energy the
suspect region actually carries at each stage of the chain. Fingerprints are
dumped to an ``.npz`` so further ablations cost nothing.

    PATH=.cache/bin:$PATH PYTHONPATH=engine python engine/tools/stem_domain_bands.py \
        --cache .cache/renders_djent --di "assets/user_di/Djent DI.wav" \
        --dump .cache/fp_djent.npz
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


def build(clean, bed, sr, separator, workdir, tag, guitar_db, subtype):
    """Mix -> master -> codec -> separate, returning (mixes, stems, guitar refs).

    Seeds match stem_domain_test exactly so the two tools are comparable. The
    third return is the guitar term as it actually entered the mix sum, which is
    the only honest reference for a separation-quality score.
    """
    mixes, stems, refs = [], [], []
    for i, y in enumerate(clean):
        r = np.random.default_rng(1000 + i)
        eq = mastering_eq(y, sr, r)
        refs.append(eq / (np.abs(eq).max() + 1e-12) * 10 ** (guitar_db / 20.0))
        mix = codec_roundtrip(limit(add_backing(eq, bed, guitar_db)), sr, subtype)
        stem = separate(mix, sr, separator, workdir, f"{tag}{i:03d}")
        mixes.append(mix)
        stems.append(mix if stem is None else stem[: len(y)])
        print(f"  [{tag}] {i + 1}/{len(clean)}", flush=True)
    return mixes, stems, refs


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB -- the standard source-separation score.

    Scale invariance matters here because the separator, the limiter and the
    peak-normalisation in `separate` all rescale arbitrarily; only the shape of
    the error relative to the target is meaningful.
    """
    n = min(len(est), len(ref))
    est, ref = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    alpha = float(est @ ref) / float(ref @ ref + 1e-30)
    target = alpha * ref
    err = est - target
    return 10.0 * np.log10(float(target @ target) / float(err @ err + 1e-30) + 1e-30)


def hf_fraction(x: np.ndarray, sr: int, cut: float = 10_000.0) -> float:
    """Fraction of total energy above `cut` -- the direct measure of whether the
    suspect bands carry signal at all."""
    X = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = X.sum()
    return float(X[f >= cut].sum() / (tot + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_v2")
    ap.add_argument("--limit", type=int, default=34)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--guitar-db", type=float, default=-6.0)
    ap.add_argument("--dump", type=pathlib.Path, default=None)
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

    bed_a = load_beds(3, n, sr, np.random.default_rng(0))
    bed_b = load_beds(3, n, sr, np.random.default_rng(7))
    print(f"beds differ: {not np.allclose(bed_a, bed_b)}\n")

    from audio_separator.separator import Separator
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_bands_"))
    try:
        sep = Separator(log_level=40, output_dir=str(workdir),
                        model_file_dir=str(ROOT / ".cache" / "uvr_models"),
                        use_soundfile=True)
        sep.load_model(model_filename=GUITAR_MODEL)

        _, stems_a, _ = build(clean, bed_a, sr, sep, workdir, "A", args.guitar_db, subtype)
        mixes_b, stems_b, refs_b = build(
            clean, bed_b, sr, sep, workdir, "B", args.guitar_db, subtype)

        # ---- separation quality, independent of the fingerprint --------
        sdr_mix = np.array([si_sdr(m, r) for m, r in zip(mixes_b, refs_b)])
        sdr_sep = np.array([si_sdr(s, r) for s, r in zip(stems_b, refs_b)])
        print("\n" + "=" * 70)
        print("SEPARATION QUALITY (SI-SDR vs the guitar term that entered the mix)")
        print("=" * 70)
        print(f"  unseparated mix   {sdr_mix.mean():+7.2f} dB mean   "
              f"{np.median(sdr_mix):+7.2f} median   "
              f"[{sdr_mix.min():+.2f}, {sdr_mix.max():+.2f}]")
        print(f"  separated stem    {sdr_sep.mean():+7.2f} dB mean   "
              f"{np.median(sdr_sep):+7.2f} median   "
              f"[{sdr_sep.min():+.2f}, {sdr_sep.max():+.2f}]")
        gain = sdr_sep - sdr_mix
        print(f"  improvement       {gain.mean():+7.2f} dB mean   "
              f"{np.median(gain):+7.2f} median   "
              f"({(gain > 0).mean():.0%} of profiles improved)")

        fp_clean = [fingerprint(y, sr, cfg) for y in clean]
        fp_a = [fingerprint(y, sr, cfg) for y in stems_a]
        fp_b = [fingerprint(y, sr, cfg) for y in stems_b]
        fp_mix_b = [fingerprint(y, sr, cfg) for y in mixes_b]

        centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
        hi = centres >= 10_000.0

        if args.dump:
            np.savez(args.dump,
                     centres=centres,
                     clean_ltas=np.array([f.ltas for f in fp_clean]),
                     a_ltas=np.array([f.ltas for f in fp_a]),
                     b_ltas=np.array([f.ltas for f in fp_b]),
                     mixb_ltas=np.array([f.ltas for f in fp_mix_b]),
                     sdr_mix=sdr_mix, sdr_sep=sdr_sep,
                     clean_scal=np.array([f.scalar_vector for f in fp_clean]),
                     a_scal=np.array([f.scalar_vector for f in fp_a]),
                     b_scal=np.array([f.scalar_vector for f in fp_b]),
                     mixb_scal=np.array([f.scalar_vector for f in fp_mix_b]))
            print(f"fingerprints dumped to {args.dump}\n")

        # ---- does the suspect band carry any signal? -------------------
        print("=" * 70)
        print(f"ENERGY ABOVE 10 kHz, fraction of total (median over {len(clean)} profiles)")
        print("=" * 70)
        for label, sigs in (("clean render", clean), ("mixed+mastered+mp3", mixes_b),
                            ("separated guitar stem", stems_b)):
            fr = np.median([hf_fraction(y, sr) for y in sigs])
            print(f"  {label:<26}{fr:12.3e}   ({10*np.log10(fr+1e-30):+7.1f} dB)")

        print("\n  mean LTAS level of the four >10 kHz bands, dB rel. band mean:")
        for label, fps in (("clean render", fp_clean), ("mixed+mp3", fp_mix_b),
                           ("separated stem", fp_b)):
            v = np.array([f.shape_vector for f in fps])
            print(f"  {label:<26}{v[:, hi].mean():+8.2f} dB   "
                  f"(spread across profiles {v[:, hi].std():5.2f} dB)")

        # ---- damage decomposition, both bands --------------------------
        print("\n" + "=" * 70)
        print("SEPARATION DAMAGE, decomposed")
        print("=" * 70)
        d = np.array([b.ltas - c.ltas for b, c in zip(fp_b, fp_clean)])
        d = d - d.mean(axis=1, keepdims=True)
        systematic = d.mean(axis=0)
        residual = d - systematic

        for name, mask in (
                ("analysis band 80 Hz-10 kHz (as reported)",
                 (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)),
                ("full LTAS 40 Hz-16 kHz (what retrieval scores)",
                 np.ones_like(centres, dtype=bool)),
                (">10 kHz only (the suspect block)", hi)):
            s = float(np.sqrt((systematic[mask] ** 2).mean()))
            r = float(np.sqrt((residual[:, mask] ** 2).mean()))
            print(f"  {name}")
            print(f"    systematic {s:6.2f} dB rms   residual {r:6.2f} dB rms   "
                  f"systematic share {s**2/(s**2+r**2):6.1%}")

        worst = np.argsort(-np.abs(systematic))[:6]
        print("\n  biggest systematic shifts (all 48 bands):")
        for i in worst:
            print(f"    {centres[i]:8.0f} Hz  {systematic[i]:+6.2f} dB")

        # ---- retrieval, every index x every band subset ----------------
        def retrieval(index_fps, query_fps, mask):
            ish = np.array([f.shape_vector for f in index_fps])[:, mask]
            isc = np.array([f.scalar_vector for f in index_fps])
            ref = (zscore_ref(ish), zscore_ref(isc))
            idx = vectors(ish, isc, ref)
            q = vectors(np.array([f.shape_vector for f in query_fps])[:, mask],
                        np.array([f.scalar_vector for f in query_fps]), ref)
            ranks, t1, t5 = self_retrieval(q, idx)
            return t1, t5, float(np.median(ranks))

        allb = np.ones_like(centres, dtype=bool)
        print("\n" + "=" * 70)
        print("RETRIEVAL: query = separated stem over bed B (except control row)")
        print("=" * 70)
        print(f"{'index / query':<40}{'bands':<18}{'top-1':>7}{'top-5':>7}{'med':>6}")
        print("-" * 78)
        for label, idx_fps, q_fps in (
            ("index clean / query separated", fp_clean, fp_b),
            ("index clean / query UNseparated", fp_clean, fp_mix_b),
            ("index separated-A / query separated-B", fp_a, fp_b),
        ):
            for bname, mask in (("all 48", allb), (">10 kHz dropped", ~hi),
                                (">10 kHz only", hi)):
                t1, t5, med = retrieval(idx_fps, q_fps, mask)
                print(f"{label:<40}{bname:<18}{t1:>7.3f}{t5:>7.3f}{med:>6.1f}")
            print()

        print("If the domain-matched row holds up with >10 kHz dropped, the win is")
        print("real amp character. If it collapses, it was riding the same artefact.")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
