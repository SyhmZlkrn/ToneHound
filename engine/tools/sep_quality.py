"""How well does htdemucs_6s actually recover the guitar -- measured directly.

Every separation number so far has been read off retrieval top-1 over 34
profiles, where one profile is worth 0.029. The reported flip from "separation
hurts" to "separation helps" is 2 clips versus 5. That is not enough resolution
to decide whether the synthetic probe was out of the separator's training
distribution, so this tool stops asking the fingerprint and asks the separator:
given the guitar signal that went *into* the mix, how much of it comes back out
of the guitar stem?

That yields one continuous number per profile instead of one hit-or-miss, so 34
profiles become 34 measurements rather than 34 coin flips, and probes can be
compared with a rank test.

The oracle target is the guitar as it entered the mix -- post mastering EQ, post
level scaling, pre bed. Limiting and mp3 damage therefore count against the
separator, but identically for every probe, so the cross-probe comparison stays
fair.

    PATH=.cache/bin:$PATH PYTHONPATH=engine python engine/tools/sep_quality.py \
        --cache .cache/renders_djent --di "assets/user_di/Djent DI.wav"
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import probe_source
from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.features import band_centers, third_octave_edges

from stem_test import add_backing, codec_roundtrip, limit, load_beds, mastering_eq
from stem_test_uvr import GUITAR_MODEL, separate

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_LAG = 4096          # htdemucs is nominally sample-aligned; verify, don't assume


def align(est: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, int]:
    """Shift `est` onto `ref` by the lag maximising cross-correlation."""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    size = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(est, size) * np.conj(np.fft.rfft(ref, size)), size)
    lags = np.concatenate([np.arange(MAX_LAG + 1), np.arange(-MAX_LAG, 0)])
    best = lags[np.argmax(np.abs(xc[lags]))]
    return (np.roll(est, -int(best)), int(best))


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR (dB). Gain-blind, so limiter make-up gain is free."""
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = ref * (est @ ref) / (ref @ ref + 1e-20)
    noise = est - proj
    return float(10 * np.log10((proj @ proj + 1e-20) / (noise @ noise + 1e-20)))


def band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    sel = (freqs >= lo) & (freqs < hi)
    return float(spec[sel].sum())


def ltas(x: np.ndarray, sr: int, cfg) -> np.ndarray:
    """Third-octave band levels in dB, mean-removed (level-independent)."""
    edges = third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (freqs >= edges[i]) & (freqs < edges[i + 1])
        out[i] = 10 * np.log10(spec[sel].sum() + 1e-20) if sel.any() else -200.0
    return out - out.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=ROOT / ".cache" / "renders_v2")
    ap.add_argument("--limit", type=int, default=34)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--guitar-db", type=float, default=-6.0)
    ap.add_argument("--npz", type=pathlib.Path, default=None,
                    help="save per-profile metrics for cross-probe comparison")
    probe_source.add_probe_args(ap)
    args = ap.parse_args()

    import soundfile as sf

    cfg, sr = GUITAR, SAMPLE_RATE
    src = probe_source.probe_from_args(args, sr)
    print(src.banner())
    probe_source.check_cache(args.cache, src, write=False)

    paths = sorted(args.cache.glob("*.npy"))[: args.limit]
    loud = src.slices[-1]
    n = min(int(args.seconds * sr), loud.stop - loud.start)
    clean = [np.load(p)[loud.start:loud.start + n] for p in paths]
    print(f"{len(clean)} profiles, {n / sr:.1f}s from the loudest pass\n")

    # Where the probe itself puts its energy -- the quantity the discovery is about.
    dry = src.audio[loud]
    tot = band_energy(dry, sr, 20, sr / 2)
    print(f"probe energy >5 kHz : {band_energy(dry, sr, 5000, sr / 2) / tot:.6f}")
    print(f"probe energy >10 kHz: {band_energy(dry, sr, 10000, sr / 2) / tot:.6f}\n")

    subtypes = sf.available_subtypes("MP3")
    subtype = "MPEG_LAYER_III" if "MPEG_LAYER_III" in subtypes else None
    bed = load_beds(3, n, sr, np.random.default_rng(0))     # bed A, as stem_domain

    from audio_separator.separator import Separator
    workdir = pathlib.Path(tempfile.mkdtemp(prefix="tonehound_sq_"))
    try:
        sep = Separator(log_level=40, output_dir=str(workdir),
                        model_file_dir=str(ROOT / ".cache" / "uvr_models"),
                        use_soundfile=True)
        sep.load_model(model_filename=GUITAR_MODEL)

        centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
        band = (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)

        cols = {k: [] for k in ("sdr", "lsd", "lag", "himix", "hiest", "sdrmix",
                                "level", "inband")}
        for i, y in enumerate(clean):
            eq = mastering_eq(y, sr, np.random.default_rng(1000 + i))   # stem_domain seeds
            # add_backing's guitar term, i.e. the guitar as it entered the mix.
            target = eq / (np.abs(eq).max() + 1e-12) * 10 ** (args.guitar_db / 20)
            mix = codec_roundtrip(limit(add_backing(eq, bed, args.guitar_db)), sr, subtype)

            stem = separate(mix, sr, sep, workdir, f"Q{i:03d}")
            if stem is None:
                print(f"  [{i + 1}/{len(clean)}] NO STEM")
                continue
            est, lag = align(stem[:n], target)
            mix_al, _ = align(mix[:n], target)

            cols["sdr"].append(si_sdr(est, target))
            cols["sdrmix"].append(si_sdr(mix_al, target))
            cols["lsd"].append(float(np.sqrt(
                ((ltas(est, sr, cfg) - ltas(target, sr, cfg))[band] ** 2).mean())))
            cols["lag"].append(lag)
            cols["himix"].append(band_energy(mix, sr, 10000, sr / 2)
                                 / (band_energy(mix, sr, 20, sr / 2) + 1e-30))
            cols["hiest"].append(band_energy(est, sr, 10000, sr / 2)
                                 / (band_energy(est, sr, 20, sr / 2) + 1e-30))
            # Is the stem actually carrying signal, or is it near-silence?
            cols["level"].append(20 * np.log10(
                (np.sqrt((est ** 2).mean()) + 1e-20) / (np.sqrt((mix[:n] ** 2).mean()) + 1e-20)))
            cols["inband"].append(band_energy(est, sr, 80, 10000)
                                  / (band_energy(est, sr, 20, sr / 2) + 1e-30))
            print(f"  [{i + 1}/{len(clean)}] SI-SDR {cols['sdr'][-1]:+6.2f} dB   "
                  f"LSD {cols['lsd'][-1]:5.2f} dB   lag {lag:+5d}", flush=True)

        sdr, lsd = np.array(cols["sdr"]), np.array(cols["lsd"])
        sdrmix, lags = np.array(cols["sdrmix"]), np.array(cols["lag"])
        himix, hiest = np.array(cols["himix"]), np.array(cols["hiest"])
        level, inband = np.array(cols["level"]), np.array(cols["inband"])

        print("\n" + "=" * 66)
        print("SEPARATION FIDELITY vs the guitar that entered the mix")
        print("=" * 66)
        for name, v in (("SI-SDR, separated stem (dB)", sdr),
                        ("SI-SDR, unseparated mix   (dB)", sdrmix),
                        ("LTAS error, analysis band (dB rms)", lsd)):
            print(f"  {name:<36} median {np.median(v):+6.2f}  "
                  f"mean {v.mean():+6.2f}  IQR [{np.percentile(v, 25):+6.2f},"
                  f"{np.percentile(v, 75):+6.2f}]")
        gain = sdr - sdrmix
        print(f"\n  separation gain over doing nothing   median {np.median(gain):+6.2f} dB, "
              f"positive in {int((gain > 0).sum())}/{len(gain)} profiles")
        print(f"  alignment lag: median {int(np.median(lags))}, "
              f"max |lag| {int(np.abs(lags).max())} samples")
        print(f"\n  >10 kHz energy fraction: mix {np.median(himix):.6f} -> "
              f"stem {np.median(hiest):.6f}  "
              f"({10 * np.log10((np.median(hiest) + 1e-30) / (np.median(himix) + 1e-30)):+.1f} dB)")
        print(f"  stem level rel. mix    : median {np.median(level):+6.2f} dB "
              f"(range {level.min():+.2f} .. {level.max():+.2f})")
        print(f"  stem energy in 80 Hz-10 kHz analysis band: median {np.median(inband):.4f}")

        if args.npz:
            np.savez(args.npz, sdr=sdr, lsd=lsd, sdrmix=sdrmix, gain=gain,
                     himix=himix, hiest=hiest, level=level, inband=inband,
                     label=src.label)
            print(f"\n  saved {args.npz}")
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
