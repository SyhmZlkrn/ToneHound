"""Cross-probe comparison of separation behaviour, from the dumped fingerprints.

``stem_domain_bands`` writes one ``.npz`` per probe. Reading them back is free,
so every question that is a function of the fingerprints rather than the audio
gets answered here instead of by another hour of separation:

  * how the separator's systematic spectral shift differs between probes -- the
    mechanism behind the sign flip, not just its sign;
  * whether the domain-matched retrieval win survives dropping the >10 kHz
    bands, per probe;
  * whether per-profile separation quality (SI-SDR) predicts retrieval rank,
    which is what decides if better separation would actually buy accuracy.

    PYTHONPATH=engine python engine/tools/compare_probes.py \
        .cache/fp_synth.npz .cache/fp_djent.npz .cache/fp_funk.npz
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from stem_test import self_retrieval, vectors, zscore_ref


def shape(ltas: np.ndarray) -> np.ndarray:
    return ltas - ltas.mean(axis=1, keepdims=True)


def retrieval(idx_sh, idx_sc, q_sh, q_sc, mask):
    ref = (zscore_ref(idx_sh[:, mask]), zscore_ref(idx_sc))
    idx = vectors(idx_sh[:, mask], idx_sc, ref)
    q = vectors(q_sh[:, mask], q_sc, ref)
    return self_retrieval(q, idx)


def main() -> int:
    files = [pathlib.Path(p) for p in sys.argv[1:]]
    if not files:
        print("usage: compare_probes.py <npz> [<npz> ...]")
        return 2
    data = {f.stem.replace("fp_", ""): np.load(f) for f in files if f.exists()}
    missing = [f for f in files if not f.exists()]
    for f in missing:
        print(f"MISSING: {f}")
    if not data:
        return 1

    centres = next(iter(data.values()))["centres"]
    hi = centres >= 10_000.0
    allb = np.ones_like(centres, dtype=bool)

    print("=" * 78)
    print("SEPARATOR'S SYSTEMATIC SPECTRAL SHIFT (separated - clean, level-removed)")
    print("=" * 78)
    print(f"{'band':>9}" + "".join(f"{k:>14}" for k in data))
    sysmap = {}
    for k, d in data.items():
        delta = shape(d["b_ltas"]) - shape(d["clean_ltas"])
        sysmap[k] = delta.mean(axis=0)
    for i in range(0, len(centres), 3):
        print(f"{centres[i]:9.0f}" + "".join(f"{sysmap[k][i]:+14.2f}" for k in data))

    print("\n  rms of that shift, by region:")
    for name, m in (("40 Hz - 300 Hz", centres < 300),
                    ("300 Hz - 2.5 kHz", (centres >= 300) & (centres < 2500)),
                    ("2.5 - 10 kHz", (centres >= 2500) & (centres < 10000)),
                    (">10 kHz", hi)):
        print(f"  {name:<20}" + "".join(
            f"{np.sqrt((sysmap[k][m]**2).mean()):>13.2f} dB" for k in data))

    print("\n" + "=" * 78)
    print("SEPARATION QUALITY, SI-SDR vs the guitar term in the mix")
    print("=" * 78)
    print(f"{'probe':<12}{'mix':>10}{'separated':>12}{'gain':>9}{'% improved':>12}")
    for k, d in data.items():
        if "sdr_mix" not in d:
            continue
        m, s = d["sdr_mix"], d["sdr_sep"]
        print(f"{k:<12}{m.mean():>+10.2f}{s.mean():>+12.2f}{(s-m).mean():>+9.2f}"
              f"{(s > m).mean():>11.0%}")

    print("\n" + "=" * 78)
    print("DOES THE DOMAIN-MATCHED WIN SURVIVE DROPPING >10 kHz?")
    print("=" * 78)
    print(f"{'probe':<10}{'index / query':<34}{'all 48':>16}{'>10k dropped':>16}")
    for k, d in data.items():
        csh, csc = shape(d["clean_ltas"]), d["clean_scal"]
        ash, asc = shape(d["a_ltas"]), d["a_scal"]
        bsh, bsc = shape(d["b_ltas"]), d["b_scal"]
        msh, msc = shape(d["mixb_ltas"]), d["mixb_scal"]
        for label, isf, isc_, qsh, qsc in (
                ("clean / separated", csh, csc, bsh, bsc),
                ("clean / UNseparated", csh, csc, msh, msc),
                ("separated-A / separated-B", ash, asc, bsh, bsc)):
            a = retrieval(isf, isc_, qsh, qsc, allb)
            b = retrieval(isf, isc_, qsh, qsc, ~hi)
            print(f"{k:<10}{label:<34}{a[1]:>16.3f}{b[1]:>16.3f}")
        print()

    print("=" * 78)
    print("DOES BETTER SEPARATION MEAN BETTER RETRIEVAL? (per-profile)")
    print("=" * 78)
    for k, d in data.items():
        if "sdr_sep" not in d:
            continue
        ash, asc = shape(d["a_ltas"]), d["a_scal"]
        bsh, bsc = shape(d["b_ltas"]), d["b_scal"]
        ranks, _, _ = retrieval(ash, asc, bsh, bsc, allb)
        sdr = d["sdr_sep"]
        r = float(np.corrcoef(sdr, -ranks.astype(float))[0, 1])
        line = f"  {k:<10} corr(SI-SDR, -rank) = {r:+.3f}"
        hit, miss = ranks == 1, ranks > 1
        if hit.any() and miss.any():
            line += (f"   SI-SDR of rank-1 hits {sdr[hit].mean():+6.2f} dB"
                     f" vs misses {sdr[miss].mean():+6.2f} dB")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
