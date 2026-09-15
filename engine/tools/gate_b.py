"""Phase 0 Gate B: does the fingerprint feature space separate real amp captures?

Gate A established that the loader renders .nam files correctly. That says
nothing about whether the *fingerprint* built from those renders is a useful
coordinate system. Everything downstream -- the profile index, the joint
(profile, IR) search, the match-EQ -- assumes that distance in this space
tracks perceived amp identity. If it does not, the matcher ranks noise, and no
amount of work on the later stages recovers it. So the assumption is measured
here, on real captures, before anything is built on top of it.

The corpus is its own ground truth. ``assets/dev_profiles`` contains two large
near-duplicate families -- ten Ceriatone King Kong captures (channels, decades,
boost pedals) and eight Suhr Reactive Load captures (knob positions) -- plus
smaller same-amp groups. Captures of one amp are *not* interchangeable, but
they are unambiguously more alike than captures of two different amps. That
gives a labelled ordering the feature space must reproduce.

The verdict is therefore a two-sided one, and both sides matter:

  * captures of the same amp must land closer together than captures of
    different amps (measured as AUC, d-prime and distribution overlap, against
    a label-permutation null so that group sizes cannot manufacture the
    result), and
  * different amps must actually be resolved -- a space that collapses
    everything to one point also satisfies "within < across" vacuously, so
    spread and rank-retrieval are checked separately.

Reported alongside the verdict: which of the 60 features carry the separation,
which are dead weight or actively harmful, and which pathologies (constants,
deterministic duplicates, scale domination) distort the distance.

Run::

    PYTHONPATH=engine python engine/tools/gate_b.py
    PYTHONPATH=engine python engine/tools/gate_b.py --cache <dir>   # reuse renders
    PYTHONPATH=engine python engine/tools/gate_b.py --di "assets/user_di/Djent DI.wav" \
        --cache <other dir>                                        # real DI probe
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_source
from tonehound import nam_render
from tonehound.config import GUITAR, SAMPLE_RATE, InstrumentConfig
from tonehound.features import (SCALAR_KEYS, Fingerprint, band_centers, fingerprint, ltas_db,
                             third_octave_edges)

PROFILE_DIR = Path(__file__).resolve().parents[2] / "assets" / "dev_profiles"

# Share of the whitened vector's squared length given to the LTAS block. The
# spectrum is what the match-EQ ultimately has to reproduce, and it is the only
# term the additive (profile, IR) search operates on, so it leads; the scalars
# are a saturation/dynamics correction on top.
LTAS_WEIGHT = 0.5

# Share given to the per-level delta block (how brightness moves as the amp is
# driven harder). Cab-invariant by construction, so it survives the joint
# (profile, IR) search untouched.
DELTA_WEIGHT = 0.25

# Families the task names explicitly: same amp, different channel/mode/pedal or
# different knob positions. Matched as substrings of the filename.
DECLARED_FAMILIES = {
    "ceriatone_king_kong": "Ceriatone King Kong",
    "suhr_rl": "[Suhr RL]",
}

# Every same-amp group the filenames support, not just the two large ones. This
# is the stricter labelling: under DECLARED_FAMILIES the six small groups get
# scored as "different amps", which puts genuinely-close pairs into the
# across-family distribution and understates separation.
EXTENDED_FAMILIES = {
    **DECLARED_FAMILIES,
    "peavey_5150_blockletter": "5150 BlockLetter",
    "peavey_6505plus": "6505+",
    "peavey_6534plus": "6534+",
    "fender_twinverb": "Fender TwinVerb",
    "magnatone_super59": "Magnatone Super 59",
    "splawn_promod": "Splawn Pro Mod",
}

# POST-HOC labelling, defined AFTER the first gate run. Reported for diagnosis
# only and deliberately NOT part of the pass/fail criteria, because redefining
# the ground truth once the numbers are known is how a failing gate gets talked
# into passing.
#
# Both corrections rest on facts independent of the distance matrix:
#  - The five "King Kong NAM Capture <pedal> Chase Tone Secret Pre Deco" files
#    are captures of PEDALS through the amp; the "Ch"/"chan" files are the amp's
#    own channels. The filenames say so. Scoring them as one family asserts a
#    pedal capture should sit close to a channel capture, which is false.
#  - Peavey renamed the 5150 to the 6505 when the Van Halen licence lapsed. Same
#    circuit, different badge, so "5150 BlockLetter vs 6505+" was never a
#    different-amp pair.
# The specific King Kong rule is listed first; family_of takes the first match.
CORRECTED_FAMILIES = {
    "ceriatone_kk_pedal": "King Kong NAM Capture",
    "ceriatone_kk_amp": "King Kong",
    "suhr_rl": "[Suhr RL]",
    "peavey_5150_6505": ("5150 BlockLetter", "6505+"),
    "peavey_6534plus": "6534+",
    "fender_twinverb": "Fender TwinVerb",
    "magnatone_super59": "Magnatone Super 59",
    "splawn_promod": "Splawn Pro Mod",
}

# Gate thresholds, fixed before the numbers were seen. AUC is the operationally
# meaningful one: it is the probability that a same-amp pair outranks a
# different-amp pair, which is exactly what a ranked matcher does.
MIN_AUC = 0.90
MIN_DPRIME = 1.5
MIN_NN_TOP1 = 0.80
MAX_NULL_P = 0.01
# A collapsed space would pass "within < across" trivially, so the different-amp
# distribution must also have real spread: its 1st percentile has to be a
# non-trivial fraction of its median. Measured on the extended labelling -- under
# the declared one the closest "different amp" pair is the two Fender TwinVerb
# captures, which are the same amp, so the ratio would report a label error as a
# collapsed feature space.
MIN_SPREAD_RATIO = 0.10


def profile_paths(directory: Path) -> list[Path]:
    """Real captures only -- ``_``-prefixed files are round-trip test fixtures."""
    return sorted(p for p in directory.glob("*.nam") if not p.name.startswith("_"))


def family_of(name: str, families: "dict[str, str | tuple[str, ...]]") -> str | None:
    """First matching rule wins, so order specific rules before general ones."""
    for key, markers in families.items():
        if isinstance(markers, str):
            markers = (markers,)
        if any(m in name for m in markers):
            return key
    return None


# --------------------------------------------------------------------------
# Rendering and fingerprinting
# --------------------------------------------------------------------------


@dataclass
class Rendered:
    name: str
    audio: np.ndarray
    receptive_field: int
    weight_count: int
    render_seconds: float


def _usable(cached: Path, expected: int) -> bool:
    """Is this cached render complete?

    A full disk truncates ``np.save`` mid-write and leaves a short .npy that
    loads without error, which is the same silent-wrong-numbers failure the
    probe stamp exists to prevent, arriving through a different door. A render
    always has exactly as many samples as the probe, so a length mismatch (or a
    header too damaged to read) means re-render rather than trust it.
    """
    try:
        with open(cached, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, _, dtype = np.lib.format._read_array_header(fh, version)
    except Exception:
        return False
    if shape != (expected,):
        print(f"  cache: discarding truncated render {cached.name} "
              f"({shape} != ({expected},))")
        return False
    return cached.stat().st_size >= expected * dtype.itemsize


def render_corpus(paths: list[Path], probe: np.ndarray,
                  cache: Path | None) -> tuple[list[Rendered], list[tuple[str, str]]]:
    """Render the probe through every profile. Returns (rendered, failures)."""
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    out: list[Rendered] = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        cached = None if cache is None else cache / (path.stem + ".npy")
        try:
            model = nam_render.load(path)
            if cached is not None and cached.exists() and _usable(cached, len(probe)):
                y, seconds = np.load(cached), 0.0
            else:
                t0 = time.time()
                y = model.render(probe)
                seconds = time.time() - t0
                if cached is not None:
                    np.save(cached, y)
            out.append(Rendered(path.stem, y, model.receptive_field,
                                model.weight_count, seconds))
        except Exception as exc:  # the indexer's skip-and-report path
            failures.append((path.stem, f"{type(exc).__name__}: {exc}"))
    return out, failures


def build_features(rendered: list[Rendered], cfg: InstrumentConfig, sr: int,
                   levels: list[slice]) -> tuple[list[Fingerprint], np.ndarray, np.ndarray, np.ndarray]:
    """Fingerprint each render and stack into (n_profiles, n_features) blocks.

    ``levels`` must be the slices of the probe that produced ``rendered`` -- the
    synthetic phrase and a real DI have different lengths, so a mismatch silently
    reads the wrong drive level.
    """
    fps = [fingerprint(r.audio, sr, cfg, levels=levels) for r in rendered]
    shape = np.array([f.shape_vector for f in fps])
    scal = np.array([f.scalar_vector for f in fps])
    # Loudest pass minus quietest: the level dependence the three-level probe
    # exists to capture, and which a single Welch average folds away.
    delta = np.array([f.level_delta for f in fps])
    return fps, shape, scal, delta


# --------------------------------------------------------------------------
# The distance
# --------------------------------------------------------------------------


def zscore(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension whitening across the corpus. Returns (z, std).

    A dimension with no variance carries no information and is zeroed rather
    than divided by ~0; ``std`` is returned so the caller can report it.
    """
    std = block.std(axis=0)
    safe = np.where(std > 1e-12, std, 1.0)
    z = (block - block.mean(axis=0)) / safe
    z[:, std <= 1e-12] = 0.0
    return z, std


def whitened_vectors(shape: np.ndarray, scal: np.ndarray,
                     ltas_weight: float = LTAS_WEIGHT,
                     delta: np.ndarray | None = None,
                     delta_weight: float = DELTA_WEIGHT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whitened, block-weighted feature vectors. Returns (v, z_shape, z_scal).

    Each block is scaled so that its expected squared length across the corpus
    is its weight, which is what makes "LTAS above scalars" a stated ratio
    rather than an accident of how many bands there happen to be.
    """
    z_shape, _ = zscore(shape)
    z_scal, _ = zscore(scal)
    if delta is None or delta.size == 0:
        a = z_shape * np.sqrt(ltas_weight / z_shape.shape[1])
        b = z_scal * np.sqrt((1.0 - ltas_weight) / z_scal.shape[1])
        return np.hstack([a, b]), z_shape, z_scal

    z_delta, _ = zscore(delta)
    scal_weight = 1.0 - ltas_weight - delta_weight
    a = z_shape * np.sqrt(ltas_weight / z_shape.shape[1])
    b = z_scal * np.sqrt(scal_weight / z_scal.shape[1])
    c = z_delta * np.sqrt(delta_weight / z_delta.shape[1])
    return np.hstack([a, b, c]), z_shape, z_scal


def cosine_distance_matrix(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    unit = v / np.where(norm > 0, norm, 1.0)
    return np.clip(1.0 - unit @ unit.T, 0.0, 2.0)


def euclidean_distance_matrix(v: np.ndarray) -> np.ndarray:
    d = v[:, None, :] - v[None, :, :]
    return np.sqrt((d**2).sum(axis=2))


# --------------------------------------------------------------------------
# Separation statistics
# --------------------------------------------------------------------------


def split_pairs(dist: np.ndarray, labels: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    """Upper-triangle distances split into (same-family, different-family).

    A profile with no family is "different" from everything, which is the
    conservative reading: an unlabelled same-amp pair lands in the across
    distribution and drags the measured separation down, never up.
    """
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same = np.array([labels[i] is not None and labels[i] == labels[j]
                     for i, j in zip(*iu)])
    d = dist[iu]
    return d[same], d[~same]


def auc_brute(within: np.ndarray, across: np.ndarray) -> float:
    """P(within < across) + 0.5 P(tie), by direct enumeration."""
    cmp = within[:, None] - across[None, :]
    return float((cmp < 0).mean() + 0.5 * (cmp == 0).mean())


def auc_ranked(within: np.ndarray, across: np.ndarray) -> float:
    """Same quantity via rank sums -- fast enough for permutation nulls."""
    from scipy.stats import rankdata
    n1, n2 = len(within), len(across)
    r = rankdata(np.concatenate([within, across]))
    u_greater = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return float(1.0 - u_greater / (n1 * n2))


def dprime(within: np.ndarray, across: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (within.var(ddof=1) + across.var(ddof=1)))
    return float((across.mean() - within.mean()) / pooled) if pooled > 0 else float("inf")


def overlap_coefficient(within: np.ndarray, across: np.ndarray, bins: int = 60) -> float:
    """Shared area of the two normalised histograms: 0 = disjoint, 1 = identical."""
    lo = min(within.min(), across.min())
    hi = max(within.max(), across.max())
    edges = np.linspace(lo, hi, bins + 1)
    pw, _ = np.histogram(within, bins=edges, density=True)
    pa, _ = np.histogram(across, bins=edges, density=True)
    return float(np.minimum(pw, pa).sum() * (edges[1] - edges[0]))


def permutation_null(dist: np.ndarray, labels: list[str | None], observed: float,
                     trials: int, seed: int = 20260904) -> tuple[float, float, float]:
    """AUC under random relabelling that preserves the family size profile.

    With 73 same-amp pairs among 561, an AUC above chance has to be shown to be
    more than an artefact of group sizes.
    """
    rng = np.random.default_rng(seed)
    sizes = [labels.count(k) for k in dict.fromkeys(x for x in labels if x is not None)]
    n = len(labels)
    idx = np.arange(n)
    null = np.empty(trials)
    for t in range(trials):
        rng.shuffle(idx)
        fake: list[str | None] = [None] * n
        cursor = 0
        for gi, size in enumerate(sizes):
            for k in idx[cursor:cursor + size]:
                fake[k] = f"g{gi}"
            cursor += size
        w, a = split_pairs(dist, fake)
        null[t] = auc_ranked(w, a)
    p = float((null >= observed).sum() + 1) / (trials + 1)
    return p, float(null.mean()), float(np.percentile(null, 99))


def nearest_neighbour_report(dist: np.ndarray, names: list[str],
                             labels: list[str | None]) -> dict[str, float]:
    """Rank-retrieval, which is what a matcher actually does with this space."""
    n = len(names)
    members = [i for i in range(n)
               if labels[i] is not None and sum(l == labels[i] for l in labels) > 1]
    top1 = 0
    aps: list[float] = []
    recalls: list[float] = []
    for i in members:
        order = [j for j in np.argsort(dist[i]) if j != i]
        hits = np.array([labels[j] == labels[i] for j in order])
        total = hits.sum()
        if hits[0]:
            top1 += 1
        ranks = np.flatnonzero(hits) + 1
        aps.append(float(np.mean(np.arange(1, total + 1) / ranks)))
        recalls.append(float(hits[:total].sum() / total))
    return {
        "n_members": float(len(members)),
        "top1": top1 / len(members),
        "map": float(np.mean(aps)),
        "recall_at_family_size": float(np.mean(recalls)),
    }


# --------------------------------------------------------------------------
# Per-feature analysis
# --------------------------------------------------------------------------


def per_feature_auc(z: np.ndarray, labels: list[str | None]) -> np.ndarray:
    """AUC of each feature used alone, as |difference| between two profiles.

    0.5 means the feature says nothing about amp identity; below 0.5 means it
    varies *more* within an amp than between amps, so it fights the metric.
    """
    out = np.empty(z.shape[1])
    for j in range(z.shape[1]):
        d = np.abs(z[:, j][:, None] - z[:, j][None, :])
        w, a = split_pairs(d, labels)
        out[j] = auc_ranked(w, a)
    return out


def leave_one_out_auc(v: np.ndarray, labels: list[str | None],
                      full_auc: float) -> np.ndarray:
    """Change in overall AUC when each feature is dropped.

    Negative means dropping the feature *improves* separation -- the sharpest
    definition of dead weight, because it accounts for redundancy with the
    features that remain.
    """
    out = np.empty(v.shape[1])
    for j in range(v.shape[1]):
        keep = np.delete(v, j, axis=1)
        w, a = split_pairs(cosine_distance_matrix(keep), labels)
        out[j] = auc_ranked(w, a) - full_auc
    return out


def redundancy_pairs(z: np.ndarray, names: list[str], threshold: float = 0.95):
    c = np.corrcoef(z, rowvar=False)
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        if np.isfinite(c[i, j]) and abs(c[i, j]) >= threshold:
            pairs.append((names[i], names[j], float(c[i, j])))
    return sorted(pairs, key=lambda p: -abs(p[2]))


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe(tag: str, x: np.ndarray) -> str:
    return (f"{tag:<22} n={len(x):<4d} min={x.min():.4f} p05={np.percentile(x, 5):.4f} "
            f"med={np.median(x):.4f} mean={x.mean():.4f} p95={np.percentile(x, 95):.4f} "
            f"max={x.max():.4f}")


def analyse(dist: np.ndarray, labels: list[str | None], names: list[str],
            title: str, trials: int) -> dict[str, float]:
    within, across = split_pairs(dist, labels)
    a_brute, a_rank = auc_brute(within, across), auc_ranked(within, across)
    assert abs(a_brute - a_rank) < 1e-9, (a_brute, a_rank)

    print(f"\n-- {title} --")
    groups = {k: labels.count(k) for k in dict.fromkeys(x for x in labels if x is not None)}
    print(f"families: {groups}  singletons: {labels.count(None)}")
    print(describe("within-family", within))
    print(describe("across-family", across))
    print(f"AUC (P same-amp pair closer)  = {a_brute:.4f}   "
          f"[brute force and rank sum agree to {abs(a_brute - a_rank):.1e}]")
    print(f"d-prime                       = {dprime(within, across):.3f}")
    print(f"histogram overlap coefficient = {overlap_coefficient(within, across):.4f}")
    print(f"cleanly separated (max within < min across) = "
          f"{bool(within.max() < across.min())}   "
          f"max_within={within.max():.4f} min_across={across.min():.4f}")
    print(f"across-family pairs closer than the within-family median: "
          f"{(across < np.median(within)).sum()}/{len(across)} "
          f"({100 * (across < np.median(within)).mean():.1f}%)")
    print(f"within-family pairs further than the across-family median: "
          f"{(within > np.median(across)).sum()}/{len(within)} "
          f"({100 * (within > np.median(across)).mean():.1f}%)")
    print(f"within p95={np.percentile(within, 95):.4f} vs across p05="
          f"{np.percentile(across, 5):.4f}")

    p, null_mean, null_p99 = permutation_null(dist, labels, a_rank, trials)
    print(f"label-permutation null ({trials} trials): mean AUC={null_mean:.4f} "
          f"p99={null_p99:.4f}  ->  p = {p:.5f}")

    nn = nearest_neighbour_report(dist, names, labels)
    print(f"retrieval over {int(nn['n_members'])} family members: "
          f"top-1 same-family={nn['top1']:.3f}  mAP={nn['map']:.3f}  "
          f"recall@(family size-1)={nn['recall_at_family_size']:.3f}")

    return {"auc": a_brute, "dprime": dprime(within, across), "p": p,
            "top1": nn["top1"], "overlap": overlap_coefficient(within, across),
            "across_p01_over_med": float(np.percentile(across, 1) / np.median(across))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", type=Path, default=PROFILE_DIR)
    ap.add_argument("--cache", type=Path, default=None,
                    help="directory for cached renders (renders are deterministic)")
    ap.add_argument("--trials", type=int, default=5000,
                    help="label permutations for the null distribution")
    probe_source.add_probe_args(ap)
    args = ap.parse_args()

    cfg, sr = GUITAR, SAMPLE_RATE
    src = probe_source.probe_from_args(args, sr)
    probe, slices = src.audio, src.slices
    paths = profile_paths(args.profiles)

    rule("ToneHound Phase 0 Gate B -- fingerprint feature space separation")
    print(f"profiles dir : {args.profiles}")
    print(f"profiles     : {len(paths)}")
    print(src.banner())
    print(f"instrument   : {cfg.name}, LTAS {cfg.ltas_lo_hz:.0f}-{cfg.ltas_hi_hz:.0f} Hz "
          f"in {cfg.n_bands} bands, analysis {cfg.analysis_lo_hz:.0f}-{cfg.analysis_hi_hz:.0f} Hz")
    probe_source.check_cache(args.cache, src, write=True)

    rendered, failures = render_corpus(paths, probe, args.cache)
    total_s = sum(r.render_seconds for r in rendered)
    print(f"rendered     : {len(rendered)}/{len(paths)} in {total_s:.1f} s")
    for name, err in failures:
        print(f"  FAILED {name}: {err}")
    if len(rendered) < 2:
        print("\nFAIL: fewer than two profiles rendered; nothing to compare.")
        return 1

    names = [r.name for r in rendered]
    fps, shape, scal, delta = build_features(rendered, cfg, sr, slices)

    rule("1. RENDER AND FINGERPRINT SANITY")
    bad = [r.name for r in rendered if not np.isfinite(r.audio).all()]
    print(f"non-finite renders            : {len(bad)} {bad}")
    nan_shape = int((~np.isfinite(shape)).sum())
    nan_scal = int((~np.isfinite(scal)).sum())
    print(f"non-finite LTAS / scalar cells: {nan_shape} / {nan_scal}")
    print(f"receptive fields              : {sorted(set(r.receptive_field for r in rendered))}")
    print(f"render peak abs               : min={min(np.abs(r.audio).max() for r in rendered):.4f} "
          f"max={max(np.abs(r.audio).max() for r in rendered):.4f}")
    identical = [(names[i], names[j])
                 for i, j in itertools.combinations(range(len(names)), 2)
                 if np.array_equal(rendered[i].audio, rendered[j].audio)]
    print(f"bit-identical render pairs    : {len(identical)} {identical}")

    rule("2. FEATURE PATHOLOGIES")
    centres = band_centers(third_octave_edges(cfg.ltas_lo_hz, cfg.ltas_hi_hz, cfg.n_bands))
    band_names = [f"ltas_{c:.0f}Hz" for c in centres]
    feat_names = band_names + list(SCALAR_KEYS)

    _, shape_std = zscore(shape)
    _, scal_std = zscore(scal)
    dead = [(feat_names[i], float(s)) for i, s in enumerate(shape_std) if s < 1e-6]
    dead += [(SCALAR_KEYS[i], float(s)) for i, s in enumerate(scal_std) if s < 1e-6]
    print(f"constant features (std < 1e-6): {len(dead)} {dead}")

    print("\nLTAS band spread across the corpus (dB std), lowest 6 and highest 6:")
    order = np.argsort(shape_std)
    for i in list(order[:6]) + list(order[-6:]):
        print(f"  {band_names[i]:<16} std={shape_std[i]:6.3f} dB   "
              f"range={shape[:, i].min():8.2f} .. {shape[:, i].max():8.2f}")

    print("\nRaw scalar scales -- what an un-normalised Euclidean distance would weigh:")
    raw_var = scal.var(axis=0)
    share = raw_var / raw_var.sum()
    for i in np.argsort(-share):
        print(f"  {SCALAR_KEYS[i]:<22} mean={scal[:, i].mean():12.4f} "
              f"std={scal_std[i]:10.4f}  raw variance share={100 * share[i]:6.2f}%")
    print(f"  -> top 2 raw features carry {100 * np.sort(share)[-2:].sum():.2f}% of raw "
          f"variance; z-scoring is what stops that domination.")

    # Deltas are computed and reported, but deliberately NOT in the shipping
    # vector: measured on this corpus they cost 0.029 AUC and 0.124 d-prime
    # (0.8052/1.177 without, 0.7764/1.053 with). Kept available because the
    # design intent is sound and a real recorded DI may change the answer.
    v, z_shape, z_scal = whitened_vectors(shape, scal)
    z_all = np.hstack([z_shape, z_scal])
    print("\nRedundant feature pairs (|r| >= 0.95 across the corpus):")
    for a, b, r in redundancy_pairs(z_all, feat_names):
        print(f"  r={r:+.4f}  {a}  <->  {b}")

    scal_red = redundancy_pairs(z_scal, list(SCALAR_KEYS), threshold=0.90)
    print("\nScalar-block redundancy at |r| >= 0.90 (these inflate the block's "
          "effective weight):")
    for a, b, r in scal_red:
        print(f"  r={r:+.4f}  {a}  <->  {b}")

    eig = np.linalg.eigvalsh(np.cov(v, rowvar=False))[::-1]
    eig = np.clip(eig, 0, None)
    cum = np.cumsum(eig) / eig.sum()
    pr = eig.sum() ** 2 / (eig**2).sum()
    print(f"\nEffective dimensionality of the {v.shape[1]}-dim whitened space: "
          f"participation ratio={pr:.2f}, "
          f"PCs for 90%/95%/99% variance = {int(np.searchsorted(cum, 0.90)) + 1}/"
          f"{int(np.searchsorted(cum, 0.95)) + 1}/{int(np.searchsorted(cum, 0.99)) + 1}")

    rule("3. DISTANCE MATRIX")
    dist = cosine_distance_matrix(v)
    print(f"metric: cosine on the whitened vector, LTAS block weight "
          f"{LTAS_WEIGHT:.2f} / scalars {1 - LTAS_WEIGHT:.2f}, "
          f"{shape.shape[1]} bands + {scal.shape[1]} scalars = {v.shape[1]} dims")
    off = dist[np.triu_indices(len(names), k=1)]
    print(describe("all pairs", off))
    print(f"pairs at distance < 1e-6 (collapsed): {(off < 1e-6).sum()}")
    closest = np.argsort(off)[:5]
    iu = np.triu_indices(len(names), k=1)
    print("\nfive closest pairs in the corpus:")
    for k in closest:
        print(f"  {off[k]:.4f}  {names[iu[0][k]]}\n          {names[iu[1][k]]}")
    print("five most distant pairs:")
    for k in np.argsort(off)[-5:]:
        print(f"  {off[k]:.4f}  {names[iu[0][k]]}\n          {names[iu[1][k]]}")

    rule("4. THE KEY QUESTION -- WITHIN-FAMILY vs ACROSS-FAMILY")
    declared = [family_of(n, DECLARED_FAMILIES) for n in names]
    extended = [family_of(n, EXTENDED_FAMILIES) for n in names]
    res_declared = analyse(dist, declared, names,
                           "declared families (the two named in the gate)", args.trials)
    res_extended = analyse(dist, extended, names,
                           "extended families (every same-amp group in the filenames)",
                           args.trials)
    corrected = [family_of(n, CORRECTED_FAMILIES) for n in names]
    res_corrected = analyse(dist, corrected, names,
                            "corrected families (POST-HOC, diagnostic only, NOT gated)",
                            args.trials)

    print("\n-- per-family within-distance breakdown (declared families) --")
    for key in DECLARED_FAMILIES:
        idx = [i for i, f in enumerate(declared) if f == key]
        sub = np.array([dist[i, j] for i, j in itertools.combinations(idx, 2)])
        other = np.array([dist[i, j] for i in idx for j in range(len(names))
                          if declared[j] != key])
        print(f"  {key:<22} n={len(idx)} within: med={np.median(sub):.4f} "
              f"max={sub.max():.4f}  |  to non-family: med={np.median(other):.4f} "
              f"min={other.min():.4f}")

    # The Ceriatone family mixes amp-only captures with pedal-in-front captures;
    # if it splits, that shows up as a bimodal within-family distribution.
    kk = [i for i, f in enumerate(declared) if f == "ceriatone_king_kong"]
    pedal = [i for i in kk if "NAM Capture" in names[i]]
    amp = [i for i in kk if i not in pedal]
    if pedal and amp:
        d_pp = np.array([dist[i, j] for i, j in itertools.combinations(pedal, 2)])
        d_aa = np.array([dist[i, j] for i, j in itertools.combinations(amp, 2)])
        d_pa = np.array([dist[i, j] for i in pedal for j in amp])
        print(f"\n  King Kong sub-structure: pedal-capture pairs med={np.median(d_pp):.4f}, "
              f"amp-only pairs med={np.median(d_aa):.4f}, "
              f"pedal-vs-amp med={np.median(d_pa):.4f}")

    rule("5. METRIC ROBUSTNESS -- does the verdict depend on the metric?")
    variants: dict[str, np.ndarray] = {
        "cosine, whitened, LTAS 0.70": dist,
        "euclidean, whitened, LTAS 0.70": euclidean_distance_matrix(v),
        "cosine, whitened, LTAS 1.00 (LTAS only)": cosine_distance_matrix(z_shape),
        "cosine, whitened, LTAS 0.00 (scalars only)": cosine_distance_matrix(z_scal),
        "cosine, whitened, LTAS 0.50": cosine_distance_matrix(
            whitened_vectors(shape, scal, 0.50)[0]),
        "cosine, whitened, LTAS 0.90": cosine_distance_matrix(
            whitened_vectors(shape, scal, 0.90)[0]),
        "euclidean, RAW dB LTAS shape only (no whitening)":
            euclidean_distance_matrix(shape),
        "euclidean, RAW scalars only (no whitening)": euclidean_distance_matrix(scal),
    }
    print(f"{'metric':<50}{'AUC decl':>10}{'AUC ext':>10}{'d-prime ext':>13}{'top1 ext':>10}")
    for label, dm in variants.items():
        wd, ad = split_pairs(dm, declared)
        we, ae = split_pairs(dm, extended)
        nn = nearest_neighbour_report(dm, names, extended)
        print(f"{label:<50}{auc_ranked(wd, ad):>10.4f}{auc_ranked(we, ae):>10.4f}"
              f"{dprime(we, ae):>13.3f}{nn['top1']:>10.3f}")

    rule("6. WHICH FEATURES CARRY THE SEPARATION")
    solo = per_feature_auc(z_all, extended)
    loo = leave_one_out_auc(v, extended, auc_ranked(*split_pairs(dist, extended)))
    order = np.argsort(-solo)
    print("Per-feature AUC used alone (extended families). 0.500 = uninformative,")
    print("< 0.500 = varies more within one amp than between amps.\n")
    print(f"{'rank':<6}{'feature':<20}{'solo AUC':>10}{'dAUC if dropped':>18}")
    print("  top 15:")
    for rank, j in enumerate(order[:15], 1):
        print(f"{rank:<6}{feat_names[j]:<20}{solo[j]:>10.4f}{loo[j]:>+18.5f}")
    print("  bottom 15:")
    for rank, j in enumerate(order[-15:], len(order) - 14):
        print(f"{rank:<6}{feat_names[j]:<20}{solo[j]:>10.4f}{loo[j]:>+18.5f}")

    n_bands = shape.shape[1]
    print(f"\nblock summary: LTAS bands solo AUC mean={solo[:n_bands].mean():.4f} "
          f"(min {solo[:n_bands].min():.4f}, max {solo[:n_bands].max():.4f}); "
          f"scalars mean={solo[n_bands:].mean():.4f} "
          f"(min {solo[n_bands:].min():.4f}, max {solo[n_bands:].max():.4f})")
    print(f"features below chance (solo AUC < 0.50): "
          f"{[feat_names[j] for j in range(len(solo)) if solo[j] < 0.50]}")
    helped = [(feat_names[j], loo[j]) for j in np.argsort(-loo)[:8] if loo[j] > 0]
    print(f"dropping these raises overall AUC (dead weight): "
          f"{[(n, round(d, 5)) for n, d in helped]}")

    print("\nEvery scalar, ranked:")
    for j in np.argsort(-solo[n_bands:]):
        print(f"  {SCALAR_KEYS[j]:<22} solo AUC={solo[n_bands + j]:.4f}   "
              f"dAUC if dropped={loo[n_bands + j]:+.5f}")

    rule("7. DIAGNOSIS -- where the separation is lost")

    def auc_for(labels_: list[str | None], dm: np.ndarray = dist) -> tuple[float, float]:
        w_, a_ = split_pairs(dm, labels_)
        return auc_ranked(w_, a_), dprime(w_, a_)

    print("7a. Is the ground truth itself heterogeneous?")
    print("    'Same amp' is not the same claim for every family. A King Kong")
    print("    channel-1 60s capture and a channel-2 80s capture are one amp on")
    print("    paper and two voicings in practice; the five 'NAM Capture <pedal>'")
    print("    files in that family are pedal captures. Re-scoring with the family")
    print("    label refined isolates how much of the loss is labelling.")
    kk_split = [("kk_pedal" if "NAM Capture" in names[i] else "kk_amp")
                if extended[i] == "ceriatone_king_kong" else extended[i]
                for i in range(len(names))]
    kk_out = [None if f == "ceriatone_king_kong" else f for f in extended]
    suhr_only = [f if f == "suhr_rl" else None for f in extended]
    for tag, lab in [("extended families, as scored above", extended),
                     ("King Kong split into pedal/amp sub-families", kk_split),
                     ("King Kong excluded from the labelling", kk_out),
                     ("Suhr RL family alone (one amp, knobs only)", suhr_only)]:
        a_, d_ = auc_for(lab)
        print(f"    {tag:<48} AUC={a_:.4f}  d-prime={d_:.3f}")

    print("\n7b. Confound: is this separating the CAPTURER's rig, not the amp?")
    print("    Each filename starts with who made the capture. If the space keys on")
    print("    interface/reamp signature it will separate capturers as well as amps,")
    print("    and none of it survives the jump to a separated song stem.")
    capturer = [" ".join(n.split()[:2]) for n in names]
    hi = centres > 10_000.0
    hi_only = whitened_vectors(shape[:, hi], scal * 0.0)[0]
    for tag, lab, dm in [("amp identity, full vector", extended, dist),
                         ("capturer identity, full vector", capturer, dist),
                         ("amp identity, bands >10 kHz only", extended,
                          cosine_distance_matrix(hi_only)),
                         ("capturer identity, bands >10 kHz only", capturer,
                          cosine_distance_matrix(hi_only))]:
        w_, a_ = split_pairs(dm, lab)
        print(f"    {tag:<42} AUC={auc_ranked(w_, a_):.4f}  d-prime={dprime(w_, a_):.3f}  "
              f"med within={np.median(w_):.4f}  med across={np.median(a_):.4f}")
    print("    With the rig held constant, does amp identity still separate?")
    for c in sorted(set(capturer)):
        idx = [i for i in range(len(names)) if capturer[i] == c]
        sub_lab = [extended[i] for i in idx]
        w_, a_ = split_pairs(dist[np.ix_(idx, idx)], sub_lab)
        if len(w_) >= 3 and len(a_) >= 3:
            print(f"      {c:<10} n={len(idx):2d}  {len(w_):3d} same-amp / {len(a_):3d} "
                  f"different-amp pairs   AUC={auc_ranked(w_, a_):.4f}  "
                  f"d-prime={dprime(w_, a_):.3f}")

    print("\n7c. Where the LTAS block's weight actually goes")
    regions = [(0.0, 100.0, "<100 Hz (below the synthetic probe's lowest note, 82 Hz)"),
               (100.0, 2500.0, "100 Hz - 2.5 kHz"),
               (2500.0, 10_000.0, "2.5 - 10 kHz"),
               (10_000.0, 1e9, ">10 kHz (above the analysis band)")]
    probe_ltas = ltas_db(probe, sr, cfg)
    zs_total = (z_shape**2).sum()
    for lo, hi_, tag in regions:
        m = (centres >= lo) & (centres < hi_)
        print(f"    {tag:<46} {m.sum():2d} bands  "
              f"{100 * (z_shape[:, m]**2).sum() / zs_total:5.1f}% of LTAS variance  "
              f"mean solo AUC={solo[:n_bands][m].mean():.4f}  "
              f"probe energy {(probe_ltas - probe_ltas.max())[m].mean():+.1f} dB rel. peak")

    print("\n7d. Effective axes of the space (PCA on the whitened vector)")
    centred = v - v.mean(axis=0)
    _, sv, vt = np.linalg.svd(centred, full_matrices=False)
    for k in range(3):
        top = np.argsort(-np.abs(vt[k]))[:6]
        share = 100 * sv[k] ** 2 / (sv**2).sum()
        print(f"    PC{k + 1} ({share:4.1f}% of variance): "
              + ", ".join(f"{feat_names[j]}({vt[k][j]:+.2f})" for j in top))

    print("\n7e. Candidate repairs, scored against both labellings")
    print("    (These are diagnostics, not a re-run of the gate. Picking the best of")
    print("     eight variants on 34 files would fit label noise, so no variant here")
    print("     is claimed as a pass -- the point is which direction moves the number.)")
    per_level = [[fingerprint(r.audio[s], sr, cfg) for s in slices] for r in rendered]
    d_ltas = np.array([p[-1].shape_vector - p[0].shape_vector for p in per_level])
    d_scal = np.array([p[-1].scalar_vector - p[0].scalar_vector for p in per_level])
    v_level = whitened_vectors(d_ltas, d_scal)[0]
    band_m = (centres >= cfg.analysis_lo_hz) & (centres <= cfg.analysis_hi_hz)
    dedup = [SCALAR_KEYS.index(k) for k in
             ("centroid", "flatness", "crest_db", "env_std_db", "transient_sharpness")]
    repairs = {
        "as shipped (48 bands + 12 scalars, LTAS 0.70)": v,
        "LTAS clipped to the analysis band 80 Hz-10 kHz": whitened_vectors(
            shape[:, band_m], scal)[0],
        "scalars de-duplicated to 5 (drop the |r|>0.95 twins)": whitened_vectors(
            shape, scal[:, dedup])[0],
        "LTAS block weight 0.50 instead of 0.70": whitened_vectors(shape, scal, 0.50)[0],
        "bands >10 kHz alone (4 features, no scalars)": hi_only,
        "+ per-level delta features (top level minus bottom)": np.hstack(
            [v, v_level * np.sqrt(0.5)]),
        "per-level delta features alone": v_level,
    }
    print(f"\n    {'variant':<52}{'AUC ext':>9}{'d-prime':>9}{'top1':>7}"
          f"{'AUC KKsplit':>13}{'d-prime':>9}")
    for tag, vec in repairs.items():
        dm = cosine_distance_matrix(vec)
        w_, a_ = split_pairs(dm, extended)
        ws, as_ = split_pairs(dm, kk_split)
        nn = nearest_neighbour_report(dm, names, extended)
        print(f"    {tag:<52}{auc_ranked(w_, a_):>9.4f}{dprime(w_, a_):>9.3f}"
              f"{nn['top1']:>7.3f}{auc_ranked(ws, as_):>13.4f}{dprime(ws, as_):>9.3f}")
    rms_dl = np.sqrt((d_ltas**2).mean(axis=1))
    print(f"\n    level-delta LTAS movement across the probe's 12 dB sweep: "
          f"median={np.median(rms_dl):.3f} dB rms, min={rms_dl.min():.3f}, "
          f"max={rms_dl.max():.3f}")

    rule("8. VERDICT")
    checks: list[tuple[str, bool, str]] = [
        ("no failed loads/renders", not failures and not bad,
         f"{len(failures)} load failures, {len(bad)} non-finite renders"),
        ("no NaN/inf in any fingerprint", nan_shape == 0 and nan_scal == 0,
         f"{nan_shape} LTAS, {nan_scal} scalar non-finite cells"),
        ("no two distinct profiles collapse to one point", (off < 1e-6).sum() == 0,
         f"{(off < 1e-6).sum()} pairs at distance < 1e-6"),
        (f"different-amp pairs stay resolved: p01/median >= {MIN_SPREAD_RATIO}",
         res_extended["across_p01_over_med"] >= MIN_SPREAD_RATIO,
         f"p01/median = {res_extended['across_p01_over_med']:.3f} (extended labels; "
         f"{res_declared['across_p01_over_med']:.3f} under the declared labels, which "
         f"call the two Fender TwinVerb captures different amps)"),
        (f"declared-family AUC >= {MIN_AUC}", res_declared["auc"] >= MIN_AUC,
         f"AUC = {res_declared['auc']:.4f}"),
        (f"declared-family d-prime >= {MIN_DPRIME}", res_declared["dprime"] >= MIN_DPRIME,
         f"d-prime = {res_declared['dprime']:.3f}"),
        (f"declared-family permutation p <= {MAX_NULL_P}", res_declared["p"] <= MAX_NULL_P,
         f"p = {res_declared['p']:.5f}"),
        (f"extended-family AUC >= {MIN_AUC}", res_extended["auc"] >= MIN_AUC,
         f"AUC = {res_extended['auc']:.4f}"),
        (f"nearest neighbour is same-amp for >= {MIN_NN_TOP1:.0%}",
         res_extended["top1"] >= MIN_NN_TOP1,
         f"top-1 = {res_extended['top1']:.3f} (extended), "
         f"{res_declared['top1']:.3f} (declared)"),
    ]
    width = max(len(c[0]) for c in checks)
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:<{width}}  {detail}")

    passed = all(ok for _, ok, _ in checks)
    print(f"\nGATE B: {'PASS' if passed else 'FAIL'}")
    if not passed:
        print("\nThe space is far better than chance and its local structure is right")
        print("(the nearest neighbour of a capture is usually another capture of the")
        print("same amp), but the ordering is not reliable enough over the whole")
        print("distribution to build a ranked matcher on as it stands. Section 7 is")
        print("the diagnosis; the shortest paths out are, in order of measured effect:")
        print("  1. the fingerprint collapses the probe's three drive levels into one")
        print("     spectrum, discarding the level dependence probe.py exists to create")
        print("  2. the 12 scalars are ~2 independent quantities wearing 12 names")
        print("  3. the four bands above 10 kHz separate the capturer's rig nearly as")
        print("     sharply as they separate amps, and a separated song stem will not")
        print("     carry that region")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
