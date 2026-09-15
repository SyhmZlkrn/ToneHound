"""Is the MERT embedding a better coordinate system than the DSP fingerprint?

The engine had a hand-built feature space -- 48 LTAS bands plus eleven scalars,
whitened and compared by cosine -- and now has a learned one. Swapping them on a
hunch would be how a project ends up with two matchers and no idea which is
worse, so this runs the *same* retrieval protocol over both and prints them side
by side.

**The protocol is the repo's, not a new one.** `stem_test.py` established it and
its numbers are what these are compared against: ground truth is exact and free,
because profile *i*'s own render, degraded, must still retrieve profile *i*. Two
kinds of query, in rising order of honesty:

  * *same performance* -- query and index are the same render. This only
    measures whether degradation destroyed the signal.
  * *unseen performance* -- the query is that profile rendered through a
    **different DI take**, so the two sides never shared a note. This is the
    protocol behind the 0.176 top-1 the old fingerprint scored, and it is the
    one worth quoting.

On top of either, `--degrade` applies the chain a real recording imposes
(mastering EQ, a drum bed, limiting, a real mp3 round trip) and `--separate`
adds htdemucs_6s, which is the full path the product actually takes.

    PYTHONPATH=engine python engine/tools/gate_mert.py --queries .cache/renders_funk
    PYTHONPATH=engine python engine/tools/gate_mert.py --queries .cache/renders_funk \
        --degrade
    PATH=.cache/bin:$PATH PYTHONPATH=engine python engine/tools/gate_mert.py \
        --queries .cache/renders_funk --degrade --separate
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tonehound import match, separate
from tonehound.config import GUITAR, SAMPLE_RATE
from tonehound.embed import EmbedConfig, MertEmbedder, combine_layers
from tonehound.features import fingerprint
from tonehound.tone_index import ToneEntry, ToneIndex

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def ranks_from_scores(scores: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Rank of each query's own profile, 1-based. ``scores`` is (queries, index)."""
    order = np.argsort(-scores if higher_is_better else scores, axis=1)
    return np.array([int(np.where(order[i] == i)[0][0]) + 1
                     for i in range(len(scores))])


def report(name: str, ranks: np.ndarray, n: int) -> dict[str, float]:
    return {
        "space": name,
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "chance_top1": 1.0 / n,
    }


def print_table(rows: list[dict[str, float]], n: int, condition: str) -> None:
    print(f"\n  {condition}   ({n} profiles, chance top-1 = {1.0 / n:.3f})")
    print(f"  {'space':<22} {'top-1':>7} {'top-5':>7} {'top-10':>7} "
          f"{'MRR':>7} {'mean':>7} {'med':>5}")
    print("  " + "-" * 66)
    for r in rows:
        print(f"  {r['space']:<22} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['top10']:>7.3f} {r['mrr']:>7.3f} {r['mean_rank']:>7.1f} "
              f"{r['median_rank']:>5.0f}")


LAYER_GROUPS: list[tuple[str, tuple[int, ...]]] = [
    ("proj+early 0-3", (0, 1, 2, 3)),
    ("early 4-7", (4, 5, 6, 7)),
    ("lower-mid 8-11", (8, 9, 10, 11)),
    ("mid 12-15", (12, 13, 14, 15)),
    ("upper-mid 16-19", (16, 17, 18, 19)),
    ("late 20-23", (20, 21, 22, 23)),
    ("last 21-24", (21, 22, 23, 24)),
    ("wide 4-15", tuple(range(4, 16))),
    ("all 0-24", tuple(range(25))),
]
"""Contiguous four-layer bands across the stack, plus two wider spans.

Four at a time because that is the width the default uses, so the comparison is
between *where* in the stack the layers are taken from and nothing else.
"""


def sweep(embedder: MertEmbedder, entries: list[ToneEntry],
          index_audio: list[np.ndarray], query_audio: list[np.ndarray],
          names: list[str]) -> list[dict[str, float]]:
    """Score every layer group, from a single forward pass per signal.

    `embed_layers` returns all 25 pooled layers at once, so the 34 candidate
    configurations below cost one pass over the corpus rather than 34. On a GPU
    that is seconds; it is the difference between settling the layer question
    and guessing at it.
    """
    print("  sweeping layers (one forward pass per signal) ...", flush=True)
    index_all = np.stack([embedder.embed_layers(x, SAMPLE_RATE) for x in index_audio])
    query_all = np.stack([embedder.embed_layers(x, SAMPLE_RATE) for x in query_audio])

    out: list[dict[str, float]] = []
    singles: list[tuple[int, float, float]] = []
    for layer in range(index_all.shape[1]):
        ranks = _ranks_for(entries, index_all, query_all, (layer,))
        singles.append((layer, float((ranks == 1).mean()),
                        float((ranks <= 5).mean())))
    for label, layers in LAYER_GROUPS:
        out.append(report(f"MERT {label}",
                          _ranks_for(entries, index_all, query_all, layers),
                          len(names)))

    print("\n  single layers (top-1 / top-5)")
    for row in range(0, len(singles), 5):
        print("    " + "  ".join(f"L{i:<2} {t1:.3f}/{t5:.3f}"
                                 for i, t1, t5 in singles[row:row + 5]))
    return out


def _ranks_for(entries: list[ToneEntry], index_all: np.ndarray,
               query_all: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    index_vectors = np.stack([combine_layers(m, layers) for m in index_all])
    query_vectors = np.stack([combine_layers(m, layers) for m in query_all])
    built = ToneIndex(entries=entries, vectors=index_vectors)
    sims = np.stack([built.similarities(q) for q in query_vectors])
    return ranks_from_scores(sims, higher_is_better=True)


# --------------------------------------------------------------------------
# the degradation chain
# --------------------------------------------------------------------------


def degrade(x: np.ndarray, bed: np.ndarray, rng: np.random.Generator,
            guitar_db: float) -> np.ndarray:
    """Mastering EQ, a backing bed, limiting, and a real mp3 round trip.

    Imported from `stem_test` rather than reimplemented: a second copy of the
    degradation would make these numbers incomparable with the ones already
    recorded in `docs/`.
    """
    import stem_test

    y = stem_test.mastering_eq(x, SAMPLE_RATE, rng)
    if bed.any():
        y = stem_test.add_backing(y, bed[:len(y)], guitar_db)
    y = stem_test.limit(y)
    return stem_test.codec_roundtrip(y, SAMPLE_RATE, None)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=pathlib.Path,
                    default=ROOT / ".cache" / "renders_djent",
                    help="renders the index is built from")
    ap.add_argument("--queries", type=pathlib.Path, default=None,
                    help="renders the queries come from; default = --index "
                         "(same performance). Point at another DI's renders "
                         "for the unseen-performance protocol.")
    ap.add_argument("--profiles", type=pathlib.Path,
                    default=ROOT / "assets" / "dev_profiles")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--degrade", action="store_true",
                    help="mastering EQ, drum bed, limiting, mp3")
    ap.add_argument("--separate", action="store_true",
                    help="also run htdemucs_6s over the degraded query")
    ap.add_argument("--guitar-db", type=float, default=-6.0,
                    help="guitar level against the bed")
    ap.add_argument("--layers", default="8,9,10,11",
                    help="comma-separated MERT layers; negative counts from the "
                         "last. Try lower layers: upper ones drift toward "
                         "musical semantics rather than timbre.")
    ap.add_argument("--sweep", action="store_true",
                    help="score every layer group from one forward pass. The "
                         "default takes the last four layers because that is "
                         "what was asked for, but upper layers of a "
                         "genre-tuned checkpoint encode what is being played "
                         "rather than what it was played through -- this is how "
                         "that gets settled by measurement.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0, help="use only N profiles")
    args = ap.parse_args()

    index_dir = args.index
    query_dir = args.queries or args.index
    names = sorted(p.stem for p in index_dir.glob("*.npy"))
    names = [n for n in names if (query_dir / f"{n}.npy").exists()]
    if args.limit:
        names = names[:args.limit]
    if not names:
        print(f"No renders shared between {index_dir} and {query_dir}")
        return 1

    n_samples = int(args.seconds * SAMPLE_RATE)
    index_audio = [np.load(index_dir / f"{n}.npy")[:n_samples] for n in names]
    query_audio = [np.load(query_dir / f"{n}.npy")[:n_samples] for n in names]

    condition = ("same performance" if query_dir == index_dir
                 else f"unseen performance ({query_dir.name})")
    if args.degrade:
        condition += " + mix/master/mp3"
    if args.separate:
        condition += " + htdemucs_6s"

    print(f"index   : {index_dir.name}  ({len(names)} profiles)")
    print(f"queries : {query_dir.name}")
    print(f"window  : {args.seconds:.0f}s")

    # -- degrade the queries --------------------------------------------
    if args.degrade:
        import stem_test

        rng = np.random.default_rng(args.seed)
        bed = stem_test.load_beds(3, n_samples, SAMPLE_RATE, rng)
        if not bed.any():
            print("  (no drum beds found; degrading without a backing bed)")
        print("  degrading queries ...", flush=True)
        query_audio = [degrade(q, bed, rng, args.guitar_db) for q in query_audio]

    if args.separate:
        print("  separating queries (about realtime) ...", flush=True)
        separator = separate.Separator(ROOT / ".cache" / "uvr_models")
        pulled = []
        for i, q in enumerate(query_audio, 1):
            result = separate.isolate(q, SAMPLE_RATE, separator=separator,
                                      tag=f"q{i}")
            pulled.append(result.samples)
            print(f"    [{i}/{len(query_audio)}] {result.stem}", flush=True)
        query_audio = pulled

    rows: list[dict[str, float]] = []

    # -- the DSP fingerprint --------------------------------------------
    print("\n  fingerprinting (DSP) ...", flush=True)
    started = time.monotonic()
    idx_fp = [fingerprint(x, SAMPLE_RATE, GUITAR) for x in index_audio]
    shape = np.array([f.shape_vector for f in idx_fp])
    scalars = np.array([f.scalar_vector for f in idx_fp])
    ref = (match.zscore_reference(shape), match.zscore_reference(scalars))
    index_matrix = match.whiten(shape, scalars, ref)

    q_fp = [fingerprint(x, SAMPLE_RATE, GUITAR) for x in query_audio]
    q_matrix = match.whiten(np.array([f.shape_vector for f in q_fp]),
                            np.array([f.scalar_vector for f in q_fp]), ref)
    distances = match.cosine_distance(q_matrix, index_matrix)
    rows.append(report("DSP fingerprint",
                       ranks_from_scores(distances, higher_is_better=False),
                       len(names)))
    dsp_s = time.monotonic() - started

    # -- MERT, centred and uncentred ------------------------------------
    print(f"  embedding (MERT) ...", flush=True)
    started = time.monotonic()
    embedder = MertEmbedder(EmbedConfig(
        layers=tuple(int(v) for v in args.layers.split(",")),
        device=args.device, dtype="float16" if args.fp16 else "float32"))
    print(f"    MERT on {embedder.device}", flush=True)
    entries = [ToneEntry(key=f"k{i}", name=name,
                         path=args.profiles / f"{name}.nam")
               for i, name in enumerate(names)]

    index_vectors = np.stack([embedder.embed(x, SAMPLE_RATE) for x in index_audio])
    query_vectors = np.stack([embedder.embed(x, SAMPLE_RATE) for x in query_audio])
    mert_s = time.monotonic() - started

    tone_index = ToneIndex(entries=entries, vectors=index_vectors)
    centred = np.stack([tone_index.similarities(q) for q in query_vectors])
    rows.append(report(f"MERT layers {args.layers}",
                       ranks_from_scores(centred, higher_is_better=True),
                       len(names)))

    # The uncentred row is the control: it is what the ranking would be if the
    # cone were ignored, and its collapse is the argument for centring.
    raw = query_vectors @ index_vectors.T
    rows.append(report("MERT (raw cosine)",
                       ranks_from_scores(raw, higher_is_better=True),
                       len(names)))

    if args.sweep:
        rows.extend(sweep(embedder, entries, index_audio, query_audio, names))

    print_table(rows, len(names), condition)

    off = ~np.eye(len(names), dtype=bool)
    print(f"\n  spread of index-to-index similarity")
    index_raw = index_vectors @ index_vectors.T
    index_centred = tone_index.matrix @ tone_index.matrix.T
    print(f"    raw      mean {index_raw[off].mean():+.3f}  "
          f"std {index_raw[off].std():.3f}")
    print(f"    centred  mean {index_centred[off].mean():+.3f}  "
          f"std {index_centred[off].std():.3f}")
    print(f"\n  cost: DSP {dsp_s:.1f}s, MERT {mert_s:.1f}s "
          f"for {2 * len(names)} signals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
