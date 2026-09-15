**GATE B: FAIL.** Tool written at `C:\Users\NZXT\Documents\ToneHound\engine\tools\gate_b.py`. Run with `PYTHONPATH=engine python engine/tools/gate_b.py` (add `--cache <dir>` to reuse renders; ~6.4 s/profile, 34 profiles). Exit code 1. Existing suite unaffected: `172 passed in 50.35s`.

## What was run

34 real captures (`_`-prefixed round-trip fixtures excluded), probe = 1402560 samples = 29.22 s at three levels, peak 0.5012. All 34 loaded and rendered, 0 non-finite samples, 0 NaN/inf feature cells, 0 constant features, 0 bit-identical render pairs, receptive field 4093 for all. Metric: cosine on the whitened vector (per-dimension z-score across the corpus, blocks rescaled so LTAS holds 0.70 and the 12 scalars 0.30 of expected squared length), 48 bands + 12 scalars = 60 dims. AUC computed by direct enumeration and by rank sums; they agree to 1.1e-16.

## The key question: within-family vs across-family

```
-- declared families (Ceriatone King Kong x10, [Suhr RL] x8, 16 singletons) --
within-family   n=73   min=0.0274 p05=0.0458 med=0.5035 mean=0.6191 p95=1.5153 max=1.7956
across-family   n=488  min=0.0141 p05=0.2346 med=1.0645 mean=1.0619 p95=1.7651 max=1.8909
AUC = 0.7394    d-prime = 0.886    histogram overlap = 0.5019
cleanly separated (max within < min across) = False
across-family pairs closer than the within-family median: 87/488 (17.8%)
within-family pairs further than the across-family median: 14/73 (19.2%)
permutation null (5000 label shuffles): mean AUC=0.5004 p99=0.6188 -> p = 0.00020
retrieval over 18 family members: top-1 same-family=0.944  mAP=0.689

-- extended families (also the 5150/6505+/6534+/TwinVerb/Magnatone/Splawn groups) --
within n=85 med=0.4854 max=1.7956 | across n=476 med=1.1052 min=0.0930
AUC = 0.7755    d-prime = 1.053    overlap = 0.4681
across closer than within-median: 72/476 (15.1%)
permutation p = 0.00020
retrieval over 33 members: top-1=0.818  mAP=0.704  recall@(family size-1)=0.607
```

Against the pre-registered bars (AUC ≥ 0.90, d-prime ≥ 1.5, permutation p ≤ 0.01, NN top-1 ≥ 0.80): AUC and d-prime fail on both labellings; the null and retrieval checks pass. The separation is real and nowhere near chance (p = 2e-4, null mean 0.5004), but ~16-18% of different-amp pairs sit closer than the median same-amp pair. That is not good enough to rank 300 profiles on.

The verdict does not depend on the metric — eight variants, all in the same band:

```
cosine, whitened, LTAS 0.70                      AUC decl 0.7394  ext 0.7755  d' 1.053  top1 0.818
euclidean, whitened, LTAS 0.70                            0.6923      0.7311     0.908       0.818
cosine, LTAS only                                         0.7177      0.7554     0.970       0.788
cosine, scalars only                                      0.7627      0.7936     1.126       0.818
cosine, LTAS 0.50                                         0.7523      0.7870     1.094       0.848
cosine, LTAS 0.90                                         0.7254      0.7629     1.000       0.788
euclidean, RAW dB LTAS (no whitening)                     0.6935      0.7345     0.898       0.788
euclidean, RAW scalars (no whitening)                     0.6878      0.7165     0.833       0.576
```

Note the LTAS-above-scalars prescription is not what the data prefers: scalars alone (0.7936) beat LTAS alone (0.7554) and beat the shipping 0.70 weighting.

## Diagnosis: four separate causes, measured

**1. Part of the "failure" is the ground truth, not the space.** Re-scoring with the family label refined:

```
extended families, as scored above            AUC=0.7755  d'=1.053
King Kong split into pedal/amp sub-families   AUC=0.8571  d'=1.511
King Kong excluded from the labelling         AUC=0.8790  d'=1.800
Suhr RL family alone (one amp, knobs only)    AUC=0.8539  d'=1.652
```

The Ceriatone family is not acoustically one thing. Its five `NAM Capture <pedal> Chase Tone Secret Pre Deco` files are pedal captures and cluster at median 0.0563; the five amp-channel captures sit at median 0.9874; pedal-vs-amp is 0.8788. The single largest within-family distance in the whole corpus is `Channel 1 60s mode` vs `chan2 70s both br sw L` at **1.7956** — same amp, different channel and decade voicing, and the space is arguably right to call them far apart. Likewise the two closest "different amp" pairs are label errors, not metric errors: `5150 BlockLetter - NoBoost` vs `6505+ Red Ch` at 0.0930 (the 6505+ is the renamed 5150, literally the same circuit) and `NAM Capture Protein` vs `Boss OS-2` at 0.0962 (both overdrive pedals). Even so, the cleanest available labelling (Suhr, one amp, knobs only) still only reaches 0.854 — below the bar. So the space is genuinely short, just not by as much as 0.7755 suggests.

**2. The 60 features are ~2 real dimensions.** Participation ratio 2.62; 4/6/11 PCs for 90/95/99% of variance.

```
PC1 (54.7%): slope(-0.21) rolloff85(-0.21) centroid(-0.21) fizz_ratio(-0.21) rolloff95(-0.20) flatness(-0.18)
PC2 (27.7%): env_std_db(-0.26) env_p90_p10_db(-0.25) transient_sharpness(-0.22) ltas_62Hz(+0.20) ltas_70Hz(+0.20)
PC3 ( 5.3%): rms_to_peak(+0.28) crest_db(-0.27) ltas_190Hz(-0.27) ltas_90Hz(-0.26)
```

PC1 is brightness, PC2 is compression, and 82% of the space is those two. The scalar block is the worst offender: `centroid`↔`fizz_ratio` r=+0.9946, `spread`↔`flatness` r=+0.9894, `centroid`↔`slope` r=+0.9824, `rolloff85`↔`slope` r=+0.9791, `rolloff85`↔`rolloff95` r=+0.9702. And `crest_db` and `rms_to_peak` are not merely correlated but algebraically identical — `crest_db == -20*log10(rms_to_peak)` holds to 3.6e-15 across all 34 files, straight from the definitions in `features.py`. Eight of the twelve scalars are one brightness axis restated; giving them 0.30 of the vector's length means brightness is counted many times over.

**3. The strongest features are outside the analysis band and are partly a capturer fingerprint.** Solo-AUC ranking is topped by `ltas_15032Hz` (0.8095), `ltas_13268Hz` (0.7942), `ltas_11711Hz` (0.7918) — all above the 10 kHz analysis ceiling. They are not noise floor (median 8.5-10.6 dB below each profile's LTAS peak, and the probe itself has energy there), but:

```
amp identity, full vector             AUC=0.7755  med within=0.4854  med across=1.1052
capturer identity, full vector        AUC=0.7474  med within=0.5425  med across=1.1249
amp identity, bands >10 kHz only      AUC=0.8326  med within=0.0156  med across=1.7316
capturer identity, bands >10 kHz only AUC=0.7849  med within=0.0259  med across=1.7726
```

Four bands above 10 kHz outscore the entire 60-dim vector (0.8326 vs 0.7755), and they separate *who made the capture* almost as sharply as *which amp* (0.7849, with same-capturer median distance 0.0259 against 1.7726 across). That is a shortcut: the song side of the pipeline reads a separated stem of a mixed, mastered, probably lossy-coded recording, which carries none of that region's rig signature. Holding the rig constant, amp identity does still separate on its own (Helga B AUC 0.8030 d'=1.206; Tim R 0.9375 d'=2.620), so the space is not *only* a rig fingerprint — but its best-scoring features here largely are.

**4. Dead weight is where the probe has no signal and where every amp agrees.** Seven bands sit below 82 Hz, the probe's lowest fundamental (probe energy −29.9 dB rel. peak there), yet after z-scoring they take 14.6% of the LTAS block's variance — `ltas_43Hz` has the second-largest spread in the corpus (7.13 dB std) driven by each capture's DC/bias term. The 100 Hz-2.5 kHz region is 26 of 48 bands and 54.2% of the variance for a mean solo AUC of 0.5839, with the worst individual bands at chance: `ltas_1801Hz` 0.5103, `ltas_1589Hz` 0.5161, `ltas_2040Hz` 0.5410, `ltas_1238Hz` 0.5412. Dropping features that *raises* overall AUC: `ltas_131Hz` (+0.00475), `ltas_2040Hz` (+0.00306), `ltas_116Hz` (+0.00304), `ltas_102Hz` (+0.00292), `ltas_2311Hz` (+0.00269). No feature scores below chance, so nothing is actively inverted — but a third of the LTAS grid is close to inert.

Also worth noting: z-scoring is doing essential work and should not be removed. Raw scalar variance is 46.85% `rolloff95` + 35.36% `rolloff85` + 15.08% `centroid` = 97.3% in three units-of-Hz features; `transient_sharpness` (std 0.0022) contributes 0.00%. An un-normalised Euclidean scalar distance drops top-1 retrieval to 0.576.

## Recommendations, in measured order of effect

These are diagnostics, not a rescued pass — picking the best of seven variants on 34 files fits label noise, and none of them clears 0.90.

```
variant                                               AUC ext  d-prime   top1  AUC KKsplit  d-prime
as shipped (48 bands + 12 scalars, LTAS 0.70)          0.7755    1.053  0.818       0.8571    1.511
LTAS clipped to the analysis band 80 Hz-10 kHz         0.7590    0.973  0.848       0.8317    1.353
scalars de-duplicated to 5 (drop the |r|>0.95 twins)   0.7715    1.044  0.818       0.8475    1.461
LTAS block weight 0.50 instead of 0.70                 0.7870    1.094  0.848       0.8654    1.536
bands >10 kHz alone (4 features, no scalars)           0.8326    1.023  0.636       0.8863    1.260
+ per-level delta features (top level minus bottom)    0.7846    1.142  0.788       0.8936    1.831
per-level delta features alone                         0.7285    0.903  0.727       0.8582    1.551
```

1. **Extract per-level features. This is the largest available win and the cheapest.** `fingerprint()` runs Welch over the whole concatenated three-level probe, averaging −18/−12/−6 dBFS into one spectrum — it throws away the exact quantity `probe.py`'s docstring says is "one of the most discriminative features available", and `level_slices()` already exists unused. Adding the top-minus-bottom deltas lifts d-prime from 1.053 to 1.142 and, on the King Kong-split labelling, 1.511 → **1.831**, the best d-prime of any variant. Level-delta features *alone* reach AUC 0.7285 with only 60 dims of difference signal. Best single delta features: `dLTAS_3361Hz` 0.7395, `dLTAS_4888Hz` 0.6989, `dLTAS_5538Hz` 0.6930. Movement across the sweep is median 0.879 dB rms (min 0.235, max 2.151) — small, which supports the earlier finding that `PROBE_LEVELS_DBFS = (-18,-12,-6)` sits above most of this corpus's compression knee. Widening the sweep downward (e.g. −36/−24/−12) should make this term substantially stronger.
2. **Collapse the scalar block.** Delete `rms_to_peak` outright (it is `crest_db` re-expressed). Keep one brightness scalar, not five — `rolloff85` is the best single one (0.7117). De-duplicating to five costs almost nothing on this corpus (0.7755 → 0.7715) while removing the hidden multiple-counting of brightness that makes PC1 54.7% of the space.
3. **Do not ship the >10 kHz bands as a headline discriminator, and validate them against a real stem before trusting them.** They score best here and are the most likely to evaporate on the song side. The honest test is a profile-to-stem experiment, not a profile-to-profile one; until that exists, treat their contribution as unverified rather than as the strongest signal.
4. **Rebalance where the LTAS grid spends its resolution.** Bands below 82 Hz measure DC, and 100 Hz-2.5 kHz is over-sampled at near-chance discriminability. Either restrict to the analysis band (which improves top-1 to 0.848 while lowering AUC — it removes the >10 kHz shortcut, so the drop is partly a *good* sign) or reallocate bands toward 2.5-10 kHz, which is the only region with both real probe energy (−2.7 dB rel. peak) and real discriminability (mean solo AUC 0.6943).
5. **Fix the ground truth before re-running the gate.** Split `ceriatone_king_kong` into pedal captures and amp-channel captures, merge 5150 BlockLetter with 6505+, and consider whether "same amp, different channel" should count as within-family at all. On the current labelling the gate is partly measuring a filename convention.

My own reading of whether v1 is dead: it is not. Top-1 retrieval of 0.944 (declared) / 0.818 (extended) says the local neighbourhood structure is already correct — the nearest capture to a capture is usually another capture of the same amp, which is the operation a matcher performs. What fails is the tail of the distribution and the separation margin, and recommendations 1 and 2 alone move d-prime from 1.511 to 1.831 on the cleaner labelling without any new signal. I would re-run this gate after those two changes and a corrected labelling before deciding anything structural. But as it stands today, on the criteria set before the numbers were seen, this is a FAIL and should not be built on.