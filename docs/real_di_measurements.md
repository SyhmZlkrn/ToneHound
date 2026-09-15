# Real-DI measurements (raw)

## Gate B

GATE B, 34 profiles, pre-registered bars AUC>=0.90, d'>=1.50, top-1>=0.80. All three probes FAIL, and both real DIs fail WORSE than the synthetic probe on the two headline numbers.

                          SYNTHETIC        DJENT DI          FUNK DI
  declared  AUC           0.7721           0.7187 (-0.053)   0.7798 (+0.008)
  declared  d'            1.014            0.816  (-0.198)   1.065  (+0.051)
  declared  top-1         0.833            0.833  ( 0.000)   0.833  ( 0.000)
  extended  AUC           0.8052           0.7594 (-0.046)   0.8105 (+0.005)
  extended  d'            1.177            0.991  (-0.186)   1.227  (+0.050)
  extended  top-1         0.758            0.758  ( 0.000)   0.727  (-0.031)
  corrected AUC (post-hoc)0.8836           0.8410 (-0.043)   0.8689 (-0.015)
  corrected d'            1.665            1.429             1.597
  corrected top-1         0.818            0.818             0.818
  VERDICT                 FAIL             FAIL              FAIL

Failed checks:
  synthetic: 4 FAIL (decl AUC, decl d', ext AUC, top-1)
  Djent:     5 FAIL -- the same four PLUS "different-amp pairs stay resolved:
             p01/median >= 0.1" which now reads 0.092 (synthetic 0.107).
             This check PASSED on synthetic and on Funk (0.122); it is the
             only structural regression, and it is marginal.
  Funk:      4 FAIL, same set as synthetic.
Permutation null p = 0.00020 on all three (5000 trials), so all three spaces are
far better than chance.

Distance-matrix shape (all 561 pairs):
  synthetic  min 0.0093  med 0.9743  max 1.8944
  Djent      min 0.0058  med 1.0581  max 1.9462
  Funk       min 0.0075  med 1.0480  max 1.9455  (read from report)
Renders sane on all three: 0 non-finite, 0 collapsed pairs, 0 bit-identical.
Render peak abs: synthetic 0.168-0.480, Djent 0.113-0.464, Funk 0.029-0.459.

7b, capturer confound (extended labels):
                                    SYNTHETIC   DJENT     FUNK
  amp identity, full vector          0.8052    0.7594    0.8105
  capturer identity, full vector     0.7722    0.7522    0.7609
  amp identity, >10 kHz only         0.8321    0.7282    0.7455
  capturer identity, >10 kHz only    0.7861    0.7254    0.7441
  Helga B rig held constant          0.8276    0.8325    0.8030
  Tim R rig held constant            0.9250    0.9000    1.0000
The >10 kHz block no longer beats the full vector on real DI (synthetic: 0.8321
> 0.8052; Djent: 0.7282 < 0.7594; Funk: 0.7455 < 0.8105). The amp-vs-capturer
gap in that block narrows to almost nothing on real DI (Djent 0.7282 vs 0.7254;
Funk 0.7455 vs 0.7441).

7c, probe energy per LTAS band group (dB rel. peak) -- this is the direct
measurement of the discovery:
                       SYNTHETIC   DJENT     FUNK
  <100 Hz               -30.0     -23.3     -29.5
  100 Hz - 2.5 kHz       -7.6     -10.8     -13.5
  2.5 - 10 kHz           -2.7     -37.5     -30.5
  >10 kHz               -15.2     -50.1     -48.8
mean solo AUC of the >10 kHz group: synthetic 0.7948, Djent 0.7368, Funk 0.7531.

Top solo-AUC feature, all three probes, is still ltas_15032Hz (synthetic 0.8268,
Djent 0.7503, Funk 0.7635) despite that band sitting ~50 dB down on real DI.
Djent top-8 are all LTAS bands >= 5.5 kHz; Funk's top 6 mix ltas_15032Hz,
rolloff95, ltas_4888Hz, ltas_11711Hz, spread, ltas_13268Hz.

7e level-delta LTAS movement over the 12 dB sweep:
  synthetic median 1.958 dB rms (0.188-7.458)
  Djent     median 2.669 dB rms (0.010-9.747)
  Funk      median 2.816 dB rms (0.006-8.942)

## Stage 1

STAGE 1 self-retrieval top-1 (engine/tools/stem_test.py, 34 profiles).
Only the Djent run was requested/run; Funk was gate_b only.

  stage                SYNTHETIC   DJENT DI    delta
  clean (control)        1.000      1.000       0.000
  mastering EQ           0.971      0.853      -0.118
  limiting               1.000      0.882      -0.118
  mp3 128k               1.000      1.000       0.000
  + drums/bass bed       0.147      0.059      -0.088
  FULL CHAIN             0.118      0.029      -0.089
  VERDICT               FAIL        FAIL (bar 0.80)

Djent full-chain band ablation:
  all 48 bands           top-1 0.029  top-5 0.294
  >10 kHz dropped        top-1 0.059  top-5 0.294
  >10 kHz ONLY           top-1 0.176  top-5 0.441
Median rank at full chain: synthetic 8.0-ish, Djent 8.0.

Fingerprint drift columns (Djent, dB): mastering EQ 1.87 (>10k) / 0.64 (80Hz-10k);
limiting 0.67/0.40; mp3 128k 0.40/0.07; + bed 6.15/3.48; FULL CHAIN 6.62/3.63.

Two things changed direction vs synthetic:
 - mastering EQ and limiting are NO LONGER nearly free on a real DI
   (0.971 -> 0.853, 1.000 -> 0.882). mp3 128k still is.
 - the bleed gap is larger, not smaller: real DI full chain is 0.029, a quarter
   of the synthetic 0.118. INSTRUMENT BLEED is still the whole domain gap, and
   it is worse when the guitar is real.

## Stage 2

STAGE 2, UVR htdemucs_6s separation (stem_test_uvr.py --limit 34, 9.0 s clips
from the loudest pass; Djent loudest pass starts t=60.5 s = 2*(30.00+0.25),
confirming the DI slices drive the clip).

  condition                          SYNTHETIC          DJENT DI
  mixed, mastered, mp3 (stage 1)   0.147 / 0.412 / 7.5  0.059 / 0.353 / 10.0
      >10 kHz dropped              0.147 / 0.412        0.088 / 0.265
      >10 kHz only                 0.176 / 0.559        0.382 / 0.588
  + UVR guitar stem (stage 2)      0.029 / 0.235 / 11.0 0.147 / 0.382 / 8.0
      >10 kHz dropped              0.059 / 0.235        0.206 / 0.382
      >10 kHz only                 0.059 / 0.235        0.029 / 0.176
  (top-1 / top-5 / median rank)

THE SIGN FLIPS. On the synthetic probe separation HURT (0.147 -> 0.029, a 5x
loss). On the real Djent DI separation HELPS (0.059 -> 0.147, a 2.5x gain), and
with the >10 kHz bands dropped it is better still (0.088 -> 0.206, 2.3x).
Median rank improves 10.0 -> 8.0 rather than degrading 7.5 -> 11.0.

Note the report's closing sentence still hardcodes "the unseparated floor
(0.118)" from the synthetic run; the actual Djent unseparated floor in this run
is 0.059 (stage-1 row) / 0.029 (stem_test full chain).

## Domain

SEPARATION DAMAGE + DOMAIN-MATCHED INDEXING (stem_domain_test.py --limit 34,
9.0 s from the loudest pass, two independent beds A/B, 68 separations).

  decomposition (analysis band)      SYNTHETIC   DJENT DI
  systematic shift (dB rms)            11.39       2.56    (-8.83)
  per-profile residual (dB rms)         8.79       6.15    (-2.64)
  systematic share of variance         62.7%      14.8%   (-47.9 pts)

  biggest systematic shifts, SYNTHETIC:  +18.91 dB @ 190 Hz, +16.75 @ 70 Hz,
      +16.50 @ 90 Hz, +16.14 @ 62 Hz, +15.95 @ 216 Hz, -15.68 @ 4314 Hz
  biggest systematic shifts, DJENT:      -11.01 dB @ 15032 Hz, -10.84 @ 13268 Hz,
      -8.54 @ 11711 Hz, -8.02 @ 10337 Hz, -8.01 @ 102 Hz, +5.10 @ 190 Hz
  The synthetic separator error was a huge low-frequency lift; on a real DI it is
  a modest high-frequency cut. Total damage is much smaller AND its character is
  completely different.

  retrieval, query = separated stem over an UNSEEN bed B
  index built from                          SYNTHETIC        DJENT DI
  clean renders (current design)      0.029 / 0.382 / 11.0   0.206 / 0.441 / 7.0
  mixed, unseparated (control)        0.147 / 0.353 /  7.5   0.088 / 0.382 / 9.0
  separated stems over bed A          0.294 / 0.559 /  5.0   0.588 / 0.824 / 1.0
  (top-1 / top-5 / median rank)

Domain-matched indexing goes 0.294 -> 0.588 top-1 (2.0x), top-5 0.559 -> 0.824,
and median rank 5.0 -> 1.0. On the real DI the domain-matched index is the best
of the three by a wide margin (0.588 vs 0.206 vs 0.088), and the ordering of
"clean renders" vs "unseparated control" reverses (synthetic: control wins
0.147 > 0.029; Djent: clean renders win 0.206 > 0.088).
"beds differ: True" confirmed in both runs.

## Surprises

ERRORS / THINGS THAT NEED FLAGGING

1. DISK FILLED MID-RUN, and one funk render was silently truncated.
   The first gate_b --di "Funk DI.wav" crashed with OSError [Errno 28] No space
   left on device while printing section 6. It had already written 34 .npy
   renders, but one of them --
     .cache/renders_funk/Tudor N [Suhr RL] Input@HI_Dep@6_Gir@6.5_Pres@8.5_
     Bas@6.5_Mid@4_Tre@10_Mas@5_Gain@10_CHAR@mid_ERA@lft_EDG@mid_FEL@up
     ESR_0.0124_normalized-6dB.npy
   -- was 128 bytes instead of 34848128. I deleted it and re-ran; gate_b
   re-rendered that one profile in 25.4 s and produced the complete report now
   in docs/gate_b_funk.txt. THE FUNK NUMBERS REPORTED ABOVE ARE FROM THE CLEAN
   RE-RUN, not the crashed one. All 34 funk renders are now 34848128 bytes.
   The render cache has NO integrity check -- only a probe-identity stamp -- so
   a truncated .npy is exactly the silent-wrong-numbers failure mode
   probe_source.py was written to prevent, arriving through a different door.
   Worth adding a size/shape check on cache load.
   Disk now: C: 466G total, 7.0G free (99% used). .cache/renders_djent 1.2G,
   .cache/renders_funk 1.1G, .cache/renders_v2 364M. Another full-length DI run
   will not fit without freeing space.

2. The chained background job was killed by the harness at its timeout partway
   through the funk render. The python process survived (it was writing straight
   to docs/gate_b_funk.txt) so nothing was lost, but that is why the funk run
   needed manual restarting.

3. RESULTS THAT LOOK IMPOSSIBLE AND SHOULD BE CHECKED BEFORE BEING BELIEVED:
   a) ltas_15032Hz is STILL the single best solo discriminator on both real DIs
      (Djent 0.7503, Funk 0.7635) even though 7c measures the probe at -50.1 and
      -48.8 dB rel. peak in that group. Whatever those bands are reading, it is
      not signal from the DI. The synthetic explanation (Karplus-Strong white
      noise) does not apply here, so there is a second mechanism -- likely NAM
      model extrapolation noise or LTAS floor -- still driving the top feature.
   b) stem_test_uvr Djent: ">10 kHz only" scores top-1 0.382 on the UNSEPARATED
      mix, higher than the full 48-band vector (0.059) and higher than any other
      row in the table. Four bands, 50 dB down, buried under drums and bass, mp3
      128k'd (which lowpasses around 16 kHz), beating the whole fingerprint 6.5x.
      That is not plausible as amp character. After separation the same block
      collapses to 0.029, which is what you would expect if htdemucs simply zeroes
      that region.
   c) Djent gate_b newly fails the "different-amp pairs stay resolved" check at
      p01/median = 0.092 vs the 0.10 bar, while Funk passes at 0.122 and
      synthetic passed at 0.107. All three are within a hair of the bar; do not
      read the Djent failure as a real structural difference.

4. Two cosmetic reporting bugs, harmless but they will confuse a later reader:
   - gate_b section 7c still prints "below the synthetic probe's lowest note,
     82 Hz" when running a real DI.
   - stem_test_uvr's closing paragraph hardcodes "the unseparated floor (0.118)"
     from the synthetic baseline regardless of probe.

5. Probe/cache binding verified working, no leakage. renders_djent stamped
   d701edcef1ef6db9 ("real DI Djent DI.wav", 4356000 samples, levels -36/-24/-12),
   renders_funk stamped 4dbf52de67ae57c8. All four Djent tools printed the same
   digest. Loudest-pass clip started at t=60.5 s = 2*(30.00+0.25) as expected for
   a 30 s take.

6. Timing, for planning: rendering 34 profiles through a 30 s DI (90.75 s of
   audio) took 1026.1 s for Djent. The eight Suhr RL models dominate -- most
   profiles render in ~15-20 s, the Suhrs take 1-3 minutes each.