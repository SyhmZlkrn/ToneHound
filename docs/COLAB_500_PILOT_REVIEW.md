# ToneHound: review of the 500-rig Colab pilot

The exported LoRA checkpoint is the strongest of the three trained neural
retrieval heads on validation, but it does **not** beat its frozen-MERT
projection baseline on the held-out test. Keep the current desktop matcher
active while the next comparison is prepared.

This review reads the four ZIP files supplied on `D:`. It checks their
contents and recomputes the reported statistics where the saved data allows
it; it does not claim to have rerun the full audio evaluation locally.

## What was trained

All three runs used the same dataset fingerprint, 500 full-rig NAM captures,
12 DI recordings, five-second windows and one input level. There were 4,000
training examples, 1,000 validation examples and 1,000 test examples.

Eight training files represent six performance groups because related funk
and power-chord takes were grouped. Validation uses Thall and neck single coil.
The test uses Drop E and solo. Each performance group stays within one split.
Every run completed 20 epochs and 2,500 optimizer steps, and selected epoch 20.

Here, **top-1** means the exact NAM capture was the first result. **Top-5** means
it appeared somewhere in the first five results. This measures retrieval of
known captures from rendered guitar audio, rather than identification of the
physical amplifier used on a commercial recording.

## Validation results

| Method | Top-1 | Top-5 |
| --- | ---: | ---: |
| Frozen 330M MERT + existing-style fitted projection | **74.5%** | **87.3%** |
| 95M MERT + LoRA + learned head | 61.9% | 81.8% |
| Frozen 95M MERT + fitted projection | 62.8% | 80.8% |
| Frozen 330M MERT + learned head (`mert_amp_v1`) | 40.9% | 63.6% |
| 95M MERT, last two blocks unfrozen + learned head | 23.7% | 43.9% |

“Frozen” has two meanings in the files. `mert_amp_v1` trains a small neural
head on fixed MERT features. Its `baseline.json` instead evaluates the fitted
covariance projection used as the reference method. That reference performs
substantially better than the learned head in this pilot.

The 330M and 95M experiments also use different pretrained checkpoints and
feature layers. Their scores compare complete configurations; they do not
isolate the effect of LoRA alone. A 95M frozen-head control would make the next
fine-tuning comparison more controlled.

## Held-out results in the supplied files

The three initial ZIPs say `trained; test not evaluated`. Only
`mert_amp_lora_final_500pilot_v1.zip` includes `retrieval_test.json`, confusion
counts and final test metrics. Its LoRA weights are byte-for-byte identical
to those in `mert_amp_lora.zip`.

| Method on the same 95M encoder and test set | Top-1 | Top-5 | MRR |
| --- | ---: | ---: | ---: |
| Raw frozen MERT, multiple-DI reference gallery | 37.0% | 53.1% | 0.4556 |
| Frozen MERT + fitted projection | **45.5%** | **61.4%** | **0.5358** |
| LoRA + learned head | 40.8% | 58.0% | 0.4949 |

LoRA loses **3.4 percentage points** of top-5 accuracy to the fitted
projection: 580 rather than 614 successful queries out of 1,000. It improves
on the raw frozen embeddings, but the fitted projection is the stronger
available comparator. MRR rewards placing the correct capture nearer the top.

The test archive does **not** contain the 330M projection comparison or held-out
scores for the other trained heads. A 330M test score cannot be inferred from
its 87.3% validation result. The 61.4% test baseline above is the **95M** model,
not a measured score for the currently deployed 51-capture desktop catalogue.

### The two test recordings behave very differently

| LoRA test performance | Queries | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| Drop E | 500 | 67.8% | **87.0%** |
| Solo | 500 | 13.8% | **29.0%** |

The solo take accounts for much of the generalization weakness. This suggests
prioritizing a broader range of independent lead performances in the next
training collection. It does not establish that every solo is difficult or
that every rhythm part is easy. Although there are 1,000 queries, they come
from only **two** held-out performances rendered through 500 rigs.

Per-recording frozen-baseline rankings were not exported, so a per-recording
LoRA-versus-baseline comparison is unavailable.

## Integrity checks

- All four archives passed CRC checks. All runs name the same dataset hash.
- The original source manifest reproduces all 1,000 held-out query IDs and
  their NAM/DI identities. All 500 expected captures occur in the test.
- Saved split groups are disjoint. Every baseline cache leaves test feature
  rows empty, consistent with training that did not embed the test recordings.
- All best-checkpoint tensors are finite. Parameter counts match the run
  metadata, and `.pt` and `.safetensors` best weights agree exactly.
- Checkpoint/config/dataset fingerprints, selected epochs and completed-step
  counts agree. `.pt` files were read using PyTorch's restricted
  `weights_only=True` loader.
- The saved frozen-projection validation scores were independently recomputed
  from their cached features. LoRA test top-1, top-5, MRR and confusion counts
  were recomputed from its 1,000 exported query rows.
- The LoRA test evaluation key matches the selected checkpoint with no 330M
  reference run. The active desktop model remains unchanged.
- The final LoRA checkpoint loaded successfully with its pinned, locally
  cached MERT-95M backbone on the RTX 2060 SUPER. A five-second training-DI
  forward pass produced a finite, unit-normalized 128-value embedding. This
  checks local compatibility, not retrieval accuracy; details are in
  `.cache/colab_500_review/local_checkpoint_smoke.json`.

These are internal-consistency checks, not proof of every action taken during
the cloud session. The partial-unfreeze run used Python 3.12/PyTorch 2.10; the
other runs used Python 3.13/PyTorch 2.11. Use one runtime version in the next
controlled comparison.

## What to do next

1. Finish the already-planned 330M baseline comparison using this fixed pilot
   dataset. Evaluate the original `mert_amp_v1` run separately; retain the
   existing LoRA final report unchanged. Compare the resulting frozen
   projection score before considering an application model change.
2. Prepare a new experiment with more independent lead and rhythm recordings,
   including varied picking, sustained notes, bends and legato. Keep related
   edits together. Reserve new performances for the next final test; these
   Drop E/solo results have now been examined and should not guide repeated
   tuning against the same test set. Keep the 500-rig pilot fixed initially so
   the effect of broader DI coverage can be measured.
3. Add known-NAM mixed-song renders and htdemucs-separated versions to a
   separate, controlled experiment, retaining disjoint backing-song and DI
   groups. These four exports contain a clean-render evaluation; they do not
   establish accuracy on separated commercial songs. The older 51-capture
   Demucs scores are a different benchmark.

All three best epochs are the final epoch. That leaves longer training as an
experiment to evaluate on validation; it is not evidence that extra epochs
will fix the solo weakness. Harder negative captures and a 95M frozen-head
control are also candidates for that next validation-only study.

## Storage and local artifacts

The validation-curve chart remains in the private local review folder described below.

The LoRA portable checkpoint is **3,476,248 bytes (3.48 MB)**. Its roughly
74 MB ZIP also holds frozen feature caches and optimizer/checkpoint copies.
The partial-unfreeze portable checkpoint is about 60 MB. These experiment
ZIPs should not be bundled wholesale into the application. LoRA still
requires its pinned MERT base model and an appropriate reference index.

Original archives remain at their supplied `D:` paths. The compact local audit
is `.cache/colab_500_review/audit.json`; the chart is
`.cache/colab_500_review/learning_curves.png`. Only the final LoRA config and
portable checkpoint were copied into that ignored review folder. No weights,
NAM files, DI audio or experiment archives were added to Git.

Dataset fingerprint:
`74f1fd8c2e59c0bdedb799a81f5d80cbbfd4c23f295d2d767f9a63f09d9dcfcc`

Final LoRA portable checkpoint SHA-256:
`14d0ac2544e9d26c19a3e93028ab80a20dfa0789b6dd97c629754d96ee3cb0ec`
