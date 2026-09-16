# ToneHound: training MERT in Colab or Kaggle

The training code is portable Python. GitHub holds the code; your private storage
holds NAM captures, DI recordings, rendered examples and model checkpoints. The
desktop app does not need to contain the training dataset.

[Open the Colab notebook](https://colab.research.google.com/github/SyhmZlkrn/ToneHound/blob/main/notebooks/ToneHound_Colab.ipynb)
or upload `notebooks/ToneHound_Kaggle.ipynb` to a private Kaggle notebook.

## What the model learns

We run several different clean guitar performances through each full-rig NAM.
The known label is **the capture that produced the sound**. Training brings
different performances through that capture closer together, and separates
different captures. This is supervised contrastive retrieval, not a classifier
that guesses a historical studio amp from a song title.

A full rig includes the cabinet response. Amp-head-only captures are excluded
from this dataset. Captures can differ in amp, cabinet, microphone, gain and
pedals; a match identifies a similar complete capture, not necessarily the
physical amp used on a record. Very similar captures make exact top-1 difficult.

## The initial 500-capture pilot

The local collection supplied in September 2026 contains 4,049 CSV rows and
4,044 files. Five CSV entries have no file. Most files use NAM A2 containers.
The preparation tool reads embedded cabinet metadata, removes duplicate model
content/alternative architecture copies, excludes explicitly bass-oriented
captures, spreads selection across gain categories, creators and tone packs,
and actually renders every selected model before packaging it.

The resulting `pilot_report.json` records the exact selection. Cabinet and
gear descriptions are creator declarations, not independent measurements.
Mixed-gain packs remain `unknown` for the optional gain-class metric; a pack
containing both clean and distorted captures must not label all of them clean.

The first pilot uses 500 captures and 12 DI files, split 8 training / 2 validation
/ 2 test. Related funk and power-chord variants are conservatively grouped in
training. Drop E and solo are the fresh held-out test recordings. A filename
alone cannot prove independence; keep alternate versions of one performance in
the same `performance_group`. Metadata/container duplicates and decoded audio
duplicates are also checked before rendering.

One five-second window per capture/performance gives **6,000 examples**, about
**1.44 GB of PCM audio** before filesystem overhead. This is training storage,
not an application download. More windows and input-gain variations multiply
that size. At 500 captures and 20 DIs, one window gives 10,000 examples / 2.4 GB.

## Folder layout

Put the private source package in your Drive, not the public repository:

```text
MyDrive/
  ToneHound-pilot-source.zip
  ToneHound-ML/
    source/
      manifest.json
      nam/*.nam
      di/*.wav
    datasets/amp-tone-v1/
      train/*.wav
      validation/*.wav
      test/*.wav
      metadata.json
    experiments/
      mert_amp_v1/
      mert_amp_lora/
      mert_amp_unfreeze_last/
```

The Colab notebook can unpack `ToneHound-pilot-source.zip` from MyDrive. The
source ZIP retains creator names, model IDs, source links and the stated license.
It is a private input bundle, not a redistributable tone pack. A CSV value such
as `t3k` is not a grant to publish the captures or learned models commercially.
See [TONE3000's sharing policy](https://www.tone3000.com/guides/tone-sharing-guidelines)
and [API terms](https://www.tone3000.com/api/terms) for distribution constraints.
No TONE3000 API calls are needed for this local collection.

## Colab steps

1. Upload the prepared source ZIP to MyDrive using your own Google account.
2. Open the linked notebook, choose a GPU runtime, and mount Drive when prompted.
3. Fetch the source and install `requirements-training.txt`. The notebook keeps
   a compatible PyTorch already supplied by the runtime. Restart if the package
   installer requests it, then rerun setup. Python 3.10 or newer is required.
4. Unpack the source ZIP, build the pinned NAM Core renderer and run the dataset
   validation/size-plan cell. The A2 renderer selects full quality (`--slim 1.0`).
5. Render the dataset. Rerunning this cell resumes the same manifest; changing
   its contents requires a new dataset directory. Only one NAM/performance is
   rendered at a time. Temporary input/output WAVs are removed after each render.
6. Run a small smoke experiment, then use a new experiment name and set
   `SMOKE_RUN = False` for real training. The smoke setting makes one optimizer
   step, but baseline extraction still reads the entire training/validation set.
7. Compare completed runs using validation results. Run the held-out test only
   after selecting the final configuration.

Colab availability, GPU model and session duration vary; the notebook does not
buy compute or assume a particular GPU. Checkpoints are stored on Drive.
Dataset reads can be faster from local `/content` storage; optionally copy the
prepared dataset there and change `DATASET`, while keeping `RUN` on Drive.
See [Colab's runtime and Drive guidance](https://research.google.com/colaboratory/faq.html).

## Commands outside the notebook

These commands work on Linux, Windows, Colab and Kaggle with appropriate paths.
The C++ renderer requires Git, CMake 3.24+, a C++20 compiler, and network access
for the pinned NAM Core dependency/submodules. It does not require JUCE.
On Windows, run the build from an x64 Visual Studio developer shell.

```bash
python scripts/setup_training_renderer.py
python engine/training/build_dataset.py --manifest /data/source/manifest.json --output /data/datasets/amp-tone-v1 --plan
python engine/training/build_dataset.py --manifest /data/source/manifest.json --output /data/datasets/amp-tone-v1 --renderer .cache/training_renderer/nam-render
python engine/training/train_mert.py --config configs/mert_amp_v1.yaml --dataset /data/datasets/amp-tone-v1 --output /data/experiments/mert_amp_v1
```

Use `nam-render.exe` on Windows. Add `--resume` to the identical training command
after interruption. The dataset/config hashes must match. Resume restores the
optimizer, gradient scaler, random generators and next batch at the last saved
optimizer boundary. Work after that boundary can be replayed. Do not resume a
different experiment into the same folder. Use `--device cpu` deliberately for
CPU debugging; a missing requested GPU produces an error rather than silently
starting a very long CPU training run.

For another local collection, copy `configs/dataset_manifest.example.json` and
fill in real paths, rig evidence and training-use authorization. The CLI
`prepare_collection.py --help` prepares a pilot from the supplied CSV format and
an explicit DI split file. Source files are copied into a separate output folder;
the original collection is not modified.

## Three experiments

| Config | Encoder | What trains |
| --- | --- | --- |
| `mert_amp_v1.yaml` | Existing GTZAN-derived MERT-330M, layers 8â€“11 | Small contrastive head; backbone frozen |
| `mert_amp_lora.yaml` | Original MERT-95M, layers 9â€“12 | Rank-8 query/value adapters in blocks 11â€“12 and head |
| `mert_amp_unfreeze_last.yaml` | Original MERT-95M, layers 9â€“12 | Complete blocks 11â€“12 and head |

Every experiment also fits a frozen covariance-projection baseline using its
own pinned encoder. Trainable blocks are the last **used** blocks; updating
layers later than the outputs being compared would have no learning signal.
Two distinct performance groups per capture are required in each training batch.
Gradient accumulation changes optimizer batch size, but does not create extra
contrastive negatives across microbatches.

The first dataset uses clean rendered guitar to isolate this change. Input gain
and additional windows can be added explicitly to a subsequent manifest. The
existing `engine/tools/train_retrieval.py --with-song-separation` experiment still
provides the earlier htdemucs controlled-remix benchmark; it is not silently
mixed into this clean pilot. A new separated-song benchmark must reserve new
backing songs and DI performances. Song files alone are not trustworthy labels
for exact studio amps. The earlier 51-capture test has already been examined;
its percentages must not be compared directly with a new 500-capture test.

## Selecting and evaluating

Run each planned experiment into its own folder on the same dataset. Then:

```bash
python engine/training/select_experiment.py --runs /data/experiments/mert_amp_v1 /data/experiments/mert_amp_lora /data/experiments/mert_amp_unfreeze_last --output /data/experiments/selection.json
python engine/training/evaluate_mert.py --run /data/experiments/SELECTED_RUN --dataset /data/datasets/amp-tone-v1 --reference-run /data/experiments/mert_amp_v1
```

Replace `SELECTED_RUN` with the selection result. Selection uses validation
top-5, then MRR; the test is never used to choose epochs or model settings. The
optional reference run compares against the current 330M projection method
refitted on exactly this dataset. It does not reuse the old 51-capture gallery.
The existing desktop `active.npz` remains untouched as an additional local
baseline. Testing a checkpoint does not authorize automatic activation.

Reports include exact-capture top-1/top-5, mean reciprocal rank, optional
family/gain label agreement, per-query rankings and confusion counts. Missing
gear labels are reported as missing. Failed queries count as misses. Training
fails visibly on corrupt/silent files instead of dropping hard examples.

## What comes back from a run

```text
config.yaml
checkpoint/last.pt              # resumable optimizer state + trained tensors
checkpoint/best.pt              # validation-selected trained tensors
checkpoint/best.safetensors     # portable trained tensors, no optimizer
baseline.npz                   # train/validation frozen vectors + projection
baseline.json
training_history.json
run_info.json                  # Git revision, versions, GPU, split, hashes
metrics.json
retrieval_test.json             # written only by final evaluation
confusion.json
```

Retain the entire folder. Adapter/head files require the pinned base encoder;
they are **not** drop-in replacements for the app's `active.npz`. A trained
backbone needs a corresponding inference loader and a rebuilt catalogue before
it can be deployed. This workflow creates reviewable candidates; native-app
integration follows evaluation of the real pilot, not the smoke test.

For Kaggle, attach the prepared dataset privately at
`/kaggle/input/<slug>/datasets/amp-tone-v1`. Outputs go to
`/kaggle/working/experiments`. Save/download outputs before ending the session.
To resume in a new session, copy the previous complete run folder into the new
writable output location and run the same command with `--resume`.

## Implementation and evidence

The pinned MERT-95M and 330M configurations use the stock HuBERT architecture:
CQT and DeepNorm are disabled. Every encoder weight must load exactly; model
Hub Python code is never executed. LoRA uses the standard frozen linear layer
plus low-rank update, initialized to make no change to the base layer.

References: [MERT-95M model/config](https://huggingface.co/m-a-p/MERT-v1-95M),
[supervised contrastive learning](https://arxiv.org/abs/2004.11362),
[LoRA formulation](https://huggingface.co/docs/peft/en/developer_guides/lora),
[NAM Core](https://github.com/sdatkinson/NeuralAmpModelerCore).
The MERT checkpoint license is CC BY-NC 4.0; this workflow does not change it.

Local validation exercises the actual optimizer in all three modes, checks
which parameters change, tests split leakage/checksums and interruption/resume,
and verifies that training does not read held-out test waveforms. A GPU smoke
run also used the real pinned 330M/95M encoders, three creator-labelled full rigs
and four real DI recordings. One optimizer step validates wiring, not accuracy.

Validation on 16 September 2026: **455 Python tests passed, 40 optional corpus
checks skipped**; 19 focused training/release checks passed. The private
500-profile pilot contains 493 A2 and 7 WaveNet captures from 230 creators and
395 tone packs. All selected captures passed a native probe render; one failing
candidate was replaced and recorded in the private audit report.
