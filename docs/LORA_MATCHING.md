# LoRA matching in the native app

The Reference panel has a **Matching model** selector. Choose **LoRA (pilot)**
to use the trained MERT-95M adapter and its learned 128-value retrieval head,
then click **Find matching tones**. Choose **Standard** to return to the existing
MERT-330M matcher and its optional multi-DI projection.

Selection is saved with the standalone session and VST3 state. New instances
select LoRA when a local export is installed; sessions saved before this
control existed restore Standard. Switching clears the previous ranking and
takes effect on the next match. It does not change live NAM processing.
The selector is disabled while analysis runs. LoRA is unavailable in the menu
until installed; reopen the editor after installing it.

## Installed pilot

The local installation uses the reviewed final `mert_amp_lora_final_500pilot_v1`
export. The original and final LoRA exports contain identical trained weights,
so they represent one model, not two different choices.

- Base: `m-a-p/MERT-v1-95M`, revision
  `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`.
- Rank-8 query/value adapters in the last two used blocks, plus the learned head.
- Checkpoint SHA-256:
  `14d0ac2544e9d26c19a3e93028ab80a20dfa0789b6dd97c629754d96ee3cb0ec`.
- Portable weights: 3,476,248 bytes. The cached base encoder is also required.
- The app's existing 51 NAM captures have a separate 128-value LoRA index.
  Training on 500 rigs does not install those 500 playable captures in the app.

This is an experimental choice, not an accuracy upgrade claim. On the supplied
500-rig held-out clean-render test, LoRA top-5 was 58.0%, compared with 61.4% for
the same 95M frozen-projection baseline. The app indexes available captures
with its configured DI probe; that gallery and separated-song queries differ
from the pilot's multiple-performance gallery and clean queries. Those
percentages do not describe the app's commercial-song accuracy. See
[the pilot review](COLAB_500_PILOT_REVIEW.md).

## Install a trained LoRA export

From the configured project folder, using its Python environment:

```powershell
python engine/tools/install_lora.py D:/mert_amp_lora_final_500pilot_v1.zip --prepare-index
```

The installer also accepts a training run directory with `config.yaml` and
`checkpoint/best.safetensors`, or a compact directory with `config.yaml` and
`best.safetensors`. Reading a training config needs PyYAML from
`requirements-training.txt`. Normal inference uses JSON and does not need
the training dataset, optimizer state, or notebook environment.

Installation loads the pinned base encoder and checks the exact adapter/head
tensor names, shapes and finite weights before activating the new manifest.
The base downloads through the existing Hugging Face cache if it is not local.
An invalid export leaves the previous installation intact. The source ZIP or
directory is never changed. Only compact weights and a JSON manifest are
installed under `.cache/matching_models/lora`.

The runtime verifies the checkpoint hash and uses restricted base-weight
loading with no remote model Python. A missing or corrupt LoRA installation
reports an error with the option to reinstall or select Standard; it does not
silently label a Standard result as LoRA.

## Retrieval and caches

Both captures and song queries pass through the same adapter and head. Audio
is resampled to 24 kHz and normalized per analysis window, as in training.
Stereo channels are embedded independently before pooling so opposite-polarity
guitars do not cancel. Inference disables gradients; CUDA runs with autocast
and can fall back to float32 CPU after an allocation failure.

LoRA searches with ordinary cosine similarity, matching its training/evaluation
method. It does not use Standard's fitted projection or centering. Index
identities include checkpoint content, model configuration and inference
version. The separate index leaves Standard catalogue descriptors and
`.cache/tone_learning/active.npz` intact. Adding a capture embeds it on the next
LoRA match; changing weights rebuilds LoRA descriptors.

Worker results include the model name, base encoder, checkpoint hash, dimension
and experimental status. **About these matches** identifies LoRA results.
The source-configured development bundle still needs this project's Python
runtime, cached base weights, local LoRA installation and NAM library; it is
not a self-contained AI installer.

## Local validation, 17 September 2026

- Python engine suite: 465 passed, 40 optional-corpus skips. Source/release
  helper tests: 4 passed.
- Real pinned MERT-95M plus the exported checkpoint built 51 finite 128-value
  capture descriptors. Standard catalogue vectors remain separate.
- Native import, 40-second stereo song selection, LoRA matching, result cards
  and automatic playback-capture loading passed with the local Periphery
  reference. This is a workflow smoke test, not an accuracy measurement.
- Model selector, switching back to Standard, saved-state round-trip and old
  session defaults passed. Editor layouts were inspected at 1440 and 1280 widths.
- Standalone/VST3 Release builds and pluginval strictness 5 passed.

Local logs and screenshots are under `.cache/native/lora-*`; private audio,
weights, binary builds and caches remain outside Git.
