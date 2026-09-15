# Offline GPU rendering

NAM rendering now supports opt-in CUDA for WaveNet and Linear captures. It keeps
float64 arithmetic, and the existing CPU path remains the default.

## Use it

Add `--render-device cuda` (or `auto`) to the song CLI. This applies to both a
cold index build and the matched-profile auditions:

```powershell
$env:PYTHONPATH = 'engine'
$env:PATH = "$PWD\.cache\bin;$env:PATH"
python engine/tools/match_song.py --song "assets/song_train/PERIPHERY - The Bad Thing.mp3" --profiles .cache/tone3000/profiles --start 40 --seconds 20 --top 6 --fp16 --render-device cuda
```

`--device` still selects MERT's device. `--fp16` still applies to MERT and
separation; NAM rendering stays float64. `cuda:0` also selects a specific GPU.

The Python entry points are:

```python
model = nam_render.load(path)
audio = model.render(di, device="cuda")

index = ToneIndex.load_or_build(entries, di, cache_path,
                               embedder=embedder, render_device="cuda")
cfg = PipelineConfig(render_device="cuda")
```

Existing float64 index caches remain reusable when switching render devices.
A warm index has no profiles to render, so this improves cold builds, additions
to the library, and auditions.

## Memory and fallback

Each CUDA chunk produces 65,536 output samples and includes the model's complete
receptive-field history. This preserves the intermediate bias and activation
history across chunk boundaries, including the initial zero padding. The GPU
graph is a separate copy; offline rendering never moves or resets live streaming
state. `NamModel.render(..., chunk_samples=...)` can adjust the chunk size.

An unavailable CUDA device or CUDA runtime failure, including out-of-memory,
falls back to a complete CPU float64 render and emits a warning. LSTM retains
its existing CPU recurrence and reports that choice. `last_render_device` and
`last_render_note` describe what actually happened on each loaded model.

Use the default sequential render map with CUDA; multiple render worker
processes compete with each other and MERT for GPU memory.

## Verification

The reference parity tests run both CPU and CUDA for WaveNet/Linear and retain
their original tolerances, including the **1e-7** legacy-weight check. Additional
tests cover padded/unpadded chunk boundaries, per-channel activation weights,
fallback, cache reuse, pipeline wiring, and isolation from streaming state.

```powershell
$env:PYTHONPATH = 'engine'
$env:HF_HUB_OFFLINE = '1' # reuse the installed checkpoint
python -m pytest engine/tests/test_nam_render_parity.py -q
python -m pytest engine/tests -q
python engine/tools/benchmark_nam_render.py
```

The benchmark reads local profiles and DI only. It includes model loading and
both device transfers, warms the CUDA context first, and compares raw float64
audio plus the normalised float32 audio sent to MERT. Run it without concurrent
tests or other indexing work for representative timings.

Measured on 2026-09-09, RTX 2060 SUPER, PyTorch 2.11.0+cu128, eight CPU threads,
40 downloaded profiles, and 15 seconds of the user's Djent DI:

| Render measurement | Result |
| --- | ---: |
| CPU float64, all 40 profiles | 99.64 s |
| CUDA float64, all 40 profiles | 25.89 s |
| Speedup | 3.85x |
| Largest raw sample difference | 2.80e-13 |
| Largest normalised float32 difference | 5.82e-11 |
| Peak CUDA tensor allocation, renderer alone | 69.17 MiB |

All 40 captures used CUDA without fallback. The full test suite passed all
573 tests in 108.20 seconds; `test_suite_gpu_final.txt` records the run.

Detailed results are in `nam_gpu_benchmark.json` and `nam_gpu_index_validation.json`.
With MERT fp16 already loaded and warmed, a separate cold index build rendered
and embedded all 40 profiles in **20.11 s**. Cached reload took **0.018 s**.
Every embedding vector was exactly equal to the existing CPU-rendered cache;
the top 10 for `docs/match_song/01_stem.mp3` were identical. Peak CUDA tensor
allocation with MERT was **894.48 MiB**. This separate run includes embedding
but excludes initial checkpoint loading, so its timing should not be added to
or directly subtracted from the standalone rendering measurement above.

These measure rendering consistency and indexing cost, not retrieval accuracy
on new songs; the accuracy limitations in `HANDOFF.md` still apply.

The full-suite check also exposed an existing deadlock in the HTTP test harness:
some synchronous requests were made while its server event loop was stopped.
Those test requests now run on a worker while the loop runs, have a timeout,
and the missing-crop test waits for index readiness. Server production code was
not changed.
