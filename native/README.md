# ToneHound native app and VST3

ToneHound uses a JUCE desktop editor and the Neural Amp Modeler C++ engine. The
standalone executable and VST3 share the same controls and signal processing.
No browser, WebView, localhost server, or Python audio callback is involved.

## Play guitar

1. Open `dist/ToneHound/ToneHound.exe`.
2. Open **I / O**, select your audio interface and its input/output channels.
   On Windows, use the interface's **ASIO** driver when available. Start with
   48 kHz and a 256-sample buffer; increase the buffer if audio breaks up.
3. Select **Input 1** or **Input 2** in the header to match your guitar socket.
4. Choose a capture, or use **Load .nam**. Turn **MONITOR ON** and play.

Monitoring starts off in the standalone app. Output starts at -6 dB. Input
adjusts the level entering the capture; Gate removes quiet input noise;
Bass/Mid/Treble are additional EQ after the fixed NAM capture. Double-click a
knob to reset it. Bypass passes the guitar through without the capture or EQ.
The selected mono guitar input feeds both outputs. Other interface channels
are not mixed into the guitar signal.

The amplifier stage displays the complete photograph supplied on its TONE3000
page at its original aspect ratio, without cropping or stretching. Local
captures without a linked Tone ID display a NAM placeholder.
The capture's own amp/cab content determines its sound; ToneHound does not add an
unrelated cabinet simulation to an amp-only capture.

Use **Library** to search TONE3000 and add captures beyond the initial 40.
After matching, **EQ match** records ten seconds of your played guitar and
applies a bounded frequency correction. **Pedals** provides Screamer, stereo
delay and reverb. Choosing **Solo** applies delay/reverb automatically;
**Rhythm** disables them. The estimated BPM and every pedal remain adjustable.
See `STUDIO_FEATURES.md` in the package for the complete procedure.

The EQ has eight editable faders (63 Hz–8 kHz); selecting another amp resets
these and Bass/Mid/Treble. Pedals have graphical knobs and footswitches.
The header adds −12…+12 semitone pitch transpose. Live mode adds 40 ms of pitch
latency and may smear low chords; Studio mode improves chord separation with
192 ms latency for recorded guitar. **Pitch settings** adds wet/dry harmony,
±50-cent fine tuning and pick-attack preservation. Zero semitones and zero cents
bypasses pitch with no added pitch latency.

The Library tab imports/exports `.tonelib` descriptors. Missing catalogue NAMs
are downloaded when selected; managed downloads use a 32 MiB cache target.
Local imports and the pre-existing library are preserved. See
`CATALOGUE_AND_MERT.md` for storage details and `MULTI_DI_TRAINING.md` for the
implemented multi-DI and song-separation comparison training.

## Reference song and crop

Use **FILE** to import MP3/WAV or drag one onto the editor. Use **LINK** to
import a song URL supported by the existing yt-dlp engine. Downloads and
decoding happen in a separate process, with progress and cancellation.

New imports preserve stereo for reference playback and Demucs separation.
MERT analyses the separated channels independently and pools their descriptions.
The reference label shows Stereo or Mono. **Re-import songs loaded before the
stereo fix from their original file or URL**, then run matching again: the old
decoded mono WAV cannot recover its lost channels. Original downloads and the
capture index can be reused.

The initial selection is 40 seconds (or the full track for 30–40-second
tracks). Drag either waveform handle to resize, drag the selected region to
move it, or enter `mm:ss.s` / seconds in Start and End. A selection cannot be
shorter than 30 seconds or extend past the track. **Play selection** auditions
the reference, **Loop** repeats it, and guitar monitoring can play alongside.

**Find matching tones** isolates and compares the selected passage using the
existing GPU-enabled pipeline. Click a result to load its actual NAM file and
play your guitar through it. Scores describe similarity within this library;
the shortlist does not identify the original recording rig. The UI reports
whether separation used the guitar, another stem, or the full mix.
**About these matches** opens the pipeline's full measured accuracy caveat.

## VST3

`dist/ToneHound/ToneHound.vst3` is the complete Windows x64 plugin bundle. Add
`dist/ToneHound` to your DAW's VST3 scan paths, or copy the **entire** `.vst3`
folder into a VST3 location your DAW scans. Its internal `.vst3` binary must
remain inside `Contents/x86_64-win` with the accompanying metadata.

Insert ToneHound on a guitar audio track and enable that track's input monitoring.
The DAW owns the interface, sample rate, buffer and track input routing.
ToneHound's I/O panel explains this; its Input 1/2 selector chooses between the
channels the DAW sends to the plugin. Parameters, selected capture, reference
path and crop are stored with the host session. Reference playback is an
audition transport; it is not synchronized to the DAW timeline.

## Local engine and assets

This development build uses this project's existing Python installation,
profiles, model weights, caches and TONE3000 connection. Keep the project in
place. It is not yet an installer that bundles the Python/CUDA environment
for another computer. Live NAM guitar processing runs entirely in C++.

The app finds the project by walking up from its executable, then uses the
build-time project path. Set `TONEHOUND_ROOT` if the project is relocated and the
VST3 is installed elsewhere. `.cache/native/runtime.json` contains the Python
executable path. Requests, progress, responses and diagnostic logs are stored
in per-job directories under `.cache/native/jobs`. Artwork and decoded audio
are cached under `.cache/native`.

The standalone saves device configuration and session state in
`%APPDATA%/ToneHound`. Closing an instance terminates its own analysis workers
and their child processes; other instances' jobs are independent.

## Build and verify

Dependencies are checked out under `native/vendor`:

| Dependency | Revision |
| --- | --- |
| JUCE 8.0.15 | `91ad83ae34a81e0833b1a2b0866f54846370ae53` |
| NeuralAmpModelerCore | `2563c0fd4cb1f9ce457d89a761738ea15097e1f3` |
| AudioDSPTools submodule | `0827c6c2fc0deced568536142ea86f189e0b98a1` |

`native/build.ps1` uses the local MSVC 14.44 x64 compiler, Windows SDK
10.0.22621, CMake and Ninja under `.cache/toolchain`. Microsoft toolchain
packages were extracted from their official Visual Studio 2022 download
manifests using the inspected `portable-msvc.py` helper in that directory.
JUCE 8.0.15 does not support MinGW. The unused initial MinGW attempt is not
part of the build.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File native/build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File native/package.ps1
native/build-msvc/ToneHound_Validate_artefacts/Release/ToneHound-Validate.exe
```

With an existing Microsoft C++ build environment, ordinary CMake also works:

```powershell
cmake -S native -B native/build-vs -G "Visual Studio 17 2022" -A x64
cmake --build native/build-vs --config Release
```

The validator exercises the real native audio processor, variable block
sizes, sample-rate conversion, stereo routing, monitoring and session state.
It also accepts `--devices` to enumerate audio drivers and
`--snapshot ABSOLUTE_PNG_PATH --reference ABSOLUTE_AUDIO_PATH` to export the
real JUCE editor for visual QA. Python boundary tests live in
`engine/tests/test_native_bridge.py`.

Current studio validation: **411 Python tests passed, 40 skipped** because the
optional development capture corpus is absent. Protocol tests generate their
own fixtures. The native VST3 passed **pluginval strictness 5**, including editor,
automation, state and audio tests. EQ/pedal DSP, ten-second capture through a
real NAM, EQ/manual-pedal state restoration, stereo reference playback and
automatic solo effects passed. See `docs/STUDIO_VALIDATION.md` for evidence.

Earlier hardware validation: the Komplete Audio ASIO driver ran at 48 kHz / 256 samples with
224 callbacks over 1.2 seconds (monitoring off). Native reference import,
play/stop, crop bounds and worker cancellation were exercised. The native
render of the user's DI agrees with the Python renderer to a maximum absolute
sample difference of `6.59e-6` after warmup. Reports are in `docs/`.

For a detailed plain-language explanation of the import, separation, matching
and live guitar paths, see `docs/HOW_TONEHOUND_WORKS.md` in the project, also
included as `HOW_TONEHOUND_WORKS.md` in the packaged build.

## Source and asset notices

JUCE's license is retained in `vendor/JUCE/LICENSE.md`; NAM's MIT license is
in `vendor/NeuralAmpModelerCore/LICENSE`. The NAM submodules retain Eigen,
AudioDSPTools, iPlug/WDL and JSON notices. JUCE contains the VST3 SDK and ASIO
headers with their own notices. `dist/ToneHound/ThirdParty` contains the primary
licenses for this local build. TONE3000 photographs and capture files retain
their creators' attribution and source-page links. They are referenced from
the existing cache and are not copied into the binary bundle.
