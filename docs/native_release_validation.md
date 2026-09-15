# Native release verification — 2026-09-09

Delivered Windows x64 artifacts: `dist/ToneHound/ToneHound.exe`,
`dist/ToneHound/ToneHound.vst3`, and `dist/ToneHound-windows-x64.zip`.

| Check | Result |
| --- | --- |
| Python engine suite | 586 passed; `test_suite_native.txt` |
| New native bridge boundary tests | 13 passed, including full-length import, URL provenance, minimum crop, invalid times, errors and Unicode JSON |
| VST3 host validation | pluginval strictness 5: SUCCESS, including editor, processing, automation, state and bus layouts; `pluginval-native.txt` |
| Native NAM processor | Finite stereo output with 1/64/127/256/512/1024-frame blocks at 44.1/48/96 kHz; monitor mute and host parameter/crop restoration pass |
| Invalid NAM path | Reports load failure and disables the model path |
| Native reference controls | Actual Python import job, time-field minimum bounds, preview sound, play and stop pass |
| Worker ownership | Cancellation completes and a subsequent library job succeeds |
| Physical audio driver | Komplete Audio ASIO Driver opened at 48 kHz / 256 frames; 224 actual callbacks in 1.2 seconds, with monitor off |
| Pipeline through file-job interface | Imported the local Periphery MP3, matched a 40-second excerpt, isolated guitar successfully, returned six actual NAM captures |
| Complete native matching workflow | Actual Find matching tones button completed the Python job, displayed six result cards, and loaded the first result's NAM for playing |
| Packaged standalone launch | `dist/ToneHound/ToneHound.exe` launched and its native window responded |
| Visual inspection | Actual JUCE component exports in `native-ui.png`, `native-io.png`, and `native-matches.png` |

`native_validation.json` reports processing timing for the first Marshall
capture on this computer. It used approximately 18–21% of the real-time budget
across measured runs. This is a single-capture measurement, not a library-wide
performance guarantee. Added resampler latency was 27 samples at 44.1 kHz,
zero at the capture's 48 kHz rate, and 42 samples at 96 kHz. Interface and host
buffer latency are additional.

The user's Djent DI was rendered by the actual native processor, with gate
off, unity input/output and flat EQ. Against the existing Python float64 NAM
renderer at 48 kHz, after discarding 4,096 warmup samples, the comparison over
715,904 samples had maximum absolute error `6.583e-6`, RMS error `2.137e-7`,
and correlation `0.9999999999998392`. Native output was written as 24-bit WAV.
See `native_di_comparison.json` and `native-di-48k.wav`. A separate 44.1 kHz
native render exercises the sample-rate converter (`native-di.wav`).

This validates the signal path and audio driver callbacks. No claim is made
that a physical guitar performance or acoustic round-trip latency was
measured. The standalone begins with monitoring off; choose the interface
and guitar channel, then enable MONITOR to play.

The package uses the project's existing local Python/CUDA installation and
asset caches. It does not bundle that environment for installation on another
computer. The VST3 is an effect with host-owned audio I/O; reference preview
is independent of host transport. Retrieval accuracy is unchanged.
