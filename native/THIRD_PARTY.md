# Third-party source notices

This local development build uses the following projects. Source and notices
are retained in `native/vendor`, including notices embedded in individual
source files. The portable binaries use the statically linked Microsoft C++
runtime and the Windows system runtime.

- **JUCE 8.0.15**, Raw Material Software Limited: JUCE license / AGPLv3,
  `native/vendor/JUCE/LICENSE.md`. Its modules include their own third-party
  notices (HarfBuzz, SheenBidi, image/audio codecs and other dependencies).
- **Neural Amp Modeler Core**, Steven Atkinson: MIT,
  `native/vendor/NeuralAmpModelerCore/LICENSE`.
- **AudioDSPTools**, Steven Atkinson: MIT. Its resampling container and WDL
  code preserve the iPlug 2 / Cockos WDL notices in the source headers.
- **Eigen**: primarily MPL 2.0, with compatible notices in `COPYING.*`.
- **JSON for Modern C++**, Niels Lohmann, copyright 2013–2025: MIT;
  `native/vendor/NeuralAmpModelerCore/Dependencies/nlohmann/json.hpp`.
- **VST3 SDK**, Steinberg Media Technologies GmbH: MIT, as retained in the
  bundled SDK's `LICENSE.txt` and individual files.
- **ASIO SDK headers**, Steinberg Media Technologies GmbH: the bundled
  `native/vendor/JUCE/modules/juce_audio_devices/native/asio/LICENSE.txt`.
- **Signalsmith Stretch** and **Signalsmith Linear**, Geraint Luff / Signalsmith
  Audio Ltd.: MIT; their `LICENSE.txt` files are included in the binary package.
  Exact revisions are recorded in `release/native_dependencies.json`.

TONE3000 capture photographs and NAM files are accessed from the user's
existing library/cache. Their source-page links and creator information are
shown in the UI; these assets are not included in the binary ZIP.

The Python analysis environment retains its existing dependency licenses.
It is not redistributed in this ZIP. `pluginval` is a separate GPLv3 validation
tool in the local tool cache and is not included in the app/plugin package.

## JSON for Modern C++ MIT notice

Copyright (c) 2013–2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
# Optional capture collection and Pitchproof reference

Nine cabinet-included NAM captures were added to the local development library
from `https://github.com/pelennor2170/NAM_models`, pinned in
`release/community_full_rigs.json`. The collection declares GPLv3; its COPYING,
README and source attribution remain under `.cache/community`. The NAM files are
not included by the application/source packaging scripts. Review creator and
distribution rights before shipping any capture pack.

Pitchproof's public specification informed the blend, fine-tuning and attack
controls. No Pitchproof binary, source code or artwork is included. Pitch DSP
continues to use the separately credited Signalsmith libraries.
