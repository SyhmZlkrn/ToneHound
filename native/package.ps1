param([string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$artifactRoot = Join-Path $PSScriptRoot 'build-msvc'
$destination = Join-Path $projectRoot 'dist/ToneHound'
$executable = Join-Path $artifactRoot "ToneHound_Standalone_artefacts/$Configuration/ToneHound.exe"
$plugin = Join-Path $artifactRoot "ToneHound_artefacts/$Configuration/VST3/ToneHound.vst3"
if (!(Test-Path -LiteralPath $executable) -or !(Test-Path -LiteralPath $plugin)) { throw 'Build the standalone and VST3 targets first.' }
New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath $executable -Destination "$destination/ToneHound.exe" -Force
# Copy bundle contents explicitly; repeated packaging must not nest bundles.
New-Item -ItemType Directory -Path "$destination/ToneHound.vst3" -Force | Out-Null
Copy-Item -LiteralPath "$plugin/Contents" -Destination "$destination/ToneHound.vst3" -Recurse -Force
Copy-Item -LiteralPath "$PSScriptRoot/README.md" -Destination "$destination/README.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/HOW_TONEHOUND_WORKS.md" -Destination "$destination/HOW_TONEHOUND_WORKS.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/stereo_matching_regression.json" -Destination "$destination/stereo_matching_regression.json" -Force
Copy-Item -LiteralPath "$projectRoot/docs/STUDIO_FEATURES.md" -Destination "$destination/STUDIO_FEATURES.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/CATALOGUE_AND_MERT.md" -Destination "$destination/CATALOGUE_AND_MERT.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/MULTI_DI_TRAINING.md" -Destination "$destination/MULTI_DI_TRAINING.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/retrieval_evaluation.json" -Destination "$destination/retrieval_evaluation.json" -Force
Copy-Item -LiteralPath "$projectRoot/docs/STUDIO_VALIDATION.md" -Destination "$destination/STUDIO_VALIDATION.md" -Force
Copy-Item -LiteralPath "$projectRoot/docs/RELEASE_AND_STORAGE.md" -Destination "$destination/RELEASE_AND_STORAGE.md" -Force
New-Item -ItemType Directory -Path "$destination/ThirdParty" -Force | Out-Null
Copy-Item -LiteralPath "$PSScriptRoot/vendor/JUCE/LICENSE.md" -Destination "$destination/ThirdParty/JUCE-LICENSE.md" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/NeuralAmpModelerCore/LICENSE" -Destination "$destination/ThirdParty/NAM-LICENSE.txt" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/JUCE/modules/juce_audio_devices/native/asio/LICENSE.txt" -Destination "$destination/ThirdParty/ASIO-LICENSE.txt" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/JUCE/modules/juce_audio_processors_headless/format_types/VST3_SDK/LICENSE.txt" -Destination "$destination/ThirdParty/VST3-LICENSE.txt" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/NeuralAmpModelerCore/Dependencies/AudioDSPTools/LICENSE" -Destination "$destination/ThirdParty/AudioDSPTools-LICENSE.txt" -Force
New-Item -ItemType Directory -Path "$destination/ThirdParty/Eigen" -Force | Out-Null
Get-ChildItem -LiteralPath "$PSScriptRoot/vendor/NeuralAmpModelerCore/Dependencies/eigen" -Filter 'COPYING*' | Copy-Item -Destination "$destination/ThirdParty/Eigen" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/NeuralAmpModelerCore/Dependencies/AudioDSPTools/dsp/ResamplingContainer/ResamplingContainer.h" -Destination "$destination/ThirdParty/iPlug-Resampler-notice.h" -Force
Copy-Item -LiteralPath "$PSScriptRoot/THIRD_PARTY.md" -Destination "$destination/ThirdParty/README.md" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/signalsmith-stretch/LICENSE.txt" -Destination "$destination/ThirdParty/Signalsmith-Stretch-LICENSE.txt" -Force
Copy-Item -LiteralPath "$PSScriptRoot/vendor/signalsmith-linear/LICENSE.txt" -Destination "$destination/ThirdParty/Signalsmith-Linear-LICENSE.txt" -Force
Compress-Archive -LiteralPath $destination -DestinationPath "$projectRoot/dist/ToneHound-windows-x64.zip" -Force
Get-ChildItem -LiteralPath $destination | Select-Object Name,Length
