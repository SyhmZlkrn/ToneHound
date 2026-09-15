param([ValidateSet('Release','Debug')][string]$Configuration = 'Release')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$toolRoot = Join-Path $projectRoot '.cache/toolchain'
$msvcRoot = Join-Path $toolRoot 'msvc'
$msvcTools = (Get-ChildItem "$msvcRoot/VC/Tools/MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName
$sdkRoot = Join-Path $msvcRoot 'Windows Kits/10'
$sdkVersion = (Get-ChildItem "$sdkRoot/Include" -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
$compilerBin = Join-Path $msvcTools 'bin/Hostx64/x64'
$cmakeExe = Join-Path $toolRoot 'python-tools/cmake/data/bin/cmake.exe'
$ninjaExe = Join-Path $toolRoot 'python-tools/bin/ninja.exe'
if (!(Test-Path $cmakeExe) -or !(Test-Path "$compilerBin/cl.exe")) {
    throw 'The local native toolchain is missing. See native/README.md.'
}
$env:PATH = "$compilerBin;$sdkRoot/bin/$sdkVersion/x64;$sdkRoot/bin/$sdkVersion/x64/ucrt;$(Split-Path -Parent $ninjaExe);$env:PATH"
$env:INCLUDE = "$msvcTools/include;$sdkRoot/Include/$sdkVersion/ucrt;$sdkRoot/Include/$sdkVersion/shared;$sdkRoot/Include/$sdkVersion/um;$sdkRoot/Include/$sdkVersion/winrt;$sdkRoot/Include/$sdkVersion/cppwinrt"
$env:LIB = "$msvcTools/lib/x64;$sdkRoot/Lib/$sdkVersion/ucrt/x64;$sdkRoot/Lib/$sdkVersion/um/x64"
& $cmakeExe -S $PSScriptRoot -B "$PSScriptRoot/build-msvc" -G Ninja "-DCMAKE_BUILD_TYPE=$Configuration" "-DCMAKE_C_COMPILER=$compilerBin/cl.exe" "-DCMAKE_CXX_COMPILER=$compilerBin/cl.exe" "-DCMAKE_MAKE_PROGRAM=$ninjaExe"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmakeExe --build "$PSScriptRoot/build-msvc" --parallel 4
exit $LASTEXITCODE
