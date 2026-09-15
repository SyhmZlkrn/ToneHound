param(
    [ValidateSet('Check', 'Install')][string]$Mode = 'Check',
    [string]$Python = 'python',
    [ValidateSet('cpu', 'cu128')][string]$Torch = 'cpu',
    [switch]$Development,
    [switch]$SkipNative
)
$ErrorActionPreference = 'Stop'
$tonehoundRoot = Split-Path -Parent $PSScriptRoot
$tonehoundVenv = Join-Path $tonehoundRoot '.venv'
$tonehoundPython = Join-Path $tonehoundVenv 'Scripts/python.exe'
$pins = Get-Content (Join-Path $tonehoundRoot 'release/native_dependencies.json') -Raw | ConvertFrom-Json

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Executable failed (exit $LASTEXITCODE)." }
}

if ($Mode -eq 'Check') {
    Write-Output 'Read-only bootstrap check; no packages, downloads or settings are changed.'
    foreach ($command in @($Python, 'git', 'cmake', 'ffmpeg')) {
        $found = Get-Command $command -ErrorAction SilentlyContinue
        [PSCustomObject]@{Dependency = $command; Available = [bool]$found}
    }
    foreach ($pin in $pins.dependencies) {
        $folder = Join-Path $tonehoundRoot $pin.path
        $present = Test-Path -LiteralPath (Join-Path $folder '.git')
        [PSCustomObject]@{Dependency = $pin.path; Available = $present}
        if ($present) {
            $actual = & git -C $folder rev-parse HEAD
            if ($actual -ne $pin.revision) { throw "Revision mismatch in $($pin.path)." }
        }
    }
    exit 0
}

# Install only into this dedicated virtual environment; never modify global Python.
Invoke-Checked $Python @('-c', 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 for this validated configuration"')
if (!(Test-Path -LiteralPath $tonehoundPython)) {
    Invoke-Checked $Python @('-m', 'venv', $tonehoundVenv)
}
Invoke-Checked $tonehoundPython @('-m', 'pip', 'install', '--index-url', "https://download.pytorch.org/whl/$Torch", 'torch==2.11.0')
$requirements = if ($Development) { 'requirements-dev.txt' } else { 'requirements-engine.txt' }
Invoke-Checked $tonehoundPython @('-m', 'pip', 'install', '-r', (Join-Path $tonehoundRoot $requirements))
Invoke-Checked $tonehoundPython @('-m', 'pip', 'check')

if (!$SkipNative) {
    foreach ($pin in $pins.dependencies) {
        $folder = Join-Path $tonehoundRoot $pin.path
        if (Test-Path -LiteralPath $folder) {
            $actual = & git -C $folder rev-parse HEAD
            if ($LASTEXITCODE -ne 0 -or $actual -ne $pin.revision) {
                throw "Existing $($pin.path) has a different revision. It was left unchanged."
            }
        } else {
            New-Item -ItemType Directory -Path $folder -Force | Out-Null
            Invoke-Checked 'git' @('-C', $folder, 'init')
            Invoke-Checked 'git' @('-C', $folder, 'remote', 'add', 'origin', $pin.repository)
            Invoke-Checked 'git' @('-C', $folder, 'fetch', '--depth', '1', 'origin', $pin.revision)
            Invoke-Checked 'git' @('-C', $folder, 'checkout', '--detach', 'FETCH_HEAD')
        }
        if ($pin.submodules) {
            Invoke-Checked 'git' @('-C', $folder, 'submodule', 'update', '--init', '--recursive', '--depth', '1')
        }
    }
}

$runtime = Join-Path $tonehoundRoot '.cache/native/runtime.json'
if (!(Test-Path -LiteralPath $runtime)) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $runtime) -Force | Out-Null
    @{python = $tonehoundPython} | ConvertTo-Json | Set-Content -LiteralPath $runtime -Encoding UTF8
} else {
    Write-Output 'Existing runtime.json preserved. Select the new venv manually if migrating this computer.'
}
Write-Output 'Dependencies prepared. See README.md for MSVC build, FFmpeg, TONE3000 login and model downloads.'
