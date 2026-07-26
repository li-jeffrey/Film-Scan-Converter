$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDirectory = Join-Path $repoRoot 'source'
$installDirectory = Join-Path $env:LOCALAPPDATA 'Film Scan Converter'
$exeNames = @(
    'Film-Scan-Converter.exe',
    'Film-Scan-Converter-Lightroom.exe'
)

& (Join-Path $PSScriptRoot 'build.ps1')

New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null

foreach ($exeName in $exeNames) {
    $builtExe = Join-Path $sourceDirectory "dist\$exeName"
    $installedExe = Join-Path $installDirectory $exeName

    if (-not (Test-Path $builtExe)) {
        throw "Build completed without producing: $builtExe"
    }

    Copy-Item -Path $builtExe -Destination $installedExe -Force
    if (-not (Test-Path $installedExe)) {
        throw "Failed to copy executable to: $installedExe"
    }

    Write-Host "Installed: $installedExe"
}
