$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDirectory = Join-Path $repoRoot 'source'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$installDirectory = Join-Path $env:LOCALAPPDATA 'Film Scan Converter'
$builds = @(
    @{
        EntryPoint = 'App.pyw'
        Name = 'Film-Scan-Converter'
    },
    @{
        EntryPoint = 'LightroomApp.pyw'
        Name = 'Film-Scan-Converter-Lightroom'
    }
)

if (-not (Test-Path $python)) {
    throw "Virtual environment Python not found: $python"
}

Push-Location $sourceDirectory
try {
    foreach ($build in $builds) {
        & $python -m PyInstaller `
            --noconfirm `
            --onefile `
            --windowed `
            --name $build.Name `
            --icon 'assets/camera-roll.ico' `
            --add-data 'assets;assets' `
            $build.EntryPoint

        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed for $($build.EntryPoint) with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}

New-Item -ItemType Directory -Path $installDirectory -Force | Out-Null

foreach ($build in $builds) {
    $exeName = "$($build.Name).exe"
    $builtExe = Join-Path $sourceDirectory "dist\$exeName"
    $installedExe = Join-Path $installDirectory $exeName

    if (-not (Test-Path $builtExe)) {
        throw "Build completed without producing: $builtExe"
    }

    Copy-Item -Path $builtExe -Destination $installedExe -Force
    if (-not (Test-Path $installedExe)) {
        throw "Failed to copy executable to: $installedExe"
    }

    Write-Host "Built and installed: $installedExe"
}
