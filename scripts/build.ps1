$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDirectory = Join-Path $repoRoot 'source'
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
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

foreach ($build in $builds) {
    $exeName = "$($build.Name).exe"
    $builtExe = Join-Path $sourceDirectory "dist\$exeName"

    if (-not (Test-Path $builtExe)) {
        throw "Build completed without producing: $builtExe"
    }

    Write-Host "Built: $builtExe"
}
