param(
    [string]$Python = 'python',
    [switch]$WaitForWeights
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
if ($WaitForWeights) {
    Write-Output 'Waiting for completed checkpoint manifest...'
    while (-not (Test-Path -LiteralPath 'weights/manifest.json')) { Start-Sleep -Seconds 5 }
}
if (-not (Test-Path -LiteralPath 'weights/manifest.json')) { throw 'Run download_weights.py first.' }
New-Item -ItemType Directory -Force -Path 'results','results_seed2','results_seed3' | Out-Null
& $Python -u run_experiment.py --steps 20 100 --seed 20261116 --output results 2>&1 | Tee-Object -FilePath results/run.log
if ($LASTEXITCODE -ne 0) { throw 'Main experiment failed.' }
& $Python make_figures.py
if ($LASTEXITCODE -ne 0) { throw 'Figure creation failed.' }
& $Python -u run_experiment.py --steps 20 --seed 20261216 --output results_seed2 2>&1 | Tee-Object -FilePath results_seed2/run.log
if ($LASTEXITCODE -ne 0) { throw 'Seed 2 failed.' }
& $Python -u run_experiment.py --steps 20 --seed 20261316 --output results_seed3 2>&1 | Tee-Object -FilePath results_seed3/run.log
if ($LASTEXITCODE -ne 0) { throw 'Seed 3 failed.' }
& $Python summarize_seeds.py
if ($LASTEXITCODE -ne 0) { throw 'Seed summary failed.' }
Write-Output 'All experiments completed.'
