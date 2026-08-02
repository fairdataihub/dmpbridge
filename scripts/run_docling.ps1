# Run all three models with the Docling extractor, one sample at a time.
#
# Safe to restart — the CLI auto-skips samples whose output files already
# exist. All GPU layers forced via num_gpu=-1 in the Ollama request.
#
# Usage:
#   .\scripts\run_docling.ps1                    # all models, samples 1-10
#   .\scripts\run_docling.ps1 -Model gemma4:e4b  # one model only
#   .\scripts\run_docling.ps1 -StartSample 5     # resume from sample 5

param(
    [string]$Model       = "",          # leave empty to run all three
    [int]$StartSample    = 1,
    [int]$EndSample      = 10,
    [int]$CooldownSec    = 10
)

$allModels = @("llama3.1:8b", "gemma4:e4b", "llama3.3:70b")
$models    = if ($Model) { @($Model) } else { $allModels }

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Extractor : docling" -ForegroundColor Cyan
Write-Host "  Models    : $($models -join ', ')" -ForegroundColor Cyan
Write-Host "  Samples   : $StartSample to $EndSample (one at a time)" -ForegroundColor Cyan
Write-Host "  GPU       : num_gpu=-1 (all layers, set in ollama.py)" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$totalFailed = @()

foreach ($m in $models) {
    Write-Host ""
    Write-Host "====== Model: $m ======" -ForegroundColor Magenta
    $failed = @()

    for ($i = $StartSample; $i -le $EndSample; $i++) {
        Write-Host "  --- sample$i ---" -ForegroundColor Yellow
        dmpbridge-wholedoc --model $m --extractor docling --start $i --end $i

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: sample$i exited with code $LASTEXITCODE" -ForegroundColor Red
            $failed += $i
        } else {
            Write-Host "  OK: sample$i done" -ForegroundColor Green
        }

        if ($i -lt $EndSample) {
            Write-Host "  Cooling down ${CooldownSec}s..."
            Start-Sleep -Seconds $CooldownSec
        }
    }

    if ($failed.Count -gt 0) {
        Write-Host "  Failed samples for ${m}: $($failed -join ', ')" -ForegroundColor Red
        $totalFailed += $failed
    }
}

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
if ($totalFailed.Count -eq 0) {
    Write-Host "  All models and samples completed successfully." -ForegroundColor Green
} else {
    Write-Host "  Some samples failed. Re-run with:" -ForegroundColor Red
    Write-Host "  .\scripts\run_docling.ps1 -Model <model> -StartSample <n>" -ForegroundColor Yellow
}
Write-Host "======================================================" -ForegroundColor Cyan
