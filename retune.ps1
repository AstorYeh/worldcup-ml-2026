# Daily auto-retune: fetch latest 2026 WC results, rerun Monte Carlo,
# commit + push ONLY if mc_results.pkl actually changed (no-noise commits).
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "retune.log"
Add-Content -Path $log -Value "============================================"
Add-Content -Path $log -Value "[$(Get-Date)] retune start"

python retune.py *>> $log

git add models/mc_results.pkl
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "chore(auto): daily retune - rerun MC with latest WC results" *>> $log 2>&1
    git push origin main *>> $log 2>&1
    Add-Content -Path $log -Value "[$(Get-Date)] CHANGED - committed and pushed"
} else {
    Add-Content -Path $log -Value "[$(Get-Date)] no new results - skipped"
}
