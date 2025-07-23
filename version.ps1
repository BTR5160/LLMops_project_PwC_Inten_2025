
# version.ps1 - Automate DVC versioning of latest run

Write-Host "🔍 Recherche du dernier dossier de run..."

# Get latest run directory
$lastRun = Get-ChildItem -Directory "data/output/runs" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $lastRun) {
   Write-Host "❌ Aucun dossier trouvé dans data/output/runs/"
   exit 1
}

$runPath = Join-Path "data/output/runs" $lastRun.Name
Write-Host "📁 Dossier trouvé : $runPath"

# DVC add
dvc add "$runPath"

# Git add & commit
$commitMessage = "Versionnage du run horodaté $($lastRun.Name)"
git add "$runPath.dvc" .gitignore
git commit -m "$commitMessage"

# DVC push
dvc push

Write-Host "✅ Run $($lastRun.Name) versionné et poussé avec succès !"

