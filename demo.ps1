$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$preferredPython = "D:\tau2-bench\.venv\Scripts\python.exe"

if (Test-Path -LiteralPath $preferredPython) {
    $python = $preferredPython
} else {
    $pythonCommand = Get-Command python -ErrorAction Stop
    $python = $pythonCommand.Source
}

Push-Location $projectRoot
try {
    & $python -m src.project_summary --project-root $projectRoot
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
