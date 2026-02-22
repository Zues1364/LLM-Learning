Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

Write-Host "[setup] repo: $repoRoot"

git config --local core.hooksPath ".githooks"
git config --local commit.template ".gitmessage.txt"
git config --local fetch.prune "true"
git config --local rebase.autoStash "true"
git config --local pull.rebase "true"
git config --local rerere.enabled "true"
git config --local commit.verbose "true"
git config --local alias.newtask '!powershell -ExecutionPolicy Bypass -File ./scripts/start-task.ps1'
git config --local alias.newtask-dry '!powershell -ExecutionPolicy Bypass -File ./scripts/start-task.ps1 -DryRun'

Write-Host "[setup] core.hooksPath    = $(git config --local core.hooksPath)"
Write-Host "[setup] commit.template   = $(git config --local commit.template)"
Write-Host "[setup] fetch.prune       = $(git config --local fetch.prune)"
Write-Host "[setup] rebase.autoStash  = $(git config --local rebase.autoStash)"
Write-Host "[setup] pull.rebase       = $(git config --local pull.rebase)"
Write-Host "[setup] rerere.enabled    = $(git config --local rerere.enabled)"
Write-Host "[setup] commit.verbose    = $(git config --local commit.verbose)"
Write-Host "[setup] alias.newtask     = $(git config --local alias.newtask)"
Write-Host "[setup] alias.newtask-dry = $(git config --local alias.newtask-dry)"

Write-Host ""
Write-Host "Done. You now have a solo-professional git workflow enabled for this repo."
