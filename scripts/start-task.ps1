param(
    [Parameter(Mandatory = $true, Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Task,

    [ValidateSet("feat", "fix", "refactor", "test", "docs", "chore")]
    [string]$Type,

    [string]$BaseBranch = "develop",
    [string]$Remote = "origin",
    [switch]$NoPull,
    [switch]$AllowDirty,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$TaskText = ($Task -join " ").Trim()
if ([string]::IsNullOrWhiteSpace($TaskText)) {
    Write-Error "Task description is required."
    exit 1
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,
        [switch]$Mutating,
        [switch]$Capture
    )

    $cmd = "git " + ($Args -join " ")
    if ($DryRun -and $Mutating) {
        Write-Host "[dry-run] $cmd"
        if ($Capture) {
            return ""
        }
        return
    }

    if ($Capture) {
        $output = & git @Args 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $cmd`n$output"
        }
        return ($output -join "`n").Trim()
    }

    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $cmd"
    }
}

function Test-GitRefExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Ref
    )
    & git show-ref --verify --quiet $Ref
    return ($LASTEXITCODE -eq 0)
}

function Convert-ToAsciiLower {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text
    )

    $normalized = $Text.Normalize([Text.NormalizationForm]::FormD)
    $sb = [System.Text.StringBuilder]::new()
    foreach ($ch in $normalized.ToCharArray()) {
        if ([Globalization.CharUnicodeInfo]::GetUnicodeCategory($ch) -ne [Globalization.UnicodeCategory]::NonSpacingMark) {
            [void]$sb.Append($ch)
        }
    }
    return $sb.ToString().ToLowerInvariant()
}

function Resolve-TypeFromTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskText
    )

    if ($Type) {
        return $Type.ToLowerInvariant()
    }

    $norm = Convert-ToAsciiLower $TaskText

    $map = @(
        @{ Type = "fix"; Patterns = @("fix", "bug", "loi", "error", "hotfix", "regression") },
        @{ Type = "refactor"; Patterns = @("refactor", "clean", "cleanup") },
        @{ Type = "test"; Patterns = @("test", "pytest", "regression test") },
        @{ Type = "docs"; Patterns = @("doc", "docs", "readme", "tai lieu") },
        @{ Type = "chore"; Patterns = @("chore", "config", "hook", "workflow") }
    )

    foreach ($entry in $map) {
        foreach ($p in $entry.Patterns) {
            if ($norm -like "*$p*") {
                return $entry.Type
            }
        }
    }

    return "feat"
}

function New-BranchSlug {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskText,
        [int]$MaxLength = 48
    )

    $norm = Convert-ToAsciiLower $TaskText
    $slug = ($norm -replace "[^a-z0-9]+", "-").Trim("-")
    $slug = ($slug -replace "-{2,}", "-")

    if ([string]::IsNullOrWhiteSpace($slug)) {
        $slug = "task"
    }

    if ($slug.Length -gt $MaxLength) {
        $slug = $slug.Substring(0, $MaxLength).Trim("-")
    }
    if ([string]::IsNullOrWhiteSpace($slug)) {
        $slug = "task"
    }

    return $slug
}

function Test-BranchExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BranchName
    )

    if (Test-GitRefExists "refs/heads/$BranchName") {
        return $true
    }
    if (Test-GitRefExists "refs/remotes/$Remote/$BranchName") {
        return $true
    }
    return $false
}

function Assert-BranchNamespaceAvailable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteName
    )

    $reservedPrefixes = @("feat", "fix", "refactor", "test", "docs", "chore")

    foreach ($prefix in $reservedPrefixes) {
        if (Test-GitRefExists "refs/heads/$prefix") {
            throw "Branch '$prefix' exists and blocks namespaced branches like '$prefix/<task>'. Rename/delete '$prefix' first."
        }
        if (Test-GitRefExists "refs/remotes/$RemoteName/$prefix") {
            throw "Remote branch '$RemoteName/$prefix' exists and may block namespaced refs '$RemoteName/$prefix/<task>'. Rename/delete remote '$RemoteName/$prefix' first."
        }
    }
}

try {
    $repoRoot = Invoke-Git -Args @("rev-parse", "--show-toplevel") -Capture
    if ([string]::IsNullOrWhiteSpace($repoRoot)) {
        throw "Not inside a git repository."
    }
    Set-Location $repoRoot

    $status = Invoke-Git -Args @("status", "--porcelain") -Capture
    if (-not $AllowDirty -and -not [string]::IsNullOrWhiteSpace($status)) {
        throw "Working tree has uncommitted changes. Commit/stash first or use -AllowDirty."
    }

    $hasLocalBase = Test-GitRefExists "refs/heads/$BaseBranch"
    $hasRemoteBase = Test-GitRefExists "refs/remotes/$Remote/$BaseBranch"

    if (-not $hasLocalBase -and -not $hasRemoteBase) {
        throw "Base branch '$BaseBranch' not found locally or on '$Remote'."
    }

    if (-not $hasLocalBase -and $hasRemoteBase) {
        Invoke-Git -Args @("checkout", "-b", $BaseBranch, "--track", "$Remote/$BaseBranch") -Mutating
    } else {
        Invoke-Git -Args @("checkout", $BaseBranch) -Mutating
    }

    if (-not $NoPull) {
        Invoke-Git -Args @("pull", "--ff-only", $Remote, $BaseBranch) -Mutating
    }

    Assert-BranchNamespaceAvailable -RemoteName $Remote

    $resolvedType = Resolve-TypeFromTask -TaskText $TaskText
    $slug = New-BranchSlug -TaskText $TaskText
    $baseName = "$resolvedType/$slug"
    $candidate = $baseName
    $suffix = 2
    while (Test-BranchExists -BranchName $candidate) {
        $candidate = "$baseName-$suffix"
        $suffix++
    }
    $branchName = $candidate

    Invoke-Git -Args @("checkout", "-b", $branchName) -Mutating

    Write-Host ""
    Write-Host "Start-task result:"
    Write-Host "  Base branch : $BaseBranch"
    Write-Host "  Type        : $resolvedType"
    Write-Host "  Slug        : $slug"
    Write-Host "  New branch  : $branchName"
    Write-Host ""
    Write-Host "Next:"
    Write-Host "  1) Start coding on this branch"
    Write-Host "  2) Commit via strict flow"
    Write-Host "  3) Push with: git push -u $Remote $branchName"
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
