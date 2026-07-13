function Get-KreaDeforumSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString(
                $sha.ComputeHash($stream)
            ) -replace "-", "").ToLowerInvariant()
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Install-KreaDeforumCheckout {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Repository,
        [Parameter(Mandatory = $true)]
        [string]$Revision,
        [Parameter(Mandatory = $true)]
        [string]$Destination,
        [string]$PatchPath = "",
        [string]$PatchedFile = "",
        [string]$PatchedSha256 = "",
        [string]$PatchSha256 = ""
    )

    $usePatch = -not [string]::IsNullOrWhiteSpace($PatchPath)
    if ($usePatch) {
        if (-not (Test-Path -LiteralPath $PatchPath -PathType Leaf)) {
            throw "Repository-owned KreaDeforum patch is missing: $PatchPath"
        }
        if (
            [string]::IsNullOrWhiteSpace($PatchedFile) -or
            $PatchedFile -match '(^|[\\/])\.\.([\\/]|$)' -or
            [System.IO.Path]::IsPathRooted($PatchedFile) -or
            $PatchedSha256 -notmatch '^[a-fA-F0-9]{64}$' -or
            $PatchSha256 -notmatch '^[a-fA-F0-9]{64}$'
        ) {
            throw "KreaDeforum patch metadata is invalid."
        }
        $actualPatchHash = Get-KreaDeforumSha256 -Path $PatchPath
        if ($actualPatchHash -ne $PatchSha256.ToLowerInvariant()) {
            throw "KreaDeforum patch artifact hash verification failed."
        }
    }

    if (Test-Path $Destination) {
        if (-not (Test-Path (Join-Path $Destination ".git"))) {
            throw "KreaDeforum exists but is not a git checkout: $Destination. Move or remove it, then re-run install.bat."
        }

        $dirty = @(& git -C "$Destination" status --porcelain)
        if ($LASTEXITCODE -ne 0) {
            throw "Could not inspect the existing KreaDeforum checkout at $Destination."
        }
        if ($dirty.Count -gt 0) {
            $patchedPath = Join-Path $Destination $PatchedFile
            $expectedDirty = " M $($PatchedFile -replace '\\', '/')"
            $diffNames = @(& git -C "$Destination" diff --name-only)
            if (
                $usePatch -and
                $dirty.Count -eq 1 -and
                [string]$dirty[0] -eq $expectedDirty -and
                $diffNames.Count -eq 1 -and
                [string]$diffNames[0] -eq ($PatchedFile -replace '\\', '/') -and
                (Test-Path -LiteralPath $patchedPath -PathType Leaf)
            ) {
                $head = & git -C "$Destination" rev-parse HEAD
                $actualHash = Get-KreaDeforumSha256 -Path $patchedPath
                if ([string]$head -ne $Revision) {
                    throw "KreaDeforum patched checkout is not at required revision $Revision."
                }
                if ($actualHash -ne $PatchedSha256.ToLowerInvariant()) {
                    throw "KreaDeforum patched hash does not match the repository-owned compatibility patch."
                }
                return
            }
            throw "KreaDeforum has local changes at $Destination. Commit, stash, or move them before re-running install.bat."
        }
    } else {
        & git clone --depth 1 --no-checkout $Repository "$Destination"
        if ($LASTEXITCODE -ne 0) {
            throw "Required Animate node KreaDeforum failed to clone."
        }
    }

    & git -C "$Destination" fetch --depth 1 $Repository $Revision
    if ($LASTEXITCODE -ne 0) {
        throw "Could not fetch pinned KreaDeforum revision $Revision."
    }
    & git -C "$Destination" -c core.autocrlf=false checkout --detach --force $Revision
    if ($LASTEXITCODE -ne 0) {
        throw "Could not check out pinned KreaDeforum revision $Revision."
    }

    $head = & git -C "$Destination" rev-parse HEAD
    $headExit = $LASTEXITCODE
    if ($headExit -ne 0 -or [string]$head -ne $Revision) {
        throw "KreaDeforum checkout did not resolve to required revision $Revision."
    }

    if ($usePatch) {
        & git -C "$Destination" -c core.autocrlf=false apply --check --ignore-space-change "$PatchPath"
        if ($LASTEXITCODE -ne 0) {
            throw "Repository-owned KreaDeforum compatibility patch does not match pinned upstream context."
        }
        & git -C "$Destination" -c core.autocrlf=false apply --ignore-space-change --whitespace=nowarn "$PatchPath"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not apply repository-owned KreaDeforum compatibility patch."
        }
        $patchedPath = Join-Path $Destination $PatchedFile
        if (-not (Test-Path -LiteralPath $patchedPath -PathType Leaf)) {
            throw "Patched KreaDeforum file is missing: $PatchedFile"
        }
        $actualHash = Get-KreaDeforumSha256 -Path $patchedPath
        if ($actualHash -ne $PatchedSha256.ToLowerInvariant()) {
            throw "KreaDeforum patched hash verification failed."
        }
        $dirty = @(& git -C "$Destination" status --porcelain)
        $diffNames = @(& git -C "$Destination" diff --name-only)
        $expectedDirty = " M $($PatchedFile -replace '\\', '/')"
        if (
            $dirty.Count -ne 1 -or
            [string]$dirty[0] -ne $expectedDirty -or
            $diffNames.Count -ne 1 -or
            [string]$diffNames[0] -ne ($PatchedFile -replace '\\', '/')
        ) {
            throw "KreaDeforum patch must produce exactly one modified file: $PatchedFile"
        }
    }
}
