function Finalize-MrFlowLayout {
    param(
        [Parameter(Mandatory=$true)][string]$ComfyDir,
        [Parameter(Mandatory=$true)][string]$CustomNodes
    )
    $package = Join-Path $CustomNodes "ComfyUI-Rebels-MrFlow"
    $outer = Join-Path $CustomNodes "Rebels_MrFlow"
    if (-not (Test-Path $outer)) {
        return $null
    }
    $inner = Join-Path $outer "ComfyUI-Rebels-MrFlow"
    if (-not (Test-Path (Join-Path $package "__init__.py"))) {
        if (Test-Path $package) {
            throw "Existing Mr. Flow package directory is incomplete; refusing to replace it."
        }
        if (-not (Test-Path (Join-Path $inner "__init__.py"))) {
            throw "Rebels_MrFlow source layout is invalid; leaving it untouched."
        }
        Copy-Item -Path $inner -Destination $package -Recurse
    }
    $workflowDir = Join-Path $ComfyDir "user\default\workflows\Rebels_MrFlow"
    New-Item -ItemType Directory -Force -Path $workflowDir | Out-Null
    Get-ChildItem -Path $outer -Filter "*.json" -File | ForEach-Object {
        Copy-Item -Path $_.FullName -Destination (Join-Path $workflowDir $_.Name) -Force
    }
    $sources = Join-Path $ComfyDir "user\__sources"
    New-Item -ItemType Directory -Force -Path $sources | Out-Null
    $archive = Join-Path $sources "Rebels_MrFlow"
    $suffix = 1
    while (Test-Path $archive) {
        $archive = Join-Path $sources "Rebels_MrFlow-$suffix"
        $suffix += 1
    }
    Move-Item -Path $outer -Destination $archive
    return $archive
}
