function ConvertTo-ComfyCommandArguments {
    param([string]$CommandLine)
    $arguments = @()
    foreach ($match in [regex]::Matches($CommandLine, '("[^"]*"|\S+)')) {
        $token = $match.Value
        if ($token.Length -ge 2 -and $token[0] -eq '"' -and $token[$token.Length - 1] -eq '"') {
            $token = $token.Substring(1, $token.Length - 2)
        }
        $arguments += $token
    }
    return $arguments
}

function Test-StableListenerPid {
    param([int]$BeforePid, [int]$AfterPid)
    return $BeforePid -gt 0 -and $BeforePid -eq $AfterPid
}

function Test-ComfyCancelCapability {
    param(
        [string]$BaseUrl,
        [scriptblock]$RequestInvoker = $null
    )
    $probeId = "krea-capability-" + [guid]::NewGuid().ToString("N")
    $uri = "$($BaseUrl.TrimEnd('/'))/api/jobs/$probeId/cancel"
    try {
        $response = if ($RequestInvoker) {
            & $RequestInvoker $uri "{}"
        } else {
            Invoke-WebRequest -Uri $uri -Method Post -Body "{}" `
                -ContentType "application/json" -UseBasicParsing -TimeoutSec 5
        }
        if ([int]$response.StatusCode -ne 200) {
            return $false
        }
        $payload = $response.Content | ConvertFrom-Json
        $property = $payload.PSObject.Properties["cancelled"]
        return $null -ne $property -and $property.Value -is [bool]
    } catch {
        return $false
    }
}

function Test-ComfyCancelRouteSource {
    param([string]$ComfyDir)
    try {
        $matches = Get-ChildItem -Path $ComfyDir -Recurse -File -Filter "*.py" |
            Select-String -Pattern '/api/jobs/.+/cancel' -CaseSensitive
        return $null -ne $matches
    } catch {
        return $false
    }
}

function Test-ComfyProcessOwnership {
    param(
        [object]$Process,
        [string]$ExpectedPython,
        [string]$ExpectedMain,
        [int]$StartedPid = 0,
        [scriptblock]$ProcessLookup = $null
    )
    if (-not $Process -or -not $Process.ExecutablePath -or -not $Process.CommandLine) {
        return $false
    }
    $executable = [IO.Path]::GetFullPath([string]$Process.ExecutablePath)
    $expectedExecutable = [IO.Path]::GetFullPath($ExpectedPython)
    if (-not $executable.Equals($expectedExecutable, [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    $expectedMainPath = [IO.Path]::GetFullPath($ExpectedMain)
    $hasMainToken = $false
    foreach ($argument in ConvertTo-ComfyCommandArguments ([string]$Process.CommandLine)) {
        try {
            if (
                $argument -match '^(?:[A-Za-z]:[\\/]|\\\\)' -and
                [IO.Path]::GetFullPath($argument).Equals($expectedMainPath, [StringComparison]::OrdinalIgnoreCase)
            ) {
                $hasMainToken = $true
                break
            }
        } catch {}
    }
    if (-not $hasMainToken) {
        return $false
    }
    if ($StartedPid -le 0) {
        return $true
    }
    $candidate = $Process
    $visited = @{}
    while ($candidate -and -not $visited.ContainsKey([int]$candidate.ProcessId)) {
        $candidatePid = [int]$candidate.ProcessId
        $visited[$candidatePid] = $true
        if ($candidatePid -eq $StartedPid) {
            return $true
        }
        $parentPid = [int]$candidate.ParentProcessId
        if ($parentPid -eq $StartedPid) {
            return $true
        }
        if ($parentPid -le 0 -or -not $ProcessLookup) {
            break
        }
        $candidate = & $ProcessLookup $parentPid
    }
    return $false
}
