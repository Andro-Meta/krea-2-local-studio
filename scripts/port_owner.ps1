# port_owner.ps1 — report which process owns a given port
param([int]$Port)
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $conn) { Write-Host "Port $Port is free." ; exit 0 }
foreach ($c in $conn) {
    $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Port $Port is in use by: $($proc.Name) (PID $($proc.Id))"
    } else {
        Write-Host "Port $Port is in use by PID $($c.OwningProcess)"
    }
}
exit 1
