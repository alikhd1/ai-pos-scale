#Requires -RunAsAdministrator
<#
disable_remote_access.ps1
-------------------------
Take away everything enable_remote_access.ps1 set up, in one go.

  * stops and removes the tunnel task, so nothing connects out any more
  * removes remote support's key from administrators_authorized_keys
  * stops OpenSSH Server and sets it to manual
  * deletes this till's tunnel key, so the server-side entry is dead too

OpenSSH Server itself is left installed - it is a Windows component and
removing it is a separate decision - but it no longer runs or accepts the key.

Run, in PowerShell started as Administrator:
    powershell -ExecutionPolicy Bypass -File D:\code\ai-pos-scale\tools\disable_remote_access.ps1
#>
$ErrorActionPreference = "Continue"
$SshDir   = Join-Path $env:ProgramData "ssh"
$TaskName = "AI POS remote support tunnel"

Write-Host "==> stopping the tunnel" -ForegroundColor Cyan
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
    Where-Object { $_.CommandLine -match "posrelay@" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }

Write-Host "==> removing remote support's key" -ForegroundColor Cyan
$ak = Join-Path $SshDir "administrators_authorized_keys"
if (Test-Path $ak) {
    $kept = Get-Content $ak | Where-Object { $_ -notmatch "ai-pos-remote-support" }
    Set-Content -Path $ak -Value $kept -Encoding ascii
}

Write-Host "==> stopping OpenSSH Server" -ForegroundColor Cyan
Stop-Service sshd -ErrorAction SilentlyContinue
Set-Service sshd -StartupType Manual -ErrorAction SilentlyContinue

Write-Host "==> deleting the tunnel key" -ForegroundColor Cyan
foreach ($f in "posrelay_key", "posrelay_key.pub", "posrelay_known_hosts", "posrelay-tunnel.ps1") {
    Remove-Item (Join-Path $SshDir $f) -Force -ErrorAction SilentlyContinue
}

$left = @()
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) { $left += "the scheduled task" }
if ((Get-Service sshd -ErrorAction SilentlyContinue).Status -eq "Running") { $left += "sshd still running" }
if ((Test-Path $ak) -and (Select-String -Path $ak -Pattern "ai-pos-remote-support" -Quiet)) { $left += "the support key" }
if ($left) { Write-Host "`nNOT fully removed: $($left -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host "`nRemote access is off. Nothing on this till connects out or accepts support's key." -ForegroundColor Green
