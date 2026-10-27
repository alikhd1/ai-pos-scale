#Requires -RunAsAdministrator
<#
enable_remote_access.ps1
------------------------
Give remote support a command line on this till, through the support server.

The till sits behind a shop router nobody controls, so nothing here opens an
inbound port. Instead the till keeps one outbound SSH connection open to the
support server (5.236.19.17) and offers, through it, two things to whoever is
already logged in on that server:

    server 127.0.0.1:2322  ->  this till's SSH server   (a command line)
    server 127.0.0.1:2359  ->  this till's VNC server   (the screen, if one is installed)

Both ends are bound to the loopback: neither is reachable from the internet,
on the server or here. On the server the till's key is restricted to opening
exactly those two listeners - it cannot get a shell there.

What this script changes, and nothing else:
  * installs Windows' own OpenSSH Server, listening on 127.0.0.1 only,
    keys only (passwords off), with PowerShell as the shell
  * authorises ONE key - remote support's - for administrators
  * creates a key this till uses for the outbound tunnel
  * a scheduled task, running as SYSTEM from boot, that keeps the tunnel up

Undo all of it with tools\disable_remote_access.ps1.

Run, in PowerShell started as Administrator:
    powershell -ExecutionPolicy Bypass -File D:\code\ai-pos-scale\tools\enable_remote_access.ps1
#>
$ErrorActionPreference = "Stop"

$SupportKey   = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIGNzP7KnA8xJnG53lW0kWdOBO+DxOGnr3AOw2lVdTzr/ ai-pos-remote-support"
$Server       = "5.236.19.17"
$ServerPort   = 2222
$SshDir       = Join-Path $env:ProgramData "ssh"
$TunnelKey    = Join-Path $SshDir "posrelay_key"
$KnownHosts   = Join-Path $SshDir "posrelay_known_hosts"
$TunnelScript = Join-Path $SshDir "posrelay-tunnel.ps1"
$TaskName     = "AI POS remote support tunnel"
$SYSTEM       = "*S-1-5-18"      # SIDs, so this works on a Persian or any other Windows
$ADMINS       = "*S-1-5-32-544"

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }

# ---------------------------------------------------------------- 1. sshd
Step "OpenSSH Server"
$cap = Get-WindowsCapability -Online -Name "OpenSSH.Server*" | Select-Object -First 1
if ($cap.State -ne "Installed") {
    try {
        Add-WindowsCapability -Online -Name $cap.Name | Out-Null
    } catch {
        Write-Host "Windows could not install OpenSSH Server: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "It downloads from Windows Update. If that is blocked here, send this message to support."
        exit 1
    }
}
Start-Service sshd                  # the first start writes the default sshd_config
Set-Service sshd -StartupType Automatic

Step "sshd: loopback only, keys only, PowerShell shell"
$cfg = Join-Path $SshDir "sshd_config"
$text = Get-Content $cfg -Raw
# [ \t] not \s, and [^\r\n] not .: the file is CRLF, and \s would let a match
# swallow a blank line while . would eat the \r and leave mixed line endings
$text = $text -replace "(?m)^#?[ \t]*ListenAddress[ \t]+0\.0\.0\.0[^\r\n]*", "ListenAddress 127.0.0.1"
$text = $text -replace "(?m)^#?[ \t]*PasswordAuthentication[ \t]+[^\r\n]*", "PasswordAuthentication no"
$text = $text -replace "(?m)^#?[ \t]*PubkeyAuthentication[ \t]+[^\r\n]*", "PubkeyAuthentication yes"
if ($text -notmatch "(?m)^ListenAddress 127\.0\.0\.1") { $text = "ListenAddress 127.0.0.1`r`n" + $text }
Set-Content -Path $cfg -Value $text -Encoding ascii
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -Force -PropertyType String `
    -Value "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" | Out-Null
# nothing needs to reach sshd from the network: the tunnel arrives on the loopback
Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue | Disable-NetFirewallRule

Step "authorising remote support's key"
$ak = Join-Path $SshDir "administrators_authorized_keys"
$existing = if (Test-Path $ak) { Get-Content $ak } else { @() }
if ($existing -notcontains $SupportKey) { Add-Content -Path $ak -Value $SupportKey -Encoding ascii }
# sshd ignores this file unless only SYSTEM and Administrators can touch it
icacls $ak /inheritance:r /grant "${SYSTEM}:F" /grant "${ADMINS}:F" | Out-Null
Restart-Service sshd

# ---------------------------------------------------------------- 2. tunnel key
Step "the key this till uses for its outbound tunnel"
if (-not (Test-Path $TunnelKey)) {
    # through cmd: Windows PowerShell 5.1 drops an empty "" argument on the way to a native program
    cmd /c "ssh-keygen -q -t ed25519 -N `"`" -C `"till-$env:COMPUTERNAME`" -f `"$TunnelKey`""
}
icacls $TunnelKey /inheritance:r /grant "${SYSTEM}:F" /grant "${ADMINS}:F" | Out-Null
icacls $TunnelKey /setowner $SYSTEM | Out-Null

# ---------------------------------------------------------------- 3. keep it up
Step "scheduled task keeping the tunnel up"
@"
# written by enable_remote_access.ps1 - keeps the remote-support tunnel open
while (`$true) {
    & "`$env:SystemRoot\System32\OpenSSH\ssh.exe" -N -T -p $ServerPort ``
        -i "$TunnelKey" -o IdentitiesOnly=yes ``
        -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes ``
        -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KnownHosts" ``
        -R localhost:2322:127.0.0.1:22 -R localhost:2359:127.0.0.1:5900 ``
        posrelay@$Server
    Start-Sleep -Seconds 20
}
"@ | Set-Content -Path $TunnelScript -Encoding ascii
icacls $TunnelScript /inheritance:r /grant "${SYSTEM}:F" /grant "${ADMINS}:F" | Out-Null

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
$action   = New-ScheduledTaskAction -Execute "powershell.exe" `
            -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$TunnelScript`""
$trigger  = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -User "SYSTEM" -RunLevel Highest | Out-Null
Start-ScheduledTask -TaskName $TaskName

# ---------------------------------------------------------------- 4. hand-over
Step "done - send the line below to remote support"
Write-Host ""
Write-Host "----------------------------------------------------------------------" -ForegroundColor Yellow
Get-Content "$TunnelKey.pub"
Write-Host "----------------------------------------------------------------------" -ForegroundColor Yellow
Write-Host ""
Write-Host "That is this till's PUBLIC key - safe to send. The tunnel starts working"
Write-Host "once support adds it on the server. To undo everything:"
Write-Host "    powershell -ExecutionPolicy Bypass -File $PSScriptRoot\disable_remote_access.ps1"
