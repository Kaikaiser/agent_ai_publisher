$projectRoot = 'E:\daima\agent\ai_assistant'
$scriptDir = Join-Path $projectRoot 'scripts'
$ps = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'

Start-Process -FilePath $ps -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $scriptDir 'start_backend.ps1'))
Start-Process -FilePath $ps -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $scriptDir 'start_frontend.ps1'))