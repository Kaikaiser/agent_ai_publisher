param(
    [switch]$Reload
)

$projectRoot = 'E:\daima\agent\ai_assistant'
$backendDir = Join-Path $projectRoot 'backend'
$pythonExe = 'D:\anaconda\envs\agent\python.exe'
$port = 8000

# Clean up an existing listener first so the service always starts on a predictable port.
$listenerLines = netstat -ano | Select-String ":$port"
foreach ($line in $listenerLines) {
    $parts = ($line.ToString() -split '\s+') | Where-Object { $_ }
    if ($parts.Length -ge 5) {
        $pid = $parts[-1]
        if ($pid -match '^\d+$') {
            taskkill /PID $pid /F | Out-Null
        }
    }
}

Set-Location $backendDir

$args = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', $port)
if ($Reload) {
    $args += '--reload'
}

& $pythonExe @args