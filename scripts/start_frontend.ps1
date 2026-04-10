$projectRoot = 'E:\daima\agent\ai_assistant'
$frontendDir = Join-Path $projectRoot 'frontend'

Set-Location $frontendDir
npm.cmd run dev -- --host 127.0.0.1 --port 5173