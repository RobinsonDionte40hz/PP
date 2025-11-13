$response = Invoke-WebRequest -Uri "http://localhost:8000/ws/socket.io/?EIO=4&transport=polling" -UseBasicParsing
Write-Host "Status:" $response.StatusCode
Write-Host "Socket.IO is working!" -ForegroundColor Green
