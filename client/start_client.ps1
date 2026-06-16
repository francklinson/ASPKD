# ASD 客户端监控服务管理脚本 (PowerShell)
# 支持: start | stop | restart | status | log | test

# 配置
$CLIENT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$PID_FILE = Join-Path $CLIENT_DIR ".client.pid"
$LOG_FILE = Join-Path $CLIENT_DIR "client.log"
$PYTHON = "python"

# 颜色输出
function Write-Color($Text, $Color) {
    Write-Host $Text -ForegroundColor $Color
}

# 获取 PID
function Get-ClientPid {
    if (Test-Path $PID_FILE) {
        $pid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
        if ($pid) {
            # 验证进程是否存在
            $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($process -and $process.Name -like "*python*") {
                return $pid
            }
        }
        Remove-Item $PID_FILE -ErrorAction SilentlyContinue
    }
    return $null
}

# 显示帮助
function Show-Help {
    Write-Host "ASD 客户端监控服务管理脚本 (PowerShell)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "用法: .\start_client.ps1 [命令]"
    Write-Host ""
    Write-Host "命令:"
    Write-Host "  start      启动客户端"
    Write-Host "  stop       停止客户端"
    Write-Host "  restart    重启客户端"
    Write-Host "  status     查看客户端状态"
    Write-Host "  log        查看实时日志"
    Write-Host "  test       测试与服务端的连接"
    Write-Host ""
}

# 启动客户端
function Start-Client {
    $pid = Get-ClientPid
    if ($pid) {
        Write-Color "客户端已在运行中 (PID: $pid)" Yellow
        return
    }
    
    Write-Color "正在启动 ASD 客户端监控..." Cyan
    Set-Location $CLIENT_DIR
    
    # 加载 .env 文件
    if (Test-Path ".env") {
        Write-Color "加载配置..." Cyan
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^([^#][^=]*)=(.*)$") {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
            }
        }
    }
    
    # 清空旧日志
    if (Test-Path $LOG_FILE) {
        Clear-Content $LOG_FILE
    }
    
    # 启动客户端
    $process = Start-Process -FilePath $PYTHON -ArgumentList "client_monitor.py" -WorkingDirectory $CLIENT_DIR -WindowStyle Hidden -PassThru -RedirectStandardOutput $LOG_FILE -RedirectStandardError $LOG_FILE
    
    # 保存 PID
    $process.Id | Out-File $PID_FILE
    
    Write-Host "等待客户端启动..." -NoNewline
    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline
        
        # 检查是否注册成功
        if (Test-Path $LOG_FILE) {
            $content = Get-Content $LOG_FILE -Raw
            if ($content -match "客户端注册成功") {
                Write-Host ""
                Write-Color "✅ 客户端启动成功!" Green
                Write-Host ""
                Write-Color "客户端信息:" Cyan
                
                # 提取客户端ID
                if ($content -match "客户端注册成功.*([a-f0-9]{12})") {
                    Write-Host "  客户端ID: $($matches[1])"
                }
                
                Write-Host "  PID: $($process.Id)"
                Write-Host "  日志文件: $LOG_FILE"
                Write-Host ""
                Write-Color "查看日志: .\start_client.ps1 log" Cyan
                return
            }
            
            if ($content -match "注册失败|ERROR") {
                Write-Host ""
                Write-Color "❌ 客户端注册失败" Red
                Write-Host "查看日志: $LOG_FILE"
                return
            }
        }
    }
    
    Write-Host ""
    Write-Color "⚠ 客户端启动超时，请检查日志" Yellow
    Write-Host "查看日志: $LOG_FILE"
}

# 停止客户端
function Stop-Client {
    $pid = Get-ClientPid
    if (-not $pid) {
        Write-Color "客户端未运行" Yellow
        # 清理残留进程
        Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*client_monitor.py*" } | Stop-Process -Force
        return
    }
    
    Write-Color "正在停止客户端 (PID: $pid)..." Cyan
    
    # 尝试优雅停止
    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($process) {
        $process.CloseMainWindow() | Out-Null
        Start-Sleep -Seconds 2
        
        if (-not $process.HasExited) {
            Stop-Process -Id $pid -Force
        }
    }
    
    if (Test-Path $PID_FILE) {
        Remove-Item $PID_FILE
    }
    
    Write-Color "✅ 客户端已停止" Green
}

# 重启客户端
function Restart-Client {
    Write-Color "重启客户端..." Cyan
    Stop-Client
    Start-Sleep -Seconds 2
    Start-Client
}

# 查看状态
function Show-Status {
    Write-Color "=== 客户端状态 ===" Cyan
    Write-Host ""
    
    $pid = Get-ClientPid
    if ($pid) {
        Write-Color "状态: 运行中" Green
        Write-Host "PID: $pid"
        
        # 获取进程信息
        $process = Get-Process -Id $pid
        Write-Host "启动时间: $($process.StartTime)"
        Write-Host "运行时长: $([DateTime]::Now - $process.StartTime)"
        
        # 从日志提取信息
        if (Test-Path $LOG_FILE) {
            $content = Get-Content $LOG_FILE -Raw
            
            # 客户端ID
            if ($content -match "客户端注册成功.*([a-f0-9]{12})") {
                Write-Host ""
                Write-Host "客户端ID: $($matches[1])"
            }
            
            # 监控目录
            if ($content -match "正在监控目录:\s*(.+)") {
                Write-Host "监控目录: $($matches[1])"
            }
            
            # 上传统计
            $uploaded = (Select-String -Path $LOG_FILE -Pattern "上传成功" -AllMatches).Matches.Count
            if ($uploaded -gt 0) {
                Write-Host "上传成功: $uploaded 个文件"
            }
        }
        
        # 测试服务端连接
        Write-Host ""
        Write-Host "服务端连接: " -NoNewline
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8004/health" -UseBasicParsing -TimeoutSec 5
            Write-Color "正常" Green
        } catch {
            Write-Color "无法连接" Red
        }
    } else {
        Write-Color "状态: 未运行" Red
    }
    
    # 日志文件大小
    if (Test-Path $LOG_FILE) {
        Write-Host ""
        $size = (Get-Item $LOG_FILE).Length
        Write-Host "日志文件: $LOG_FILE ($([math]::Round($size/1KB, 2)) KB)"
    }
}

# 查看日志
function Show-Logs {
    if (-not (Test-Path $LOG_FILE)) {
        Write-Color "日志文件不存在" Yellow
        return
    }
    
    Write-Color "正在监听日志: $LOG_FILE (按 Ctrl+C 退出)..." Cyan
    Write-Host ""
    Get-Content $LOG_FILE -Wait
}

# 测试连接
function Test-Connection {
    Write-Color "=== 连接测试 ===" Cyan
    Write-Host ""
    
    # 测试服务端健康检查
    Write-Host "服务端健康检查: " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8004/health" -UseBasicParsing -TimeoutSec 5
        Write-Color "✓ 正常" Green
        Write-Host "  响应: $($response.Content)"
    } catch {
        Write-Color "✗ 失败" Red
    }
    
    # 测试客户端注册接口
    Write-Host ""
    Write-Host "客户端注册接口: " -NoNewline
    try {
        $body = @{client_name="test"} | ConvertTo-Json
        $response = Invoke-WebRequest -Uri "http://localhost:8004/api/client/register" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 5
        if ($response.Content -match "success") {
            Write-Color "✓ 正常" Green
        } else {
            Write-Color "✗ 失败" Red
        }
    } catch {
        Write-Color "✗ 失败" Red
    }
    
    # 测试客户端状态接口
    Write-Host ""
    Write-Host "客户端状态接口: " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8004/api/client/status" -UseBasicParsing -TimeoutSec 5
        $data = $response.Content | ConvertFrom-Json
        $count = if ($data.clients) { $data.clients.Count } else { 0 }
        Write-Color "✓ 正常" Green
        Write-Host "  $count 个客户端在线"
    } catch {
        Write-Color "✗ 失败" Red
    }
}

# 主逻辑
$command = $args[0]

switch ($command) {
    "start" { Start-Client }
    "stop" { Stop-Client }
    "restart" { Restart-Client }
    "status" { Show-Status }
    "log" { Show-Logs }
    "logs" { Show-Logs }
    "test" { Test-Connection }
    default { Show-Help }
}
