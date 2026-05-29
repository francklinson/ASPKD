@echo off
chcp 65001 >nul
REM ASD 客户端简化启动脚本（后台运行版）
REM 类似 debug_client.bat 但支持后台运行

title ASD Client
setlocal enabledelayedexpansion

REM ============================================
REM 基础配置
REM ============================================
set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "ENV_FILE=%CLIENT_DIR%\.env"
set "LOG_FILE=%CLIENT_DIR%\client.log"
set "PID_FILE=%CLIENT_DIR%\.client.pid"

REM ============================================
REM 从.env文件加载配置
REM ============================================
set "SERVER_URL=http://localhost:8004"
set "WS_URL=ws://localhost:8004"
set "MONITOR_DIR=%CLIENT_DIR%\monitor"
set "CLIENT_NAME=客户端-01"
set "LOG_LEVEL=INFO"

if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "KEY=%%a"
        set "VAL=%%b"
        for /f "tokens=*" %%k in ("!KEY!") do set "KEY=%%k"
        for /f "tokens=*" %%v in ("!VAL!") do set "VAL=%%v"
        set "VAL=!VAL:\r=!"
        
        if "!KEY!"=="ASD_SERVER_URL" set "SERVER_URL=!VAL!"
        if "!KEY!"=="ASD_WS_URL" set "WS_URL=!VAL!"
        if "!KEY!"=="ASD_MONITOR_DIR" set "MONITOR_DIR=!VAL!"
        if "!KEY!"=="ASD_CLIENT_NAME" set "CLIENT_NAME=!VAL!"
        if "!KEY!"=="ASD_LOG_LEVEL" set "LOG_LEVEL=!VAL!"
    )
)

REM ============================================
REM 自动检测Python
REM ============================================
set "PYTHON="
for %%P in (python python3 py) do (
    if not defined PYTHON (
        %%P --version >nul 2>&1 && set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    echo [错误] 未找到Python!
    pause
    exit /b 1
)

REM ============================================
REM 显示配置信息
REM ============================================
echo ============================================
echo    ASD 客户端启动
echo ============================================
echo Python:     %PYTHON%
echo 服务器:     %SERVER_URL%
echo WebSocket:  %WS_URL%
echo 监控目录:   %MONITOR_DIR%
echo 客户端名:   %CLIENT_NAME%
echo 日志级别:   %LOG_LEVEL%
echo ============================================
echo.

REM ============================================
REM 检查是否在运行
REM ============================================
if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    tasklist /fi "pid eq !OLD_PID!" 2>nul | findstr "!OLD_PID!" >nul
    if !errorlevel!==0 (
        echo [提示] 客户端已在运行中 (PID: !OLD_PID!)
        echo 查看日志: %LOG_FILE%
        pause
        exit /b 0
    )
)

REM ============================================
REM 创建监控目录
REM ============================================
if not exist "%MONITOR_DIR%" (
    mkdir "%MONITOR_DIR%"
)

REM ============================================
REM 清空旧日志
REM ============================================
if exist "%LOG_FILE%" (
    echo. > "%LOG_FILE%"
)

REM ============================================
REM 设置环境变量并启动
REM ============================================
cd /d "%CLIENT_DIR%"

set "ASD_SERVER_URL=%SERVER_URL%"
set "ASD_WS_URL=%WS_URL%"
set "ASD_MONITOR_DIR=%MONITOR_DIR%"
set "ASD_CLIENT_NAME=%CLIENT_NAME%"
set "ASD_LOG_LEVEL=%LOG_LEVEL%"
set "ASD_LOG_FILE=%LOG_FILE%"

echo 正在启动客户端...
echo.

REM 使用PowerShell在后台启动进程
powershell -WindowStyle Hidden -Command "
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = '%PYTHON%'
    $psi.Arguments = 'client_monitor.py'
    $psi.WorkingDirectory = '%CLIENT_DIR%'
    $psi.WindowStyle = 'Hidden'
    $psi.UseShellExecute = $false
    $psi.EnvironmentVariables['ASD_SERVER_URL'] = '%SERVER_URL%'
    $psi.EnvironmentVariables['ASD_WS_URL'] = '%WS_URL%'
    $psi.EnvironmentVariables['ASD_MONITOR_DIR'] = '%MONITOR_DIR%'
    $psi.EnvironmentVariables['ASD_CLIENT_NAME'] = '%CLIENT_NAME%'
    $psi.EnvironmentVariables['ASD_LOG_LEVEL'] = '%LOG_LEVEL%'
    $psi.EnvironmentVariables['ASD_LOG_FILE'] = '%LOG_FILE%'
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.Id | Out-File -FilePath '%PID_FILE%' -Encoding ASCII
"

REM 检查是否成功启动
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    echo [OK] 客户端已启动，PID: !PID!
    echo 日志文件: %LOG_FILE%
    echo.
    echo 查看日志命令:
    echo   type "%LOG_FILE%"
    echo.
    timeout /t 2 /nobreak >nul
) else (
    echo [错误] 启动失败，请运行 debug_client.bat 查看详细错误
    pause
    exit /b 1
)
