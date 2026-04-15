@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "PID_FILE=%CLIENT_DIR%\.client.pid"
set "LOG_FILE=%CLIENT_DIR%\client.log"
set "ENV_FILE=%CLIENT_DIR%\.env"

if "%~1"=="" goto :show_help
if "%~1"=="help" goto :show_help
goto :main

:show_help
echo ASD 客户端监控服务管理脚本
echo.
echo 用法: start_client.bat [命令]
echo.
echo 命令:
echo   start      启动客户端
echo   stop       停止客户端
echo   status     查看客户端状态
pause
goto :eof

:main
if "%~1"=="start" goto :start_client
if "%~1"=="stop" goto :stop_client
if "%~1"=="status" goto :show_status
echo 未知命令: %~1
goto :show_help

:start_client
echo 正在启动 ASD 客户端...
echo.

set "PYTHON=python"
for %%P in (python python3 py) do (
    if not defined PYTHON (
        %%P --version >nul 2>&1 && set "PYTHON=%%P"
    )
)

if not defined PYTHON (
    echo [错误] 未找到Python
    pause
    exit /b 1
)

echo Python: %PYTHON%
echo.

if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    tasklist /fi "pid eq !OLD_PID!" 2>nul | findstr "!OLD_PID!" >nul
    if !errorlevel!==0 (
        echo 客户端已在运行中 (PID: !OLD_PID!)
        goto :eof
    )
)

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
        if "!KEY!"=="ASD_SERVER_URL" set "ENV_SERVER_URL=!VAL!"
        if "!KEY!"=="ASD_WS_URL" set "ENV_WS_URL=!VAL!"
        if "!KEY!"=="ASD_MONITOR_DIR" set "ENV_MONITOR_DIR=!VAL!"
        if "!KEY!"=="ASD_CLIENT_NAME" set "ENV_CLIENT_NAME=!VAL!"
        if "!KEY!"=="ASD_LOG_LEVEL" set "ENV_LOG_LEVEL=!VAL!"
    )
)

set "SERVER_URL=!ENV_SERVER_URL!"
set "WS_URL=!ENV_WS_URL!"
set "MONITOR_DIR=!ENV_MONITOR_DIR!"
set "CLIENT_NAME=!ENV_CLIENT_NAME!"
set "LOG_LEVEL=!ENV_LOG_LEVEL!"

if not defined SERVER_URL set "SERVER_URL=http://localhost:8004"
if not defined WS_URL set "WS_URL=ws://localhost:8004"
if not defined MONITOR_DIR set "MONITOR_DIR=%CLIENT_DIR%\monitor"
if not defined CLIENT_NAME set "CLIENT_NAME=客户端-01"
if not defined LOG_LEVEL set "LOG_LEVEL=INFO"

powershell -Command "$env:ASD_LOG_LEVEL = '%LOG_LEVEL%'.Trim(); $env:ASD_SERVER_URL = '%SERVER_URL%'.Trim(); $env:ASD_WS_URL = '%WS_URL%'.Trim(); $env:ASD_MONITOR_DIR = '%MONITOR_DIR%'.Trim(); $env:ASD_CLIENT_NAME = '%CLIENT_NAME%'.Trim(); $env:ASD_LOG_FILE = '%LOG_FILE%'.Trim(); $env:PYTHONIOENCODING = 'utf-8'; Start-Process -FilePath '%PYTHON%' -ArgumentList 'client_monitor.py' -WorkingDirectory '%CLIENT_DIR%' -WindowStyle Minimized -PassThru | ForEach-Object { $_.Id | Out-File -FilePath '%PID_FILE%' -Encoding ASCII }"

if not exist "%MONITOR_DIR%" mkdir "%MONITOR_DIR%"
echo. > "%LOG_FILE%"

timeout /t 3 /nobreak >nul

set "PID="
for /f "skip=1 tokens=2 delims=," %%a in ('wmic process where "name='python.exe' or name='python3.exe'" get ProcessId,CommandLine /format:csv 2^>nul ^| findstr "client_monitor.py"') do (
    if not defined PID set "PID=%%a"
)

if defined PID (
    echo !PID! > "%PID_FILE%"
    echo [OK] 客户端已启动，PID: !PID!
) else (
    echo [提示] 进程可能已启动，但无法获取PID
)

echo 日志文件: %LOG_FILE%
echo.
echo 查看日志: type "%LOG_FILE%"
goto :eof

:stop_client
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    if defined PID (
        taskkill /pid !PID! /f 2>nul
        echo 已停止客户端
    )
    del "%PID_FILE%"
) else (
    echo 客户端未运行
)
taskkill /f /im python.exe 2>nul
taskkill /f /im python3.exe 2>nul
goto :eof

:show_status
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    tasklist /fi "pid eq !PID!" 2>nul | findstr "!PID!" >nul
    if !errorlevel!==0 (
        echo 状态: 运行中
        echo PID: !PID!
    ) else (
        echo 状态: 未运行
        del "%PID_FILE%"
    )
) else (
    echo 状态: 未运行
)
if exist "%LOG_FILE%" (
    for %%i in ("%LOG_FILE%") do echo 日志文件: %%~nxi (%%~zi bytes)
)
goto :eof
