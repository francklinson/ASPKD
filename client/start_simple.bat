@echo off
chcp 65001 >nul
REM ASD 客户端简化启动脚本

title ASD Client
setlocal enabledelayedexpansion

set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "ENV_FILE=%CLIENT_DIR%\.env"

REM 加载.env配置
if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "KEY=%%a"
        set "VAL=%%b"
        for /f "tokens=*" %%k in ("!KEY!") do set "KEY=%%k"
        for /f "tokens=*" %%v in ("!VAL!") do set "VAL=%%v"
        if "!KEY!"=="ASD_SERVER_URL" set "SERVER_URL=!VAL!"
        if "!KEY!"=="ASD_MONITOR_DIR" set "MONITOR_DIR=!VAL!"
        if "!KEY!"=="ASD_CLIENT_NAME" set "CLIENT_NAME=!VAL!"
    )
)

if not defined SERVER_URL set "SERVER_URL=http://localhost:8004"
if not defined MONITOR_DIR set "MONITOR_DIR=%CLIENT_DIR%\monitor"
if not defined CLIENT_NAME set "CLIENT_NAME=客户端-01"

set "LOG_FILE=%CLIENT_DIR%\client.log"

echo ============================================
echo    ASD 客户端启动
echo ============================================
echo 服务器: %SERVER_URL%
echo 监控目录: %MONITOR_DIR%
echo 日志文件: %LOG_FILE%
echo ============================================
echo.

REM 设置环境变量
set "ASD_SERVER_URL=%SERVER_URL%"
set "ASD_MONITOR_DIR=%MONITOR_DIR%"
set "ASD_CLIENT_NAME=%CLIENT_NAME%"
set "ASD_LOG_FILE=%LOG_FILE%"

cd /d "%CLIENT_DIR%"

REM 启动客户端（直接运行，不后台）
echo 正在启动客户端...
echo 按 Ctrl+C 停止
echo.

python client_monitor.py

echo.
echo 客户端已退出
echo.
pause
