@echo off
chcp 65001 >nul
REM ASD 客户端监控服务管理脚本 (Windows)
REM 支持: start | stop | restart | status

setlocal enabledelayedexpansion

REM 配置
set "CLIENT_DIR=%~dp0"
set "PID_FILE=%CLIENT_DIR%.client.pid"
set "LOG_FILE=%CLIENT_DIR%client.log"
set "PYTHON=python"

REM 颜色设置
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM 显示帮助
if "%~1"=="" goto :show_help
if "%~1"=="help" goto :show_help
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
goto :main

:show_help
echo ASD 客户端监控服务管理脚本 (Windows)
echo.
echo 用法: start_client.bat [命令]
echo.
echo 命令:
echo   start      启动客户端
echo   stop       停止客户端
echo   restart    重启客户端
echo   status     查看客户端状态
echo   log        查看实时日志
echo   test       测试与服务端的连接
echo.
pause
goto :eof

:main
if "%~1"=="start" goto :start_client
if "%~1"=="stop" goto :stop_client
if "%~1"=="restart" goto :restart_client
if "%~1"=="status" goto :show_status
if "%~1"=="log" goto :show_logs
if "%~1"=="logs" goto :show_logs
if "%~1"=="test" goto :test_connection
echo 未知命令: %~1
goto :show_help

:start_client
REM 检查是否已运行
call :get_pid
if not "!PID!"=="" (
    echo 客户端已在运行中 (PID: !PID!)
    goto :eof
)

echo 正在启动 ASD 客户端监控...
cd /d "%CLIENT_DIR%"

REM 清空旧日志
if exist "%LOG_FILE%" (
    echo. > "%LOG_FILE%"
)

REM 启动客户端（隐藏窗口）
start /B "ASD Client" %PYTHON% client_monitor.py > "%LOG_FILE%" 2>&1

REM 获取PID
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "PID:"') do (
    set "PID=%%a"
    echo !PID! > "%PID_FILE%"
    goto :started
)

:started
echo 等待客户端启动...
timeout /t 3 /nobreak >nul

REM 检查是否注册成功
findstr /c:"客户端注册成功" "%LOG_FILE%" >nul 2>&1
if %errorlevel%==0 (
    echo 客户端启动成功!
    echo PID: !PID!
    echo 日志文件: %LOG_FILE%
    echo.
    echo 查看日志: start_client.bat log
) else (
    echo 客户端启动失败或仍在初始化
    echo 查看日志: %LOG_FILE%
)
goto :eof

:stop_client
call :get_pid
if "!PID!"=="" (
    echo 客户端未运行
    REM 清理残留
    taskkill /f /im python.exe 2>nul
    if exist "%PID_FILE%" del "%PID_FILE%"
    goto :eof
)

echo 正在停止客户端 (PID: !PID!)...
taskkill /pid !PID! /f 2>nul
if exist "%PID_FILE%" del "%PID_FILE%"
echo 客户端已停止
goto :eof

:restart_client
call :stop_client
timeout /t 2 /nobreak >nul
call :start_client
goto :eof

:show_status
echo === 客户端状态 ===
echo.

call :get_pid
if "!PID!"=="" (
    echo 状态: 未运行
) else (
    echo 状态: 运行中
    echo PID: !PID!
    
    REM 从日志提取信息
    if exist "%LOG_FILE%" (
        findstr /c:"客户端注册成功" "%LOG_FILE%" >nul 2>&1
        if %errorlevel%==0 (
            echo 客户端已注册到服务端
        )
        
        for /f "delims=" %%i in ('findstr /c:"正在监控目录" "%LOG_FILE%" 2^>nul') do (
            echo %%i
        )
        
        for /f %%i in ('findstr /c:"上传成功" "%LOG_FILE%" 2^>nul ^| find /c /v ""') do (
            if %%i gtr 0 echo 上传成功: %%i 个文件
        )
    )
    
    REM 测试服务端连接
    echo.
    echo 服务端连接测试...
    curl -s http://localhost:8004/health >nul 2>&1
    if %errorlevel%==0 (
        echo 服务端连接: 正常
    ) else (
        echo 服务端连接: 无法连接
    )
)

if exist "%LOG_FILE%" (
    echo.
    for %%i in ("%LOG_FILE%") do echo 日志文件: %%~nxi (%%~zi bytes)
)
goto :eof

:show_logs
if not exist "%LOG_FILE%" (
    echo 日志文件不存在
    goto :eof
)
echo 正在监听日志: %LOG_FILE%
echo 按 Ctrl+C 退出...
type "%LOG_FILE%"
goto :eof

:test_connection
echo === 连接测试 ===
echo.

echo 服务端健康检查...
curl -s http://localhost:8004/health 2>nul
if %errorlevel%==0 (
    echo 正常
) else (
    echo 失败
)

echo.
echo 客户端注册接口测试...
curl -s -X POST http://localhost:8004/api/client/register -H "Content-Type: application/json" -d "{\"client_name\":\"test\"}" 2>nul
if %errorlevel%==0 (
    echo 正常
) else (
    echo 失败
)
goto :eof

:get_pid
set "PID="
if exist "%PID_FILE%" (
    set /p PID=<"%PID_FILE%"
    REM 验证PID是否有效
    tasklist /fi "pid eq !PID!" 2>nul | findstr "!PID!" >nul
    if errorlevel 1 (
        set "PID="
        del "%PID_FILE%" 2>nul
    )
)
goto :eof
