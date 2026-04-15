@echo off
chcp 65001 >nul
REM ASD 客户端监控服务管理脚本 (Windows)
REM 支持: start | stop | restart | status
REM 兼容: Windows 7/8/10/11 及 Windows Server 2008+

setlocal enabledelayedexpansion

REM ============================================
REM 配置区域 - 可从.env文件或环境变量读取
REM ============================================
set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "PID_FILE=%CLIENT_DIR%\.client.pid"
set "LOG_FILE=%CLIENT_DIR%\client.log"
set "ENV_FILE=%CLIENT_DIR%\.env"

REM ============================================
REM 从.env文件加载配置 (如果存在)
REM ============================================
if exist "%ENV_FILE%" (
    REM 读取.env文件中的配置
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "KEY=%%a"
        set "VAL=%%b"
        REM 去除空格
        for /f "tokens=*" %%k in ("!KEY!") do set "KEY=%%k"
        for /f "tokens=*" %%v in ("!VAL!") do set "VAL=%%v"
        REM 设置到环境变量
        if "!KEY!"=="ASD_SERVER_URL" set "ENV_SERVER_URL=!VAL!"
        if "!KEY!"=="ASD_WS_URL" set "ENV_WS_URL=!VAL!"
        if "!KEY!"=="ASD_CLIENT_NAME" set "ENV_CLIENT_NAME=!VAL!"
        if "!KEY!"=="ASD_CLIENT_ID" set "ENV_CLIENT_ID=!VAL!"
        if "!KEY!"=="ASD_MONITOR_DIR" set "ENV_MONITOR_DIR=!VAL!"
        if "!KEY!"=="ASD_LOG_LEVEL" set "ENV_LOG_LEVEL=!VAL!"
        if "!KEY!"=="ASD_LOG_FILE" set "ENV_LOG_FILE=!VAL!"
        if "!KEY!"=="ASD_PYTHON" set "ENV_PYTHON=!VAL!"
    )
)

REM ============================================
REM 配置优先级: 环境变量 > .env文件 > 默认值
REM ============================================

REM Python配置
if defined ASD_PYTHON (
    set "PYTHON=!ASD_PYTHON!"
) else if defined ENV_PYTHON (
    set "PYTHON=!ENV_PYTHON!"
) else (
    REM 自动检测Python
    set "PYTHON="
    for %%P in (python python3 py) do (
        if not defined PYTHON (
            %%P --version >nul 2>&1 && set "PYTHON=%%P"
        )
    )
    if not defined PYTHON (
        echo [错误] 未找到Python，请安装Python或设置ASD_PYTHON环境变量
        pause
        exit /b 1
    )
)

REM 服务器地址配置
if defined ASD_SERVER_URL (
    set "SERVER_URL=!ASD_SERVER_URL!"
) else if defined ENV_SERVER_URL (
    set "SERVER_URL=!ENV_SERVER_URL!"
) else (
    set "SERVER_URL=http://localhost:8004"
)

REM WebSocket地址配置
if defined ASD_WS_URL (
    set "WS_URL=!ASD_WS_URL!"
) else if defined ENV_WS_URL (
    set "WS_URL=!ENV_WS_URL!"
) else (
    set "WS_URL=!SERVER_URL!"
)
REM 将http替换为ws
set "WS_URL=!WS_URL:http://=ws://!"
set "WS_URL=!WS_URL:https://=wss://!"

REM 监控目录配置
if defined ASD_MONITOR_DIR (
    set "MONITOR_DIR=!ASD_MONITOR_DIR!"
) else if defined ENV_MONITOR_DIR (
    set "MONITOR_DIR=!ENV_MONITOR_DIR!"
) else (
    set "MONITOR_DIR=%CLIENT_DIR%\monitor"
)

REM 客户端名称配置
if defined ASD_CLIENT_NAME (
    set "CLIENT_NAME=!ASD_CLIENT_NAME!"
) else if defined ENV_CLIENT_NAME (
    set "CLIENT_NAME=!ENV_CLIENT_NAME!"
) else (
    set "CLIENT_NAME=客户端-01"
)

REM 日志级别配置
if defined ASD_LOG_LEVEL (
    set "LOG_LEVEL=!ASD_LOG_LEVEL!"
) else if defined ENV_LOG_LEVEL (
    set "LOG_LEVEL=!ENV_LOG_LEVEL!"
) else (
    set "LOG_LEVEL=INFO"
)

REM ============================================
REM 颜色设置 - 兼容旧版Windows
REM ============================================
set "USE_COLOR=0"
for /f "tokens=2 delims=[]" %%a in ('ver') do set "WIN_VER=%%a"
echo %WIN_VER% | findstr "10\." >nul && set "USE_COLOR=1"
echo %WIN_VER% | findstr "6\." >nul && set "USE_COLOR=1"

if %USE_COLOR%==1 (
    set "RED=[31m"
    set "GREEN=[32m"
    set "YELLOW=[33m"
    set "BLUE=[34m"
    set "NC=[0m"
) else (
    set "RED="
    set "GREEN="
    set "YELLOW="
    set "BLUE="
    set "NC="
)

REM ============================================
REM 主逻辑
REM ============================================

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
echo   setup      检查环境并安装依赖
echo.
echo 环境变量:
echo   ASD_PYTHON      Python解释器路径 (默认: 自动检测)
echo   ASD_SERVER_URL  服务器地址 (默认: http://localhost:8004)
echo   ASD_MONITOR_DIR 监控目录 (默认: .\monitor)
echo.
echo 示例:
echo   set ASD_SERVER_URL=http://192.168.1.100:8004
echo   start_client.bat start
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
if "%~1"=="setup" goto :setup_env
echo 未知命令: %~1
goto :show_help

REM ============================================
REM 环境检查和设置
REM ============================================
:setup_env
echo === 环境检查 ===
echo.

REM 显示配置来源
echo [配置信息]
if exist "%ENV_FILE%" (
    echo   配置文件: %ENV_FILE% (已加载)
) else (
    echo   配置文件: 未找到 (使用默认配置)
)
echo   服务器地址: %SERVER_URL%
echo   WebSocket:  %WS_URL%
echo   监控目录:   %MONITOR_DIR%
echo   客户端名:   %CLIENT_NAME%
echo   日志级别:   %LOG_LEVEL%
echo.

echo Python版本:
%PYTHON% --version 2>nul || (
    echo [错误] Python未找到: %PYTHON%
    exit /b 1
)
echo.

echo 检查依赖...
%PYTHON% -c "import httpx, websockets, watchdog" 2>nul && (
    echo [OK] 所有依赖已安装
) || (
    echo [提示] 正在安装依赖...
    %PYTHON% -m pip install -r "%CLIENT_DIR%\requirements.txt"
)
echo.

echo 创建监控目录...
if not exist "%MONITOR_DIR%" (
    mkdir "%MONITOR_DIR%"
    echo [OK] 已创建: %MONITOR_DIR%
) else (
    echo [OK] 目录已存在: %MONITOR_DIR%
)
echo.

echo 检查服务器连接 (最多等待10秒)...
call :check_server
echo.

echo === 环境检查完成 ===
echo.
echo 提示: 如果服务器连接失败，请检查:
echo   1. 服务器地址是否正确 (当前: %SERVER_URL%)
echo   2. 服务端是否已启动
echo   3. 防火墙是否允许连接
echo   4. 网络连接是否正常
pause
goto :eof

REM ============================================
REM 启动客户端
REM ============================================
:start_client
call :get_pid
if not "!PID!"=="" (
    echo 客户端已在运行中 (PID: !PID!)
    goto :eof
)

echo 正在启动 ASD 客户端监控...
echo ============================================
echo Python:     %PYTHON%
echo 服务器:     %SERVER_URL%
echo WebSocket:  %WS_URL%
echo 监控目录:   %MONITOR_DIR%
echo 客户端名:   %CLIENT_NAME%
echo 日志级别:   %LOG_LEVEL%
echo ============================================
cd /d "%CLIENT_DIR%"

REM 检查主程序是否存在
if not exist "client_monitor.py" (
    echo [错误] 未找到 client_monitor.py
    pause
    exit /b 1
)

REM 清空旧日志
if exist "%LOG_FILE%" (
    echo. > "%LOG_FILE%"
)

REM 创建监控目录
if not exist "%MONITOR_DIR%" (
    mkdir "%MONITOR_DIR%"
)

REM 设置环境变量并启动客户端 (传递所有配置给Python程序)
set "ASD_SERVER_URL=%SERVER_URL%"
set "ASD_WS_URL=%WS_URL%"
set "ASD_MONITOR_DIR=%MONITOR_DIR%"
set "ASD_CLIENT_NAME=%CLIENT_NAME%"
set "ASD_LOG_LEVEL=%LOG_LEVEL%"
set "ASD_LOG_FILE=%LOG_FILE%"

REM 如果.env中有CLIENT_ID也传递
if defined ENV_CLIENT_ID (
    set "ASD_CLIENT_ID=!ENV_CLIENT_ID!"
)

REM 启动客户端（后台运行）
echo 正在启动Python程序...

REM 先创建一个空的日志文件确保路径正确
echo. > "%LOG_FILE%" 2>nul

REM 使用start /B命令启动，重定向到日志文件
REM 注意：Windows批处理中start /B的重定向需要特殊处理
start /B /MIN cmd /C "%PYTHON% client_monitor.py > "%LOG_FILE%" 2>&1"

REM 记录启动时间戳
set "START_TIME=%TIME%"
echo 启动时间: %START_TIME%

REM 等待进程启动
timeout /t 2 /nobreak >nul

REM 获取PID - 通过查找最近启动的python进程
set "PID="
set "FOUND=0"

REM 方法1: 使用wmic获取最近启动的python进程
for /f "skip=1 tokens=2 delims=," %%a in ('wmic process where "name='python.exe' or name='python3.exe'" get ProcessId,CommandLine /format:csv 2^>nul ^| findstr "client_monitor.py"') do (
    if !FOUND!==0 (
        set "PID=%%a"
        set "FOUND=1"
    )
)

REM 方法2: 如果wmic失败，尝试从tasklist查找
if not defined PID (
    for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" /fo list ^| findstr /i "PID:"') do (
        REM 验证这个PID是否是我们的进程（通过检查日志文件是否被写入）
        timeout /t 1 /nobreak >nul
        findstr /c:"客户端初始化" "%LOG_FILE%" >nul 2>&1
        if !errorlevel!==0 (
            set "PID=%%a"
            goto :pid_found
        )
    )
)

:pid_found
if defined PID (
    echo !PID! > "%PID_FILE%"
    echo 检测到进程PID: !PID!
) else (
    echo [警告] 无法获取进程PID，但程序可能仍在运行
)

echo.
echo 等待客户端初始化...
timeout /t 3 /nobreak >nul

REM 检查日志输出
echo.
echo [最近日志输出]:
REM 使用PowerShell显示最后20行（兼容Windows）
powershell -Command "if (Test-Path '%LOG_FILE%') { Get-Content '%LOG_FILE%' -Tail 20 } else { Write-Host '日志文件不存在' }"

REM 检查是否注册成功
findstr /c:"客户端注册成功" "%LOG_FILE%" >nul 2>&1
if %errorlevel%==0 (
    echo.
    echo ============================================
    echo  客户端启动成功!
    echo ============================================
    if defined PID echo  PID: !PID!
    echo  日志文件: %LOG_FILE%
    echo.
    echo  查看日志: start_client.bat log
    echo  查看状态: start_client.bat status
) else (
    findstr /c:"ERROR" "%LOG_FILE%" >nul 2>&1
    if %errorlevel%==0 (
        echo.
        echo [错误] 客户端启动出错，请查看日志:
        type "%LOG_FILE%"
    ) else (
        findstr /c:"客户端初始化" "%LOG_FILE%" >nul 2>&1
        if %errorlevel%==0 (
            echo.
            echo [提示] 客户端正在初始化中...
            echo  查看日志: start_client.bat log
            echo  等待几秒后查看状态: start_client.bat status
        ) else (
            echo.
            echo [错误] 客户端可能未能正常启动
            echo  请检查:
            echo    1. Python是否正常工作: %PYTHON% --version
            echo    2. 依赖是否安装: %PYTHON% -c "import httpx, websockets, watchdog"
            echo    3. 查看详细日志: type "%LOG_FILE%"
        )
    )
)
goto :eof

REM ============================================
REM 停止客户端
REM ============================================
:stop_client
call :get_pid
if "!PID!"=="" (
    echo 客户端未运行
    REM 清理残留
    taskkill /f /im python.exe 2>nul
    taskkill /f /im python3.exe 2>nul
    taskkill /f /im py.exe 2>nul
    if exist "%PID_FILE%" del "%PID_FILE%"
    goto :eof
)

echo 正在停止客户端 (PID: !PID!)...
taskkill /pid !PID! /f 2>nul
if exist "%PID_FILE%" del "%PID_FILE%"
echo 客户端已停止
goto :eof

REM ============================================
REM 重启客户端
REM ============================================
:restart_client
call :stop_client
timeout /t 2 /nobreak >nul
call :start_client
goto :eof

REM ============================================
REM 查看状态
REM ============================================
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
    call :check_server
)

if exist "%LOG_FILE%" (
    echo.
    for %%i in ("%LOG_FILE%") do echo 日志文件: %%~nxi (%%~zi bytes)
)
goto :eof

REM ============================================
REM 查看日志
REM ============================================
:show_logs
if not exist "%LOG_FILE%" (
    echo 日志文件不存在
    goto :eof
)
echo 正在监听日志: %LOG_FILE%
echo 按 Ctrl+C 退出...
type "%LOG_FILE%"
goto :eof

REM ============================================
REM 测试连接
REM ============================================
:test_connection
echo === 连接测试 ===
echo.
echo 服务器地址: %SERVER_URL%
echo.

echo 服务端健康检查...
call :check_server

echo.
echo 客户端注册接口测试...
call :http_request "%SERVER_URL%/api/client/register" "POST" "{\"client_name\":\"test\"}"
goto :eof

REM ============================================
REM 检查服务器连接 (兼容多种方式，带超时)
REM ============================================
:check_server
REM 尝试多种方式检测服务器
set "SERVER_OK=0"
set "CHECK_TIMEOUT=8"

echo   正在检测 %SERVER_URL%/health ...

REM 方式1: PowerShell (Windows 7+) - 使用Start-Process实现超时控制
set "PS_TIMEOUT=0"
timeout /t 1 /nobreak >nul

REM 创建临时脚本文件
echo try { Invoke-WebRequest -Uri '%SERVER_URL%/health' -TimeoutSec 5 -UseBasicParsing ^| Out-Null; exit 0 } catch { exit 1 } > "%TEMP%\asd_check.ps1"

REM 使用Start-Process启动PowerShell并等待，带超时
start /B /MIN powershell -ExecutionPolicy Bypass -File "%TEMP%\asd_check.ps1" >nul 2>&1
set "PS_PID=!ERRORLEVEL!"

REM 等待最多6秒
for /l %%i in (1,1,6) do (
    timeout /t 1 /nobreak >nul
    tasklist /fi "imagename eq powershell.exe" /fo csv 2>nul | findstr /i "asd_check" >nul
    if errorlevel 1 (
        REM 进程已结束，检查退出码
        set "PS_TIMEOUT=0"
        goto :ps_done
    )
)
set "PS_TIMEOUT=1"
taskkill /f /im powershell.exe 2>nul

:ps_done
if %PS_TIMEOUT%==0 (
    REM 检查是否成功
    powershell -Command "try { Invoke-WebRequest -Uri '%SERVER_URL%/health' -TimeoutSec 3 -UseBasicParsing ^| Out-Null; Write-Host 'OK'; exit 0 } catch { exit 1 }" 2>nul | findstr "OK" >nul
    if !errorlevel!==0 (
        echo 服务端连接: 正常
        set "SERVER_OK=1"
        del "%TEMP%\asd_check.ps1" 2>nul
        goto :check_done
    )
)
del "%TEMP%\asd_check.ps1" 2>nul

REM 方式2: 使用bitsadmin (Windows XP+) - 自带超时
echo   尝试使用 bitsadmin 检测...
bitsadmin /transfer asd_test /download /priority normal "%SERVER_URL%/health" "%TEMP%\health_test.tmp" >nul 2>&1
if %errorlevel%==0 (
    if exist "%TEMP%\health_test.tmp" (
        echo 服务端连接: 正常
        set "SERVER_OK=1"
        del "%TEMP%\health_test.tmp" 2>nul
        goto :check_done
    )
)
del "%TEMP%\health_test.tmp" 2>nul

REM 方式3: 使用certutil - 快速超时
echo   尝试使用 certutil 检测...
certutil -urlcache -split -f "%SERVER_URL%/health" "%TEMP%\health_test.tmp" >nul 2>&1
timeout /t 2 /nobreak >nul
if %errorlevel%==0 (
    if exist "%TEMP%\health_test.tmp" (
        for %%F in ("%TEMP%\health_test.tmp") do set "FSIZE=%%~zF"
        if !FSIZE! gtr 0 (
            echo 服务端连接: 正常
            set "SERVER_OK=1"
            del "%TEMP%\health_test.tmp" 2>nul
            goto :check_done
        )
    )
)
del "%TEMP%\health_test.tmp" 2>nul

REM 方式4: 使用ping检测主机是否可达
echo   尝试 ping 检测主机...
for /f "tokens=2 delims=/:" %%a in ("%SERVER_URL%") do set "HOST=%%a"
for /f "tokens=1 delims=:" %%a in ("%HOST%") do set "HOST=%%a"
ping -n 1 -w 2000 !HOST! >nul 2>&1
if %errorlevel%==0 (
    echo 服务端连接: 主机可达，但HTTP服务可能未启动
    echo   请检查: 1) 服务端是否运行  2) 端口是否正确  3) 防火墙设置
) else (
    echo 服务端连接: 无法连接 (主机不可达)
    echo   请检查: 1) 服务器地址是否正确  2) 网络连接  3) 防火墙设置
)

:check_done
goto :eof

REM ============================================
REM HTTP请求 (兼容多种方式)
REM ============================================
:http_request
set "URL=%~1"
set "METHOD=%~2"
set "DATA=%~3"

REM 尝试PowerShell
powershell -Command "$url='%URL%'; $method='%METHOD%'; $body='%DATA%'; try { $r=Invoke-WebRequest -Uri $url -Method $method -Body $body -ContentType 'application/json' -TimeoutSec 5 -UseBasicParsing; Write-Host '请求成功' } catch { Write-Host '请求失败' }" 2>nul
if %errorlevel%==0 goto :eof

echo 无法执行HTTP请求 (PowerShell不可用)
goto :eof

REM ============================================
REM 获取PID
REM ============================================
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
