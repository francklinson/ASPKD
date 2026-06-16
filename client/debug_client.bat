@echo off
chcp 65001 >nul
REM ASD 客户端调试脚本
REM 用于排查启动问题


title ASD Client Debug
setlocal enabledelayedexpansion

echo ============================================
echo    ASD 客户端调试工具
echo ============================================
echo.

REM 获取脚本目录
set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "PROJECT_DIR=!CLIENT_DIR!\.."
set "ENV_FILE=%CLIENT_DIR%\.env"
set "LOG_FILE=%CLIENT_DIR%\client.log"

echo [目录信息]
echo   脚本目录: %CLIENT_DIR%
echo   项目目录: %PROJECT_DIR%
echo   配置文件: %ENV_FILE%
echo   日志文件: %LOG_FILE%
echo.

REM ============================================
REM 检查1: Python环境
REM ============================================
echo [检查1: Python环境]
echo.

set "PYTHON="

REM 尝试多种方式找到Python
for %%P in (python python3 py) do (
    if not defined PYTHON (
        echo   尝试: %%P
        %%P --version >nul 2>&1 && (
            echo   [OK] 找到: %%P
            set "PYTHON=%%P"
        ) || (
            echo   [X] 未找到: %%P
        )
    )
)

if not defined PYTHON (
    echo.
    echo [错误] 未找到Python!
    echo   请确保Python已安装并添加到PATH环境变量
    echo   或者设置 ASD_PYTHON 环境变量指向Python路径
    pause
    exit /b 1
)

echo.
echo   使用Python: %PYTHON%
%PYTHON% --version
echo.

REM ============================================
REM 检查2: 项目文件
REM ============================================
echo [检查2: 项目文件]
echo.

if exist "%CLIENT_DIR%\client_monitor.py" (
    echo   [OK] client_monitor.py 存在
) else (
    echo   [X] client_monitor.py 不存在!
    echo      路径: %CLIENT_DIR%\client_monitor.py
)

if exist "%CLIENT_DIR%\requirements.txt" (
    echo   [OK] requirements.txt 存在
) else (
    echo   [X] requirements.txt 不存在!
)

if exist "%ENV_FILE%" (
    echo   [OK] .env 配置文件存在
    echo.
    echo   [.env文件内容]
    type "%ENV_FILE%"
) else (
    echo   [警告] .env 配置文件不存在，将使用默认配置
)
echo.

REM ============================================
REM 检查3: Python依赖
REM ============================================
echo [检查3: Python依赖]
echo.

echo   检查 httpx...
%PYTHON% -c "import httpx; print('  [OK] httpx 版本:', httpx.__version__)" 2>nul || echo   [X] httpx 未安装

echo   检查 websockets...
%PYTHON% -c "import websockets; print('  [OK] websockets 版本:', websockets.__version__)" 2>nul || echo   [X] websockets 未安装

echo   检查 watchdog...
%PYTHON% -c "import watchdog; print('  [OK] watchdog 版本:', watchdog.__version__)" 2>nul || echo   [X] watchdog 未安装

echo   检查 asyncio...
%PYTHON% -c "import asyncio; print('  [OK] asyncio 可用')" 2>nul || echo   [X] asyncio 不可用

echo.

REM ============================================
REM 检查4: 服务器连接
REM ============================================
echo [检查4: 服务器连接]
echo.

REM 从.env读取服务器地址
set "SERVER_URL=http://localhost:8004"
if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        if "%%a"=="ASD_SERVER_URL" set "SERVER_URL=%%b"
    )
)

echo   服务器地址: %SERVER_URL%
echo   测试连接...

powershell -Command "try { $r = Invoke-WebRequest -Uri '%SERVER_URL%/health' -TimeoutSec 5 -UseBasicParsing; Write-Host '   [OK] 服务器连接成功'; Write-Host '   响应:' $r.Content } catch { Write-Host '   [X] 服务器连接失败:' $_.Exception.Message }" 2>nul

echo.

REM ============================================
REM 检查5: 监控目录
REM ============================================
echo [检查5: 监控目录]
echo.

REM 从.env读取监控目录
set "MONITOR_DIR=%CLIENT_DIR%\monitor"
if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        if "%%a"=="ASD_MONITOR_DIR" set "MONITOR_DIR=%%b"
    )
)

echo   监控目录: %MONITOR_DIR%

if exist "%MONITOR_DIR%" (
    echo   [OK] 监控目录已存在
) else (
    echo   [警告] 监控目录不存在，将尝试创建
    mkdir "%MONITOR_DIR%" 2>nul
    if exist "%MONITOR_DIR%" (
        echo   [OK] 监控目录创建成功
    ) else (
        echo   [X] 监控目录创建失败!
    )
)
echo.

REM ============================================
REM 检查6: 尝试直接运行Python脚本
REM ============================================
echo [检查6: 直接运行测试]
echo.
echo   这将尝试直接运行 client_monitor.py 并显示输出
echo   按 Ctrl+C 可以随时停止
echo.
echo ============================================
echo.

REM 设置环境变量
set "ASD_SERVER_URL=%SERVER_URL%"
set "ASD_MONITOR_DIR=%MONITOR_DIR%"
set "ASD_CLIENT_NAME=调试客户端"
set "ASD_LOG_LEVEL=DEBUG"
set "ASD_LOG_FILE=%LOG_FILE%"

echo [环境变量]
echo   ASD_SERVER_URL=%ASD_SERVER_URL%
echo   ASD_MONITOR_DIR=%ASD_MONITOR_DIR%
echo   ASD_CLIENT_NAME=%ASD_CLIENT_NAME%
echo   ASD_LOG_LEVEL=%ASD_LOG_LEVEL%
echo   ASD_LOG_FILE=%ASD_LOG_FILE%
echo.
echo [启动客户端...]
echo.

cd /d "%CLIENT_DIR%"

REM 直接运行Python脚本，显示所有输出
%PYTHON% client_monitor.py

echo.
echo ============================================
echo [运行结束]
echo.

REM 检查是否生成了日志
if exist "%LOG_FILE%" (
    echo [日志文件已生成]
    echo   路径: %LOG_FILE%
    echo   大小: 
    for %%F in ("%LOG_FILE%") do echo %%~zF bytes
    echo.
    echo [日志内容预览 - 最后20行]:
    powershell -Command "Get-Content '%LOG_FILE%' -Tail 20" 2>nul || type "%LOG_FILE%"
) else (
    echo [警告] 日志文件未生成!
    echo   预期路径: %LOG_FILE%
)

echo.
echo ============================================
echo    调试完成
echo ============================================
echo.
pause
