@echo off
chcp 65001 >nul
REM ASD 客户端连接诊断工具
REM 用于排查连接问题

title ASD 客户端连接诊断
setlocal enabledelayedexpansion

set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"
set "ENV_FILE=%CLIENT_DIR%\.env"

REM ============================================
REM 从.env文件加载配置
REM ============================================
echo 正在读取配置...
if exist "%ENV_FILE%" (
    echo   找到配置文件: %ENV_FILE%
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ENV_FILE%") do (
        set "KEY=%%a"
        set "VAL=%%b"
        REM 去除空格
        for /f "tokens=*" %%k in ("!KEY!") do set "KEY=%%k"
        for /f "tokens=*" %%v in ("!VAL!") do set "VAL=%%v"
        REM 读取配置
        if "!KEY!"=="ASD_SERVER_URL" set "SERVER_URL=!VAL!"
        if "!KEY!"=="ASD_MONITOR_DIR" set "MONITOR_DIR=!VAL!"
        if "!KEY!"=="ASD_CLIENT_NAME" set "CLIENT_NAME=!VAL!"
    )
) else (
    echo   未找到配置文件，使用默认设置
)

REM 设置默认值
if not defined SERVER_URL set "SERVER_URL=http://localhost:8004"
if not defined MONITOR_DIR set "MONITOR_DIR=%CLIENT_DIR%\monitor"
if not defined CLIENT_NAME set "CLIENT_NAME=客户端-01"

cls
echo ============================================
echo    ASD 客户端 - 连接诊断工具
echo ============================================
echo.
echo [配置信息]
echo   服务器地址: %SERVER_URL%
echo   监控目录:   %MONITOR_DIR%
echo   客户端名:   %CLIENT_NAME%
echo.

REM ============================================
REM 步骤1: 解析URL
REM ============================================
echo [步骤 1/5] 解析服务器地址...
for /f "tokens=2 delims=/:" %%a in ("%SERVER_URL%") do set "HOST=%%a"
for /f "tokens=1 delims=:" %%a in ("%HOST%") do set "HOST=%%a"

for /f "tokens=3 delims=/:" %%a in ("%SERVER_URL%") do (
    if "%%a"=="" (
        set "PORT=80"
    ) else (
        for /f "tokens=1 delims=/" %%b in ("%%a") do set "PORT=%%b"
    )
)

echo   主机: %HOST%
echo   端口: %PORT%
echo   [OK]
echo.

REM ============================================
REM 步骤2: Ping测试
REM ============================================
echo [步骤 2/5] Ping 测试 (检测主机是否可达)...
ping -n 2 -w 3000 %HOST% >nul 2>&1
if %errorlevel%==0 (
    echo   [OK] 主机 %HOST% 可以访问
) else (
    echo   [失败] 无法 ping 通 %HOST%
    echo   可能原因:
    echo     - 服务器关机或网络断开
    echo     - 服务器禁用了 ping 响应
    echo     - 防火墙阻止了 ICMP
)
echo.

REM ============================================
REM 步骤3: 端口检测
REM ============================================
echo [步骤 3/5] 端口连通性测试...
echo   尝试连接 %HOST%:%PORT%...

REM 使用PowerShell测试端口
timeout /t 1 /nobreak >nul
powershell -Command "$t=New-Object Net.Sockets.TcpClient; try { $t.Connect('%HOST%', %PORT%); Write-Host '[OK] 端口 %PORT% 已开放'; exit 0 } catch { Write-Host '[失败] 端口 %PORT% 无法连接'; exit 1 }" 2>nul

echo.

REM ============================================
REM 步骤4: HTTP请求测试
REM ============================================
echo [步骤 4/5] HTTP 请求测试...
echo   请求: %SERVER_URL%/health
echo   超时: 5秒
echo.

REM 使用PowerShell发送HTTP请求，带超时
echo try { $r=Invoke-WebRequest -Uri '%SERVER_URL%/health' -TimeoutSec 5 -UseBasicParsing; Write-Host "[OK] HTTP 响应状态:" $r.StatusCode; Write-Host "响应内容:" $r.Content } catch { Write-Host "[失败] 请求异常:" $_.Exception.Message } > "%TEMP%\asd_diag.ps1"

powershell -ExecutionPolicy Bypass -File "%TEMP%\asd_diag.ps1" 2>nul
del "%TEMP%\asd_diag.ps1" 2>nul

echo.

REM ============================================
REM 步骤5: 检查本地环境
REM ============================================
echo [步骤 5/5] 本地环境检查...

REM Python检查
echo   Python:
python --version 2>nul && echo     [OK] 已安装 || echo     [失败] 未安装

REM 依赖检查
echo   Python依赖:
python -c "import httpx" 2>nul && echo     [OK] httpx 已安装 || echo     [失败] httpx 未安装
python -c "import websockets" 2>nul && echo     [OK] websockets 已安装 || echo     [失败] websockets 未安装
python -c "import watchdog" 2>nul && echo     [OK] watchdog 已安装 || echo     [失败] watchdog 未安装

echo.

REM ============================================
REM 诊断结果
REM ============================================
echo ============================================
echo    诊断完成
echo ============================================
echo.
echo 常见问题及解决方案:
echo.
echo 1. 如果 Ping 失败但端口测试成功:
echo    - 服务器可能禁用了 ping，但服务正常运行
echo    - 可以继续尝试启动客户端
echo.
echo 2. 如果端口测试失败:
echo    - 检查服务器是否已启动服务端程序
echo    - 检查服务器防火墙是否开放 %PORT% 端口
echo    - 检查服务器地址和端口是否正确
echo.
echo 3. 如果 HTTP 测试失败:
echo    - 服务端可能没有正确运行
echo    - 检查服务端日志是否有错误
echo    - 确认服务端监听的是 0.0.0.0 而不仅是 localhost
echo.
echo 4. 如果 Python 依赖缺失:
echo    运行: pip install -r requirements.txt
echo.

choice /C YN /M "是否再次测试"
if errorlevel 2 goto :end
if errorlevel 1 goto :start

:end
echo.
echo 按任意键退出...
pause >nul
