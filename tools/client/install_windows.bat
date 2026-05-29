@echo off
chcp 65001 >nul
REM ASD 客户端 Windows 安装脚本
REM 自动检查并安装Python依赖

title ASD 客户端安装
setlocal enabledelayedexpansion

set "CLIENT_DIR=%~dp0"
set "CLIENT_DIR=!CLIENT_DIR:~0,-1!"

echo ============================================
echo    ASD 客户端 - Windows 安装程序
echo ============================================
echo.

REM ============================================
REM 检查Python
REM ============================================
echo [1/4] 检查Python安装...

set "PYTHON="
set "PYTHON_VERSION="

REM 尝试多种方式查找Python
for %%P in (python python3 py) do (
    if not defined PYTHON (
        for /f "tokens=*" %%v in ('%%P --version 2^>nul') do (
            set "PYTHON=%%P"
            set "PYTHON_VERSION=%%v"
        )
    )
)

if not defined PYTHON (
    echo.
    echo [错误] 未检测到Python!
    echo.
    echo 请安装Python 3.8或更高版本:
    echo   1. 访问 https://www.python.org/downloads/
    echo   2. 下载并安装Python 3.8+
    echo   3. 安装时勾选 "Add Python to PATH"
    echo.
    echo 按任意键打开Python下载页面...
    pause >nul
    start https://www.python.org/downloads/
    exit /b 1
)

echo [OK] 找到Python: %PYTHON_VERSION%
echo.

REM ============================================
REM 检查pip
REM ============================================
echo [2/4] 检查pip...

%PYTHON% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [警告] pip未安装，尝试安装...
    %PYTHON% -m ensurepip --upgrade
    if errorlevel 1 (
        echo [错误] 无法安装pip
        pause
        exit /b 1
    )
)

echo [OK] pip已安装
echo.

REM ============================================
REM 安装依赖
REM ============================================
echo [3/4] 安装依赖...
echo.

cd /d "%CLIENT_DIR%"

REM 升级pip
%PYTHON% -m pip install --upgrade pip

echo.
echo 安装项目依赖...
%PYTHON% -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)

echo.
echo [OK] 依赖安装完成
echo.

REM ============================================
REM 创建配置
REM ============================================
echo [4/4] 创建配置文件...

if not exist "%CLIENT_DIR%\.env" (
    if exist "%CLIENT_DIR%\.env.example" (
        copy "%CLIENT_DIR%\.env.example" "%CLIENT_DIR%\.env" >nul
        echo [OK] 已创建默认配置文件: .env
        echo     请编辑 .env 文件配置服务器地址
    ) else (
        echo [警告] 未找到 .env.example 文件
    )
) else (
    echo [OK] 配置文件已存在: .env
)
echo.

REM 创建监控目录
if not exist "%CLIENT_DIR%\monitor" (
    mkdir "%CLIENT_DIR%\monitor"
    echo [OK] 已创建监控目录: monitor\
) else (
    echo [OK] 监控目录已存在: monitor\
)
echo.

REM ============================================
REM 完成
REM ============================================
echo ============================================
echo    安装完成!
echo ============================================
echo.
echo 使用说明:
echo   1. 编辑 .env 文件配置服务器地址
echo   2. 将音频文件放入 monitor 目录
echo   3. 运行 start_client.bat start 启动客户端
echo.
echo 快速测试:
echo   start_client.bat setup  - 检查环境
echo   start_client.bat test   - 测试连接
echo   start_client.bat start  - 启动客户端
echo.

choice /C YN /M "是否现在运行环境检查"
if errorlevel 2 goto :end
if errorlevel 1 call "%CLIENT_DIR%\start_client.bat" setup

:end
pause
