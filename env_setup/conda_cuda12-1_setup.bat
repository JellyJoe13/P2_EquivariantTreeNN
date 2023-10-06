@echo off

where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo conda could not be found on this machine. Terminating script.
    exit /b
)

set /p answer="Conda found, proceeding with environment setup? (y/n) "
if /i !"%answer%"=="y" (
    echo Halting script.
    exit /b
)

echo Proceeding installation...
echo + Creation of environment...
call conda create -n "P2" python==3.9.* --yes

echo + Activating environment
call conda activate P2

echo + Installing basic libraries...
call conda install -y numpy pandas

echo + Installing pytorch...
call conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo + Installing advanced libraries...
call conda install -y matplotlib scikit-learn jupyterlab