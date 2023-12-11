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
call conda create -n "P2" python==3.9.18 --yes

echo + Activating environment
call conda activate P2

echo + Installing basic libraries...
call conda install -y numpy=3.9.18 pandas=2.0.3

echo + Installing pytorch...
call conda install -y pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

echo + Installing advanced libraries...
call conda install -y matplotlib=3.7.2 scikit-learn=1.3.0 jupyterlab=3.6.3 multiprocess=0.70.15

echo + Installing hyperparameter framework and parallelization
call conda install -y optuna=3.4.0 plotly=5.9.0 multiprocess=0.70.15 tqdm=4.65.0

echo + Installing libraries for documentation generation
call conda install -y sphinx=7.2.6 sphinx-rtd-theme=1.3.0