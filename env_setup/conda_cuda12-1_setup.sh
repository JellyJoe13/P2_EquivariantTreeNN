#!/bin/bash

# Check if conda is available on the machine
if ! command -v conda &> /dev/null
then
    echo "conda could not be found on this machine. Terminating script."
    exit 1
fi

# Prompt the user for confirmation
read -p "Conda found, proceeding with environment setup? (y/n) " answer
if [ "$answer" != "y" ]; then
    echo "Halting script."
    exit 0
fi

# Proceed with installation
echo "Proceeding installation..."

echo "+ Creation of environment..."
conda create -n "P2" python==3.9.18 --yes

echo "+ Activating environment"
conda activate P2

echo "+ Installing basic libraries..."
conda install -y numpy=3.9.18 pandas=2.0.3

echo "+ Installing pytorch..."
conda install -y pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia

echo "+ Installing advanced libraries..."
conda install -y matplotlib=3.7.2 scikit-learn=1.3.0 jupyterlab=3.6.3 multiprocess=0.70.15

echo "+ Installing hyperparameter framework and parallelization"
conda install -y optuna=3.4.0 plotly=5.9.0 multiprocess=0.70.15 tqdm=4.65.0

echo "+ Installing libraries for documentation generation"
call conda install -y sphinx=7.2.6 sphinx-rtd-theme=1.3.0