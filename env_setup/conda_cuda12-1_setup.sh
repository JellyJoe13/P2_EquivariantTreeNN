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
conda create -n "P2" python==3.9.* --yes

echo "+ Activating environment"
conda activate P2

echo "+ Installing basic libraries..."
conda install -y numpy pandas

echo "+ Installing pytorch..."
conda install -y pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch -c nvidia

echo "+ Installing advanced libraries..."
conda install -y matplotlib scikit-learn jupyterlab
