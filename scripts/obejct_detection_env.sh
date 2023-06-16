#!/bin/bash

#  assert correct run dir
run_dir="02514-Deep-Learning-In-Computer-Vision"
if ! [ "$(basename $PWD)" = $run_dir ];
then
    echo -e "\033[0;31mScript must be submitted from the directory: $run_dir\033[0m"
    exit 1
fi

### Go one level up to create venv
cd ..

### Set variables
env_name=DLCI-venv
python_ref=$PWD/$env_name/bin/python3
pip_ref=pip

### Make python environment
module unload python3
module unload cuda
module load python3/3.10.11 
module load cuda/11.7
python3 -m venv $env_name


### activate env
source $env_name/bin/activate

### update pip and package setup tools
$python_ref -E -m $pip_ref install --upgrade $pip_ref
$python_ref -E -m $pip_ref install --upgrade setuptools
$python_ref -E -m $pip_ref install --upgrade wheel

### install python requirements
$python_ref -m $pip_ref install -r 02514-Deep-Learning-In-Computer-Vision/requirements.txt
# $python_ref -m $pip_ref install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$python_ref -m $pip_ref install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

### Go back to run dir and deactivate env
cd $run_dir
deactivate
