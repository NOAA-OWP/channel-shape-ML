#!/bin/bash
# Conda
source /home/arashmod/miniconda3/etc/profile.d/conda.sh
# source /home/arash.rad/anaconda3/etc/profile.d/conda.sh

conda activate base

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*ae-model.*" ; 

then
    echo "Environment found..."
else 
    echo "Creating new environment..."
    conda env create --file ae_env.yaml
fi

conda activate ae-model
echo "(ae-model) environment activated"
echo "Running scripts..."

# Main python training

start=`date +%s`
python3 autoencoder_extract.py > "AE_output.out"
end=`date +%s`
echo Execution time was `expr $end - $start` seconds.

# write file
output=output_bash.out  
ls > $output 