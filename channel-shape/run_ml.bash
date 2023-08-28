#!/bin/bash
# Conda
# my directory source /home/arashmod/miniconda3/etc/profile.d/conda.sh
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 10000

if [ ! -d /home/arash.rad/Channel/conus-fhg/outputs ]; then
    echo "outputs floder not found, creating one..."
    mkdir -p /home/arash.rad/Channel/conus-fhg/outputs;
fi

source /home/arash.rad/anaconda3/etc/profile.d/conda.sh

conda activate base

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*ml-model.*" ; 

then
    echo "Environment found..."
else 
    echo "Creating new environment..."
    conda env create --file ml_env.yaml
fi

conda activate ml-model
echo "(ml-model) environment activated"
echo "Running scripts..."

# Main python training
usage() { echo "Usage: $0 [-c submission name <string>] [-n number of cores <int>] [-x transform x <True|False>] [-y transfrom y <True|False>] [-r R2 threshold <True|False>]" 1>&2; exit 1; }
c="test"
n=-1
x=True
y=True
r=0.8

while getopts ":c:n:x:y:r:" opt; do
    case "${opt}" in
        c)
            c=${OPTARG}
            ;;
        n)
            n=${OPTARG}
            ;;
        x)
            x=${OPTARG}
            ;;
        y)
            y=${OPTARG}
            ;;
        r)
            r=${OPTARG}
            ;;
        \?)
          echo error "Invalid option: -$OPTARG" >&2
          exit 1
          ;;

        :)
          echo error "Option -$OPTARG requires an argument."
          exit 1
          ;;
        *)
            usage
             c="test"
             n=-1
             x=False
             y=False
             r=0.4
            ;;
    esac
done
shift $((OPTIND-1))

echo "custom name = ${c}"
echo "number of threads = ${n}"
echo "x transformation = ${x}"
echo "y transformation = ${y}"
echo "R2 treshold to cut = ${r}"

start=`date +%s`
python3 ml_model.py $c $n $x $y $r > "${c}_Ml_output.out"
end=`date +%s`
echo Execution time was `expr $end - $start` seconds.

# write file
output=output_bash.out  
ls > $output 