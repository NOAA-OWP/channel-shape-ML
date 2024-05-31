#!/bin/bash
# Conda
# source /home/arashmod/miniconda3/etc/profile.d/conda.sh

ulimit -v unlimited
ulimit -s unlimited
ulimit -u 10000

# Get the current working directory
current_dir=$(pwd)
# Define the relative path to the outputs folder
relative_path="/outputs"
# Construct the full path
output_path="${current_dir}${relative_path}"

if [ ! -d "$output_path" ]; then
    echo "outputs folder not found, creating one..."
    mkdir -p "$output_path"
fi

anaconda_base=$(conda info --base)
anaconda_base="${anaconda_base}/etc/profile.d/conda.sh"
source "$anaconda_base"

conda activate base

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

if find_in_conda_env ".*WD-env.*" ; 

then
    echo "Environment found..."
else 
    echo "Creating new environment..."
    conda env create --file wd_env.yaml
fi

conda activate WD-env
echo "(wd-model) environment activated"
echo "Running scripts..."

# Main python training
usage() { echo "Usage: $0 [-c submission name <string>] [-n number of cores <int>] [-x transform x <True|False>] [-y transfrom y <True|False>] [-r R2 threshold <True|False>]" 1>&2; exit 1; }
c="test"
n=-1
x=True
y=True
r=0.4
t=3

while getopts ":c:n:x:y:r:t:" opt; do
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
        t)
            t=${OPTARG}
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
             t=3
            ;;
    esac
done
shift $((OPTIND-1))

echo "custom name = ${c}"
echo "number of threads = ${n}"
echo "x transformation = ${x}"
echo "y transformation = ${y}"
echo "R2 treshold to cut = ${r}"
echo "count treshold to cut = ${t}"

start=`date +%s`
python3 model.py $c $n $x $y $r $t > "${c}_Ml_output.out"
end=`date +%s`
echo Execution time was `expr $end - $start` seconds.

# write file
output=output_bash.out  
ls > $output 