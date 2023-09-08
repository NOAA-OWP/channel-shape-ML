#!/bin/bash
# Conda
# my directory source /home/amodaresirad/miniconda3/etc/profile.d/conda.sh
#Account and Email Information
#SBATCH -A amodaresirad ## User ID
#SBATCH --mail-type=end
#SBATCH --mail-user=arashmodaresirad@u.boisestate.edu

#SBATCH -J quant       # job name
#SBATCH -o outputs/quant_results_final.o%j # output and error file name (%j expands to jobID)
#SBATCH -e outputs/quant_errors_final.e%j
#SBATCH -n 2               # Run one process
#SBATCH --cpus-per-task=28 # allow job to multithread across all cores
#SBATCH -t 40-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 10000

if [ ! -d /home/amodaresirad/River_new/bf/bankfull_W_D/outputs ]; then
    echo "outputs floder not found, creating one..."
    mkdir -p /home/amodaresirad/River_new/bf/bankfull_W_D/outputs;
fi

source /home/amodaresirad/anaconda3/etc/profile.d/conda.sh

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