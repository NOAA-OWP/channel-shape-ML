#!/bin/bash

# Check the operating system
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    # Linux
    echo "Detected Linux. Running Linux-specific commands."
    # Check if Anaconda is installed
    if command -v conda &> /dev/null
    then
        echo "Anaconda is already installed."
    else
        # Download and install Anaconda
        mkdir -p ~/miniconda3
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh

        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        echo "Anaconda has been installed."
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    echo "Detected macOS. Running macOS-specific commands."
    if command -v conda &> /dev/null
    then
        echo "Anaconda is already installed."
    else
        # Download and install Anaconda
        mkdir -p ~/miniconda3
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
        bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
        rm -rf ~/miniconda3/miniconda.sh

        ~/miniconda3/bin/conda init bash
        ~/miniconda3/bin/conda init zsh
        echo "Anaconda has been installed."
    fi

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    echo "Detected Windows. Running Windows-specific commands."
    if command -v conda &> /dev/null
    then
        echo "Anaconda is already installed."
    else
        # Download and install Anaconda
        curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
        start /wait "" miniconda.exe /S
        del miniconda.exe
        echo "Anaconda has been installed."
    fi
        
else
    # Unsupported OS
    echo "Unsupported operating system: $OSTYPE"
fi


anaconda_base=$(conda info --base)

# Check if the command was successful (exit code 0)
if [ $? -eq 0 ]; then
    
    anaconda_base="${anaconda_base}/etc/profile.d/conda.sh"
    echo "Anaconda base directory: $anaconda_base"

    if [ -f "$anaconda_base" ]; then
        # Source the conda.sh file to set up Anaconda
        source "$anaconda_base"
        echo "Anaconda has been initialized."

        conda activate base

        find_in_conda_env(){
            conda env list | grep "${@}" >/dev/null 2>/dev/null
        }

        if find_in_conda_env ".*WD-model.*" ; then
            echo "Environment found..."
            conda info --envs
            conda activate WD-model
            echo "(WD-model) environment activated"
            echo "Running scripts..."

            # Main ml estimations
            usage() { echo "Usage: $0 [-n number of cores <int>]" 1>&2; exit 1; }
            n=-1
            
            while getopts ":n:" opt; do
                case "${opt}" in
                    n)
                        n=${OPTARG}
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
                        n=-1
                        ;;
                esac
            done
            shift $((OPTIND-1))
            echo "number of threads = ${n}"
            
            start=`date +%s`
            python3 deploy.py $n > "deploy_output.out"
            end=`date +%s`
            echo Execution time was `expr $end - $start` seconds.

            # write file
            output=output_bash.out  
            ls > $output 
        else 
            echo "Creating new environment..."
            conda env create --file wd_env.yaml
        fi

    else
        echo "conda.sh file not found in $anaconda_base/etc/profile.d/."
    fi
else
    echo "Failed to retrieve Anaconda base directory."
fi
