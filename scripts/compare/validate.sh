#!/bin/bash

# Save the current working directory
original_directory=$(pwd)

# Get the directory of the script
script_directory=$(dirname "$(readlink -f "$0")")

# Change to the script's directory
cd "$script_directory" 

if [ "$#" -eq 1 ]; then
    echo "Validating model: $1. Random inputs will be generated."
    python run_actual.py --model $1 --gen_input
    python run_onnx.py --model $1
    python compare.py > result.txt
    echo "Compare results saved in result.txt."
else
    echo "Please provide an onnx file path as a single argument."
fi

# Change back to the original working directory
cd "$original_directory" 
