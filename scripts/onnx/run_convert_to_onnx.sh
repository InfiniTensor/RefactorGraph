#!/bin/bash 
  
while getopts ":i:o:" opt; do  
    case $opt in  
        i)  
            model_path=$OPTARG  
            ;;  
        o)  
            output_path=$OPTARG  
            ;;  
        \?)  
            echo "Invalid option: -$OPTARG"  
            exit 1  
            ;;  
    esac  
done  
if [ -z "$model_path" ] || [ -z "$output_path" ]; then  
    echo "Model path and output path are required."  
    exit 1  
fi  

# 确保输出目录存在  
mkdir -p "$output_path"  
  
# 运行第一个Python文件并保存输出到文件  
python3 make_serialize.py --model "$model_path" --output "$output_path"  
  
# 运行第二个Python文件并保存输出到文件  
python3 to_onnx.py --input "$output_path" 
  
# 输出完成信息  
echo "Models have been run successfully. Outputs are saved in $output_path."