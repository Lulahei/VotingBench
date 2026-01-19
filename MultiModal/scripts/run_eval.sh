#!/bin/bash

# sleep 3h
# 定义 narrator_model 和 model 的值数组
narrator_models=("Qwen2" "GLM4V" "InternVL2" "DeepSeekVL" "LLama" "Step" "human")
models=("DeepSeekVL")
# models=("Qwen2" "GLM4V" "Qwen" "InternVL2")

# 遍历每个 narrator_model
for narrator_model in "${narrator_models[@]}"; do   
    # 遍历每个 model
    for model in "${models[@]}"; do
        # 跳过 narrator_model 和 model 相同的情况
        if [ "$narrator_model" != "$model" ]; then
            # 设置 run_type 的值
            if [ "$model" == "Step" ]; then
                run_type="api"
            elif [ "$model" == "human" ]; then
                run_type="human"
            else
                run_type="local"
            fi

            python3 run_v3.py --narrator_model "$narrator_model" --model "$model" --run_type "$run_type"
            echo ""  # 输出一个空行
            echo ""  
            echo "" 
        fi
    done
done