#!/bin/bash

# 定义 narrator_model 的值数组
# narrator_models=("Qwen2" "GLM4V" "Qwen" "InternVL2")
# narrator_models=("DeepSeekVL" "LLama")
narrator_models=("ZeroOne")

# 遍历每个 narrator_model
for narrator_model in "${narrator_models[@]}"; do
    python3 run_v3.py --model "$narrator_model" --running_step "run_description_generation"
                echo ""  # 输出一个空行
                echo "" 
done 