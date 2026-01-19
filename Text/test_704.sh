#!/bin/bash

export gpt_apikey=
export gpt_modelpath=gpt-4o
export doubao_apikey=
export kimi_apikey=
export kimi_modelpath=moonshot-v1-8k
export Qwen_apikey=
export Qwen_modelpath=qwen-plus
export glm_apikey=
export glm_modelpath=glm-4-plus
export ds_apikey=
nohup python game_one_word.py > log1.txt 2>&1 &
#nohup python Aihumanv.py > log2.txt 2>&1 &
