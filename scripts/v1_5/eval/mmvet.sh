#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path "./checkpoint_glimpse-linear-sigmoid/llava-v1.5-7b-sft-glimpse-linear-sigmoid/"\
    --question-file "./playground/data/eval/mm-vet/llava-mm-vet.jsonl" \
    --image-folder "./playground/data/eval/mm-vet/mm-vet/images/" \
    --answers-file "./playground/data/eval/mm-vet/answers/llava-v1.5-7b-sigmoid.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p ./playground/data/eval/mm-vet/results

python ./scripts-glmpse-canchaloss/convert_mmvet_for_eval.py\
    --src "./playground/data/eval/mm-vet/answers/llava-v1.5-7b-sigmoid.jsonl"\
    --dst "./playground/data/eval/mm-vet/results/llava-v1.5-7b-sigmoid.json"

