#!/bin/bash

# SFT / finetune: instruction tuning
# Run from the repository root: bash scripts/v1_5/finetune.sh

export LLAVA_SILENT_LOGS=1
export LLAVA_DEBUG_SEQ=0
export LLAVA_LOG_LOSS_BREAKDOWN=0

# When the corporate network/proxy cannot reach W&B, W&B may throw ProxyError and block ~90s.
# To enable cloud logging: export REPORT_TO=wandb
# Or local-only offline logs: export WANDB_MODE=offline and REPORT_TO=wandb
: "${REPORT_TO:=wandb}"
: "${WANDB_PROJECT:=llava-v1_5-linear-canchaloss-7b}"
: "${WANDB_NAME:=finetune-$(date +%Y%m%d-%H%M%S)}"
: "${VISION_FUSION_GATE_INIT:=0.3}"
: "${FUSION_GATE_LR:=5e-3}"
: "${LOG_DIR:=./logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetune-${WANDB_NAME}.log"

EXTRA=()
if [ -n "${RESUME_FROM:-}" ]; then
    EXTRA+=(--resume_from_checkpoint "$RESUME_FROM")
fi

CMD=(
    deepspeed llava/train/train_mem.py
    --deepspeed ./scripts/zero3.json
    --model_name_or_path ./checkpoints/llava-v1.5-7b-dgp-pretrain/
    --version v1
    --data_path ./playground/data/llava_v1_5_mix665k.json
    --image_folder ./playground/data
    --vision_tower ./clip-vit-large-patch14-336
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --use_layerwise_vision_fusion True
    --vision_router_hidden_dim 1024
    --vision_router_dropout 0.1
    --vision_fusion_gate_init "$VISION_FUSION_GATE_INIT"
    --fusion_gate_lr "$FUSION_GATE_LR"
    --tune_mm_mlp_adapter False
    --freeze_vision_tower True
    --train_llm_layers True
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 True
    --output_dir ./checkpoints/llava-v1.5-7b-dgp-sft/
    --num_train_epochs 1
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 4
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 100
    --extra_save_steps 5194
    --learning_rate 2e-5
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --max_grad_norm 1.0
    --tf32 True
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to "$REPORT_TO"
)

echo "Training logs: $LOG_FILE"
"${CMD[@]}" "${EXTRA[@]}" 2>&1 | tee -a "$LOG_FILE"
