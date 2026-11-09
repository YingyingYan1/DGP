#!/bin/bash

# Pretrain: train projector + layerwise vision fusion + LLM backbone; freeze vision_tower
# Save full model weights (same trainable scope as SFT)
# Run from the repository root: bash scripts/v1_5/pretrain.sh

# Silence extra train.py / llava_arch debug prints by default
export LLAVA_SILENT_LOGS=1
export LLAVA_DEBUG_SEQ=0
# Set to 1 to print lm_loss / router entropy regularizer breakdown (aligned with logging_steps)
export LLAVA_LOG_LOSS_BREAKDOWN=0

# When the corporate network/proxy cannot reach W&B, W&B may throw ProxyError and block ~90s.
# To enable cloud logging: export REPORT_TO=wandb
# Or local-only offline logs: export WANDB_MODE=offline and REPORT_TO=wandb
: "${REPORT_TO:=wandb}"
: "${WANDB_PROJECT:=llava-v1_5-linear-canchaloss-7b}"
: "${WANDB_NAME:=pretrain-$(date +%Y%m%d-%H%M%S)}"
: "${VISION_FUSION_GATE_INIT:=0.3}"
: "${FUSION_GATE_LR:=5e-3}"
: "${ROUTER_ENTROPY_LOSS_COEF:=0.01}"
: "${LOG_DIR:=./logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pretrain-${WANDB_NAME}.log"

MODEL_PATH="./llava-v1.5-7b"
VISION_TOWER="./clip-vit-large-patch14-336"
DATA_PATH="./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="./playground/data/LLaVA-Pretrain/images"
OUTPUT_DIR="./checkpoints/llava-v1.5-7b-dgp-pretrain"

CMD=(
    deepspeed llava/train/train_mem.py
    --deepspeed ./scripts/zero3.json
    --model_name_or_path "$MODEL_PATH"
    --version plain
    --data_path "$DATA_PATH"
    --image_folder "$IMAGE_FOLDER"
    --vision_tower "$VISION_TOWER"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --use_layerwise_vision_fusion True
    --vision_router_hidden_dim 1024
    --vision_router_dropout 0.1
    --vision_fusion_gate_init "$VISION_FUSION_GATE_INIT"
    --fusion_gate_lr "$FUSION_GATE_LR"
    --router_entropy_loss_enabled True
    --router_entropy_loss_coef "$ROUTER_ENTROPY_LOSS_COEF"
    --tune_mm_mlp_adapter False
    --freeze_vision_tower True
    --train_llm_layers True
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs 1
    --per_device_train_batch_size 8
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 4
    --evaluation_strategy no
    --save_strategy steps
    --save_steps 200
    --learning_rate 2e-5
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 2048
    --max_grad_norm 1.0
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to "$REPORT_TO"
)

echo "model_path: $MODEL_PATH" | tee -a "$LOG_FILE"
echo "vision_tower: $VISION_TOWER" | tee -a "$LOG_FILE"
echo "data_path: $DATA_PATH" | tee -a "$LOG_FILE"
echo "image_folder: $IMAGE_FOLDER" | tee -a "$LOG_FILE"
echo "output_dir: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "report_to: $REPORT_TO" | tee -a "$LOG_FILE"
echo "Training logs: $LOG_FILE"
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
