# export WANDB_MODE=disabled
DATASET=esci
OUTPUT_DIR=./ckpt/$DATASET/MERGE/
torchrun --nproc_per_node=1 --master_port=2314 ./GR_train/finetune.py \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --per_device_batch_size 512 \
    --learning_rate 5e-4 \
    --epochs 100 \
    --index_file .index.json \
    --temperature 1.0 \
    --dataset $DATASET \
    --data_path ./data \
    --base_model google-t5/t5-base

