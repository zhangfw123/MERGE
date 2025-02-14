DATASET=esci
DATA_PATH=./data/
OUTPUT_DIR=./results/$DATASET/
RESULTS_FILE=./results/$DATASET/res.json
CKPT_PATH=./ckpt/$DATASET/model_$DATASET/checkpoint-17000
python3 ./GR_training/test.py \
    --gpu_id 0 \
    --ckpt_path $CKPT_PATH \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --results_file $RESULTS_FILE \
    --test_batch_size 64 \
    --num_beams 100 \
    --test_prompt_ids 0 \
    --index_file .index.merge.json

