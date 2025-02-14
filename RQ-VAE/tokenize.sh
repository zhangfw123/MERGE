model_path=$1
dataset=esci
model=best_collision_model.pth
mkdir $model_path/index
python ./RQ-VAE/generate_indices.py\
    --dataset $dataset \
    --alpha 1e-1 \
    --beta 1e-4 \
    --epoch 10000 \
    --output_dir $model_path/index/ \
    --checkpoint $model_path/$model