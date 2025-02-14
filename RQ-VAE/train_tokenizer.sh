qd_align=$1
trade_off_inner_outer=$2
inner_outer_layer_weight=$3
ckpt_dir=$4
dataset=esci

python RQ-VAE/main.py \
  --device cuda \
  --data_path ./data/$dataset/$dataset.emb-t5-td.npy \
  --alpha 0.01 \
  --beta 0.0001 \
  --ckpt_dir ./ckpt/RQ-VAE/$dataset/$ckpt_dir/ \
  --eval_step 1\
  --epochs 300\
  --batch_size 2048\
  --num_emb_list 256 256 256 256\
  --sk_epsilons 0.0 0.0 0.0 0.003\
  --qd_align $qd_align\
  --trade_off_inner_outer $trade_off_inner_outer\
  --inner_outer_layer_weight $inner_outer_layer_weight
  