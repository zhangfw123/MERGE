import argparse
import random
import torch
import numpy as np
from time import time
import logging
# import wandb
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets import EmbDataset,ConEmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os

def parse_list(input_string):
    return [float(item) for item in input_string.split(',')]
def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="../data", help="Input data path.")
    parser.add_argument('--valid_batch_size', type=int, default=1024, help='batch size')

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--qd_align', type=float, default=0.1, help='qd alignment weight')
    parser.add_argument('--trade_off_inner_outer', type=float, nargs='+', default=[1.0, 0.75, 0.25, 0.0], help='trade_off_inner_outer weight')
    parser.add_argument('--inner_outer_layer_weight', type=float, nargs='+', default=[0.01, 0.01, 0.01, 0.01], help='inner_outer_layer_weight weight')
    
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb', type=str, default="./RQ-VAE/ckpt/Instruments-32d-sasrec.pt", help='cf emb')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")
    parser.add_argument('--job_name', type=str, default="worker", required=False, help='')
    parser.add_argument('--task_index', type=int, default=0, required=False, help='')
    parser.add_argument('--worker_hosts', type=str, default="", required=False, help='')

    return parser.parse_args()

def collate_fn(batch):
    # print(len(batch))
    doc_embs = []
    query_embs = []
    indexs = []
    old_to_new_indexs = {}
    inner_contrastive_pairs = []
    outer_contrastive_pairs = []
    idx_cnt = 0
    map_function_inner = lambda x: (old_to_new_indexs.get(x[0], None), old_to_new_indexs.get(x[1], None))
    map_function_outer = lambda x: (old_to_new_indexs.get(x[0], None), old_to_new_indexs.get(x[1], None), old_to_new_indexs.get(x[2], None))
    for bc in batch:
        if bc[1] not in old_to_new_indexs:
            old_to_new_indexs[bc[1]] = idx_cnt
            idx_cnt += 1
        doc_embs.append(bc[0])
        query_embs.append(bc[2])
        indexs.append(old_to_new_indexs[bc[1]])
        if len(inner_contrastive_pairs) < 5000000:
            if bc[5] is not None:
                for extra_index in bc[5]:
                    old_to_new_indexs[extra_index] = idx_cnt 
                    indexs.append(old_to_new_indexs[extra_index])
                    idx_cnt += 1
                doc_embs.append(bc[6])
                query_embs.append(torch.cat([bc[2] for i in range(len(bc[6]))], dim=0))
            inner_contrastive_pairs += list(map(map_function_inner, bc[3]))
            outer_contrastive_pairs += list(map(map_function_outer, bc[4]))
        
            
    
    doc_embs = torch.cat(doc_embs, dim=0)
    query_embs = torch.cat(query_embs, dim=0)
    return doc_embs, query_embs, inner_contrastive_pairs, outer_contrastive_pairs

if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    print(args)
    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = ConEmbDataset(args.data_path)
    valid_data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  qd_align = args.qd_align,
                  trade_off_inner_outer = args.trade_off_inner_outer,
                  inner_outer_layer_weight = args.inner_outer_layer_weight
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_data,num_workers=args.num_workers,
                             batch_size=args.valid_batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(data_loader, valid_data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)




