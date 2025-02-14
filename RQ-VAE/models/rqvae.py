import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer



class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 qd_align = 0.1,
                 trade_off_inner_outer = [1.0, 0.75, 0.25, 0.0],
                 inner_outer_layer_weight = [0.01, 0.01, 0.01, 0.01]
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.alpha = alpha
        self.beta = beta
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        self.qd_align = qd_align
        self.trade_off_inner_outer = trade_off_inner_outer
        self.inner_outer_layer_weight = inner_outer_layer_weight

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
    def relevance_docs_contrastive_loss(self, x_q, contrastive_pairs, temperature=0.1):
        batch_size = x_q.size(0)
        device = x_q.device
        temperature = 1.0
        # 计算余弦相似度矩阵
        sim_matrix = F.cosine_similarity(x_q.unsqueeze(1), x_q.unsqueeze(0), dim=2)  # (batch_size, batch_size)
        sim_matrix = sim_matrix / temperature  # 缩放相似度

        # 初始化损失
        loss = 0.0
        
        # 遍历所有正样本对
        for idx_1, idx_2 in contrastive_pairs:
            # 正样本对的相似度
            pos_sim = sim_matrix[idx_1, idx_2]

            # 负样本对的相似度
            neg_sim = torch.cat([sim_matrix[idx_1, :idx_2], sim_matrix[idx_1, idx_2+1:]])

            # 计算对比损失
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim])  # (num_classes,)
            labels = torch.zeros(1, dtype=torch.long, device=device)  # 正样本对的索引为 0

            # 计算交叉熵损失
            loss += F.cross_entropy(logits.unsqueeze(0), labels)  # logits: (1, num_classes), labels: (1,)

        # 平均损失
        loss = loss / len(contrastive_pairs)
        return loss
    def relevance_docs_contrastive_loss_matrix(self, x_q, contrastive_pairs, temperature=0.1):
        """
        计算对比损失（矩阵形式）

        :param x_q: 编码后的特征，形状为 (batch_size, feature_dim)
        :param contrastive_pairs: 正样本对列表，每个元素是一个元组 (idx_1, idx_2)
        :param temperature: 温度参数，控制对比损失的敏感度
        :return: 对比损失
        """
        batch_size = x_q.size(0)
        device = x_q.device

        sim_matrix = F.cosine_similarity(x_q.unsqueeze(1), x_q.unsqueeze(0), dim=2) / temperature

        # 将 contrastive_pairs 转换为两个索引张量 idx1 和 idx2
        idx1 = torch.tensor([pair[0] for pair in contrastive_pairs], dtype=torch.long, device=device)  # (N,)
        idx2 = torch.tensor([pair[1] for pair in contrastive_pairs], dtype=torch.long, device=device)  # (N,)

        # 获取所有正样本对的相似度，形状为 (N, 1)
        pos_sim = sim_matrix[idx1, idx2].unsqueeze(1)  # Positive similarities

        # 获取对应 idx1 的所有相似度，形状为 (N, batch_size)
        all_sim = sim_matrix[idx1, :]  # (N, batch_size)

        # 创建掩码，以排除正样本对的相似度
        mask = torch.ones_like(all_sim, dtype=torch.bool)
        mask.scatter_(1, idx2.view(-1, 1), False)  # 将正样本的位置置为 False

        # 提取负样本的相似度，形状为 (N, batch_size - 1)
        neg_sim = all_sim.masked_select(mask).view(all_sim.size(0), -1)

        # 将正样本相似度放在第一列，负样本放在后面，形状为 (N, 1 + batch_size - 1)
        logits = torch.cat([pos_sim, neg_sim], dim=1)

        # 标签全为 0，表示正样本是 logits 的第一个位置
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)  # (N,)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, x, q_embs, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, use_sk=True):
        x = self.encoder(x)
        q_encode =  self.encoder(q_embs)
        x_q, rq_loss, outer_con_losses, inner_triplet_losses, indices = self.rq(x, labels, outer_contrastive_pairs, inner_triplet_pairs,  use_sk=use_sk)
        out = self.decoder(x_q)
        qd_align_loss = F.mse_loss(x, q_encode, reduction='mean')
        return out, rq_loss, indices, x_q, outer_con_losses, inner_triplet_losses, qd_align_loss
    
    
    
    def vq_initialization(self,x, use_sk=True):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs, labels, use_sk=False):
        x_e = self.encoder(xs)
        _, _,  _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        return indices
    @torch.no_grad()
    def get_embs(self, xs):
        x_e = self.encoder(xs)
        return x_e
    def compute_loss(self, out, quant_loss, dense_out, outer_con_losses=None, inner_triplet_losses=None, qd_align_loss=None, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')
        total_loss = loss_recon + self.quant_loss_weight * quant_loss

        if outer_con_losses is not None and inner_triplet_losses is not None:
            for i in range(len(self.inner_outer_layer_weight)):
                total_loss = total_loss + self.inner_outer_layer_weight[i]*(self.trade_off_inner_outer[i]*outer_con_losses[i] + (1-self.trade_off_inner_outer[i])*inner_triplet_losses[i])
        elif inner_triplet_losses is not None:
            for i in range(len(self.inner_outer_layer_weight)):
                total_loss = total_loss + self.inner_outer_layer_weight[i]*inner_triplet_losses[i]
        elif outer_con_losses is not None:
            for i in range(len(self.inner_outer_layer_weight)):
                total_loss = total_loss + self.inner_outer_layer_weight[i]*outer_con_losses[i]
        if qd_align_loss is not None:
            total_loss = total_loss + self.qd_align*qd_align_loss  

        return total_loss, loss_recon, quant_loss