import torch
import torch.nn as nn
from torch.nn import functional as F

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, sk_epsilons, beta = 1,
                 kmeans_init = False, kmeans_iters = 100, sk_iters=100,):
        super().__init__()
        self.n_e_list = n_e_list
        print(n_e_list, sk_epsilons)
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=beta,
                                                        kmeans_init = self.kmeans_init,
                                                        kmeans_iters = self.kmeans_iters,
                                                        sk_epsilon=sk_epsilon,
                                                        sk_iters=sk_iters)
                                        for n_e, sk_epsilon in zip(n_e_list,sk_epsilons) ])


    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)
    
    def vq_ini(self, x):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):

            x_res = quantizer.vq_init(residual, use_sk=True)
            residual = residual - x_res
            x_q = x_q + x_res


    def inner_triplet_loss(self, triplets, features, margin=0.2):
        triplets = torch.tensor(triplets, dtype=torch.long)
        anchors = features[triplets[:, 0]]
        positives = features[triplets[:, 1]]
        negatives = features[triplets[:, 2]]
        pos_distances = F.pairwise_distance(anchors, positives, p=2)
        neg_distances = F.pairwise_distance(anchors, negatives, p=2)
        losses = F.relu(pos_distances - neg_distances + margin)
        return losses.mean()
            
    
    def outer_contrastive_loss(self, x_q, contrastive_pairs, temperature=0.1):
        device = x_q.device
        sim_matrix = F.cosine_similarity(x_q.unsqueeze(1), x_q.unsqueeze(0), dim=2) / temperature
        # print(contrastive_pairs)
        idx1 = torch.tensor([pair[0] for pair in contrastive_pairs], dtype=torch.long, device=device)  # (N,)
        idx2 = torch.tensor([pair[1] for pair in contrastive_pairs], dtype=torch.long, device=device)  # (N,)

        pos_sim = sim_matrix[idx1, idx2].unsqueeze(1)  

        all_sim = sim_matrix[idx1, :]  

        mask = torch.ones_like(all_sim, dtype=torch.bool)
        mask.scatter_(1, idx2.view(-1, 1), False) 

        neg_sim = all_sim.masked_select(mask).view(all_sim.size(0), -1)

        logits = torch.cat([pos_sim, neg_sim], dim=1)

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)  # (N,)

        loss = F.cross_entropy(logits, labels)

        return loss
    
    def forward(self, x, labels, outer_contrastive_pairs=None, inner_triplet_pairs=None, use_sk=True):
        all_losses = []
        all_indices = []
        outer_con_losses = []
        inner_triplet_losses = []
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            label = labels[str(idx)]
            
            x_res, loss, indices = quantizer(residual,label, idx, use_sk=use_sk)
            if outer_contrastive_pairs is not None:
                outer_con_loss = self.outer_contrastive_loss(residual, outer_contrastive_pairs)
                outer_con_losses.append(outer_con_loss)
            if inner_triplet_pairs is not None:
                inner_triplet_loss = self.inner_triplet_loss(inner_triplet_pairs, residual)
                inner_triplet_losses.append(inner_triplet_loss)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        return x_q, mean_losses, outer_con_losses, inner_triplet_losses, all_indices