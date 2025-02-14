import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm

import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os
from datasets import EmbDataset,ConEmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, emb_idx = data[0], data[1]
            data = data.to(self.device)

            self.model.vq_initialization(data)    
    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_outer_con_loss = 0
        total_qd_align_loss = 0
        total_inner_triplet_loss = 0
    
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]

        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):
            doc_embs, query_embs, outer_contrastive_pairs, inner_triplet_pairs, qd_align_w = data
            doc_embs = doc_embs.to(self.device)
            query_embs = query_embs.to(self.device)
            qd_align_w  = qd_align_w.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices, dense_out, outer_con_losses, inner_triplet_losses, qd_align_loss = self.model(doc_embs, query_embs, self.labels, outer_contrastive_pairs, inner_triplet_pairs, qd_align_w=qd_align_w)
            loss, loss_recon, quant_loss = self.model.compute_loss(out, rq_loss, dense_out, outer_con_losses, inner_triplet_losses, qd_align_loss, xs=doc_embs)
            
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_outer_con_loss += torch.stack(outer_con_losses).mean().item()
            total_qd_align_loss += qd_align_loss.item()
            total_inner_triplet_loss += torch.stack(inner_triplet_losses).mean().item()
            total_quant_loss += quant_loss.item()
        return total_loss, total_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss, quant_loss.item()


    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data, self.labels)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set))/num_sample


        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        train_loss_output += set_color("outer_con_loss", "blue") + ": %.4f" % total_outer_con_loss
        train_loss_output +=", "
        train_loss_output += set_color("inner_triplet_loss", "blue") + ": %.4f" % total_inner_triplet_loss
        train_loss_output +=", "
        train_loss_output += set_color("qd_align_loss", "blue") + ": %.4f" % total_qd_align_loss
        return train_loss_output + "]"

    def fit(self, data, valid_data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss, quant_loss = self._train_epoch(data, epoch_idx)

            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss, total_outer_con_loss, total_inner_triplet_loss, total_qd_align_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(valid_data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)

                if epoch_idx>50:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate




