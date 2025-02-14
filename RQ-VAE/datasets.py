import numpy as np
import torch
import torch.utils.data as data
import random
import json
def calculate_average_embedding(embeddings_list):
    # 将嵌入列表转换为NumPy数组
    embeddings_array = np.array(embeddings_list)
    
    # 计算平均值
    average_embedding = np.mean(embeddings_array, axis=0)
    
    return average_embedding
class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb, index

    def __len__(self):
        return len(self.embeddings)

class ConEmbDataset(data.Dataset):

    def __init__(self, data_path, file_path=""):

        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]
        self._load_query_info(file_path)
        self._create_query_embs()
    def _sample_contrastive_pair(self, index):
        inner_contrastive_pairs = []
        outer_contrastive_pairs = []
        extra_index = set()
        if index in self.doc2query:
            random_query = random.choice(list(self.doc2query[index]))
            q_emb = self.query_embeddings[random_query]
            c_rel = self.doc2query[index][random_query]
            inner_contrastive_doc = index
            while inner_contrastive_doc == index and len(self.query2docs[random_query]) > 1:
                inner_contrastive_doc = random.choice(list(self.query2docs[random_query].keys()))
            if inner_contrastive_doc != index:
                extra_index.add(inner_contrastive_doc)
                inner_contrastive_pairs.append((index, inner_contrastive_doc))
            sam_rel_doc = index
            while sam_rel_doc == index and len(self.query2docs_rel[random_query][c_rel]) > 1:
                sam_rel_doc = random.choice(self.query2docs_rel[random_query][c_rel])
            rel_list = [1,2,3]
            rel_list.remove(c_rel)
            other_rel = random.choice(rel_list)
            other_rel_docs = []
            if other_rel in self.query2docs_rel[random_query]:
                other_rel_docs = random.choices(self.query2docs_rel[random_query][other_rel], k=2)
            if len(other_rel_docs) == 2:
                extra_index.add(other_rel_docs[0])
                extra_index.add(other_rel_docs[1])
                extra_index.add(sam_rel_doc)
                if other_rel < c_rel:
                    outer_contrastive_pairs.append((index, sam_rel_doc, other_rel_docs[0]))
                else:
                    outer_contrastive_pairs.append((other_rel_docs[0], other_rel_docs[1], index, sam_rel_doc))
        else:
            q_emb = self.embeddings[index]
        return q_emb, inner_contrastive_pairs, outer_contrastive_pairs, extra_index
    def _create_query_embs(self):
        query_embs = []
        for query in self.query2docs:
            doc_list = [d for d in self.query2docs[query] if self.query2docs[query][d] == 3]
            query_emb = calculate_average_embedding(self.embeddings[doc_list])
            query_embs.append(query_emb)
        self.query_embeddings = np.array(query_embs)
        print("create query embeddings ok.")
                
    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb).view(1, -1)
        q_emb, inner_contrastive_pairs, outer_contrastive_pairs, extra_index = self._sample_contrastive_pair(index)

        if extra_index:
            extra_tensor_emb = self.embeddings[list(extra_index)]
            extra_tensor_emb = torch.FloatTensor(extra_tensor_emb)
        else:
            extra_index = None
            extra_tensor_emb = None
        q_emb = torch.FloatTensor(q_emb).view(1, -1)
        return tensor_emb, index, q_emb, inner_contrastive_pairs, outer_contrastive_pairs, extra_index, extra_tensor_emb
    

    def __len__(self):
        return len(self.embeddings)
    
    def _load_query_info(self, file_path):
        with open(file_path + "train_query_to_index.json", 'r') as f:
            self.query_index = json.load(f)
        self.query2docs = {}
        self.doc2query = {}
        self.query2docs_rel = {}
        self.index2query = {}
        for query, idx in self.query_index.items():
            self.index2query[idx] = query
            
            
        with open(file_path + "qrels.txt", 'r') as f:
            for line in f.readlines():
                (q, idx, rel) = line.strip().split("\t")
                idx = int(idx)
                rel = int(rel)
                if self.query_index[q] not in self.query2docs:
                    self.query2docs[self.query_index[q]] = {}
                    self.query2docs_rel[self.query_index[q]] = {}
                if rel not in self.query2docs_rel[self.query_index[q]]:
                    self.query2docs_rel[self.query_index[q]][rel] = []
                if idx not in self.doc2query:
                    self.doc2query[idx] = {}
                self.query2docs_rel[self.query_index[q]][rel].append(idx)
                self.query2docs[self.query_index[q]][idx] = rel
                self.doc2query[idx][self.query_index[q]] = rel
