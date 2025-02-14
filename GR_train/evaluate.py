import math
import numpy as np
from collections import defaultdict


def get_topk_results(predictions, scores, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(" ","") for _ in predictions]
    generated_results = []
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        one_pred = [p[0] for p in sorted_pairs]
        generated_results.append(one_pred)
        target_item = targets[b]
        target_item_cnt = {}
        
        for i in target_item:
            if i not in target_item_cnt:
                target_item_cnt[i] = 1
            else:
                target_item_cnt[i] += 1
                print(i, target_item_cnt[i])
        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_item:
                for num in range(target_item_cnt[sorted_pred[0]]):
                    one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results, generated_results
def get_topk_ranking_results(predictions, targets, k, all_items=None):
    results = []
    B = len(targets)
    predictions = [_.strip().replace(" ","") for _ in predictions]
    generated_results = []
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        generated_results.append(batch_seqs)
        target_item = targets[b]
        one_results = []
        for pred in batch_seqs:
            if pred in target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)
    return results,generated_results
def get_metrics_results(topk_results, generated_results, target, targets_with_rel, metrics):
    res = {}
    # print(target)
    for m in metrics:
        if m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(generated_results, k, targets_with_rel)
        elif m.lower().startswith("recall"):
            k = int(m.split("@")[1])
            res[m] = recall_k(topk_results, k, target)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    
    return res

def dcg(relevances, k):
    """Calculate DCG given relevances and k."""
    relevances = np.array(relevances)[:k]
    return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))

def ndcg_k(ranked_topk_results, k, targets_with_rel):
    ndcg_scores = 0.0
    
    for predicted, target_with_rel in zip(ranked_topk_results, targets_with_rel):
        target_dict = {target: rel for target, rel in target_with_rel}
        
        predicted_rels = [target_dict.get(p, 0) for p in predicted]
        dcg_score = dcg(predicted_rels, k)
        
        ideal_rels = sorted([rel for _, rel in target_with_rel], reverse=True)
        idcg_score = dcg(ideal_rels, k)
        
        ndcg_score = dcg_score / idcg_score if idcg_score > 0 else 0
        ndcg_scores += ndcg_score
    
    return ndcg_scores
def recall_k(topk_results, k, target_size):
    recall = 0.0
    for n,row in enumerate(topk_results):
        res = row[:k]
        recall += sum(res) / len(target_size[n])  
    return recall 


