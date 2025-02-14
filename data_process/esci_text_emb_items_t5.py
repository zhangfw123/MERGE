import argparse
import collections
import gzip
import html
import json
import os
import random
from collections import defaultdict
import re
import torch
from tqdm import tqdm
import numpy as np
from utils import set_device, load_json, load_plm, clean_text
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig, AutoTokenizer, AutoModel


def load_data_esci(args):

    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    return item2feature

def generate_text(item2feature, features):
    item_text_list = []
    cnt = 0
    for item in item2feature:
        data = item2feature[item]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                text.append(f"{meta_key}:{meta_value.strip()}")

        item_text_list.append([int(item), " ".join(text)])
        cnt += 1
    return item_text_list

def preprocess_text(args):
    print('Process text data: ')
    print(' Dataset: ', args.dataset)
    item2feature = load_data_esci(args)
    item_text_list = generate_text(item2feature, ['product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color'])
    return item_text_list



def process_model_output(model, encoded_sentences, decoder_input_ids):
    
    with torch.no_grad():
        outputs = model(input_ids=encoded_sentences.input_ids,
                        attention_mask=encoded_sentences.attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_hidden_states=True)
    # Check if outputs have 'last_hidden_state' attribute
    if hasattr(outputs, 'last_hidden_state'):
        masked_output = outputs.last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
    elif hasattr(outputs, 'hidden_states'):
        # Use the last layer hidden states if available
        masked_output = outputs.hidden_states[-1] * encoded_sentences['attention_mask'].unsqueeze(-1)
    elif hasattr(outputs, 'encoder_last_hidden_state'):
        # If only logits are available, use them directly
        masked_output = outputs.encoder_last_hidden_state * encoded_sentences['attention_mask'].unsqueeze(-1)
    else:
        raise ValueError("Model output type not supported.")

    mean_output = masked_output.sum(dim=1) / encoded_sentences['attention_mask'].sum(dim=-1, keepdim=True)
    return mean_output.detach().cpu()



def generate_item_embedding(args, item_text_list, tokenizer, model, word_drop_ratio=-1):
    print(f'Generate Text Embedding: ')
    print(' Dataset: ', args.dataset)
    
    items, texts = zip(*item_text_list)
    order_texts = [[0]] * len(items)
    for item, text in zip(items, texts):
        order_texts[item] = text
    for text in order_texts:
        assert text != [0]

    embeddings = []
    start, batch_size = 0, 1
    while start < len(order_texts):
        if (start+1)%1==0:
            print("==>",start+1)
        field_texts = order_texts[start: start + batch_size]
        sentences = field_texts
        field_embeddings = []
        encoded_sentences = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        encoded_sentences['input_ids'] = encoded_sentences['input_ids'].long()
        with torch.no_grad():
            decoder_input_ids = torch.full((len(sentences), 1), tokenizer.pad_token_id).long().to('cuda')
            mean_output = process_model_output(model, encoded_sentences, decoder_input_ids)
        field_embeddings.append(mean_output)
            
        field_mean_embedding = torch.stack(field_embeddings, dim=0).mean(dim=0)
        embeddings.append(field_mean_embedding)
        start += batch_size
    embeddings = torch.cat(embeddings, dim=0).to(torch.float32).numpy()
    print('Embeddings shape: ', embeddings.shape)

    file = os.path.join(args.root, args.dataset + '.emb-' + args.plm_name + ".npy")
    np.save(file, embeddings)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='esci', help='esci')
    parser.add_argument('--root', type=str, default="data")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--plm_name', type=str, default='t5-base')
    parser.add_argument('--plm_checkpoint', type=str, default='google-t5/t5-base')
    parser.add_argument('--max_sent_len', type=int, default=2048)
    parser.add_argument('--word_drop_ratio', type=float, default=-1, help='word drop ratio, do not drop by default')
    parser.add_argument('--job_name', type=str, default="worker", required=False, help='')
    parser.add_argument('--task_index', type=int, default=0, required=False, help='')
    parser.add_argument('--worker_hosts', type=str, default="", required=False, help='')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.root = os.path.join(args.root, args.dataset)

    device = set_device(args.gpu_id)
    args.device = device

    item_text_list = preprocess_text(args)

    plm_tokenizer, plm_model = load_plm(args.plm_checkpoint)
    if plm_tokenizer.pad_token_id is None:
        plm_tokenizer.pad_token_id = 0
    plm_model = plm_model.to(device)

    generate_item_embedding(args, item_text_list,plm_tokenizer,
                            plm_model, word_drop_ratio=args.word_drop_ratio)

