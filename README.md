# MERGE (ACL 2025)

This repo is for source code of ACL 2025 paper "Multi-level Relevance Document Identifier Learning for Generative Retrieval". Paper Link: https://aclanthology.org/2025.acl-long.497.pdf.


## Requirements

For RQ-VAE: requirements_rqvae.txt

For GR: requirements_gr.txt




## Preprocess data

1. download shopping_queries_dataset_examples.parquet, shopping_queries_dataset_products.parquet, shopping_queries_dataset_sources.csv from https://github.com/amazon-science/esci-data

2. preprocess

```
Using process_data.ipynb to obtain the preprocessed data, including: doc_to_relevance_docs.json, esci_lang.item.json, esci_lang.test.seen.json, esci_lang.train.json, product_id_to_index.json. 

lang: us, es, jp
```


## Generate Doc Embeddings
```
python data_process/esci_text_emb_items_t5.py --root data --dataset esci --plm_name t5-base --plm_checkpoint [plm_model]
```

## Train RQ-VAE in MERGE

```
bash ./RQ-VAE/train_tokenizer.sh 0.01 "1.0 0.75 0.25 0.0" "0.001 0.001 0.001 0.001" merge
```

## Tokenize and Obtain DocIDs

```
bash ./RQ-VAE/tokenize.sh [model_path] [model_name]
```


## Train GR model 

```
bash ./GR_train/train.sh
```

## Test

```
bash ./GR_train/test.sh
```


## Qrels for other DR models
```
gen_query_doc_relevance.ipynb.
```


## Citation

```
@inproceedings{zhang2025multi,
  title={Multi-level Relevance Document Identifier Learning for Generative Retrieval},
  author={Zhang, Fuwei and Liu, Xiaoyu and Jia, Xinyu and Zhang, Yingfei and Zhang, Shuai and Li, Xiang and Zhuang, Fuzhen and Lin, Wei and Zhang, Zhao},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={10066--10080},
  year={2025}
}
```

## Acknowledgment
This code is based on https://github.com/HonghuiBao2000/LETTER.


