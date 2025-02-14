# MERGE

This is the pytorch implementation of MERGE:


## Requirements

For RQ-VAE: requirements_rqvae.txt
For GR: requirements_gr.txt


### Preprocess data

1. download raw esci data

2. preprocess 
```
python data_process/extract_esci_item.py
```

### Generate Doc Embeddings
```
python data_process/esci_text_emb_items_t5.py --root data --dataset esci --plm_name t5-base --plm_checkpoint [plm_model]
```

### Train RQ-VAE in MERGE

```
bash ./RQ-VAE/train_tokenizer.sh 0.01 "1.0 0.75 0.25 0.0" "0.001 0.001 0.001 0.001" merge
```

### Tokenize and Obtain DocIDs

```
bash ./RQ-VAE/tokenize.sh [model_path] [model_name]
```


### Train GR model 

```
bash ./GR_train/train.sh
```

### Test

```
bash ./GR_train/test.sh
```



This code is based on https://github.com/HonghuiBao2000/LETTER.
