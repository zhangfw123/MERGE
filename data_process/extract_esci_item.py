import pandas as pd
import json

# 读取数据
df_examples = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("esci-data/shopping_queries_dataset/shopping_queries_dataset_sources.csv")


df_examples_products = pd.merge(
    df_examples,
    df_products,
    how='left',
    left_on=['product_locale','product_id'],
    right_on=['product_locale', 'product_id']
)

df_task_1 = df_examples_products[df_examples_products['small_version'] == 1]

lang='us'
df_task_1_us = df_task_1[(df_task_1['product_locale'] == lang)]

df_task_1_train_us = df_task_1[(df_task_1["split"] == "train") & (df_task_1['product_locale'] == lang)]

df_task_1_test_us = df_task_1[(df_task_1["split"] == "test") & (df_task_1['product_locale'] == lang)]

import json

selected_columns = df_task_1_us[['product_id', 'product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']]

product_info_dict = selected_columns.set_index('product_id').T.to_dict()
indexed_product_info_dict = {}
product_id_to_index = {}


for index, (product_id, product_info) in enumerate(product_info_dict.items()):
    indexed_product_info_dict[index] = product_info
    product_id_to_index[product_id] = index

print(len(indexed_product_info_dict))
print(len(product_id_to_index))


with open(f'./data/esci_{lang}/esci_{lang}.item.json', 'w', encoding='utf-8') as json_file:
    json.dump(indexed_product_info_dict, json_file, ensure_ascii=False, indent=4)

with open(f'./data/esci_{lang}/product_id_to_index.json', 'w') as f:
    json.dump(product_id_to_index, f)




