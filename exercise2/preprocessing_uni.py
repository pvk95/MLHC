import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle as pkl

EMB_SIZE = 512

train_df = pd.read_csv('project2_data/10k_diabetes/diab_train.csv')
valid_df = pd.read_csv('project2_data/10k_diabetes/diab_validation.csv')
test_df = pd.read_csv('project2_data/10k_diabetes/diab_test.csv')

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
# module_path = "uni_module"

# Import the Universal Sentence Encoder's TF Hub module
# If the module cannot be downloaded this way, do it manually:
#   1. Download from -> https://tfhub.dev/google/universal-sentence-encoder/2?tf-hub-format=compressed
#   2. Write the path of the uncompressed module in 'module_path'
#   3. Run 'embed = hub.Module(module_path)' instead of 'embed = hub.Module(module_url)'

embed = hub.Module(module_url)
# embed = hub.Module(module_path)
# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

# store all the sentences in a list
sentences = list()
# store indices of nan values to add them again later
indices = list()
index = 0
all_df = pd.concat([train_df, valid_df, test_df])
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    for i in range(all_df.shape[0]):
        if pd.isna(all_df.iloc[i]['diag_1_desc']):
            indices.append(index)
        else:
            sentences.append(all_df.iloc[i]['diag_1_desc'])
        index += 1
        if pd.isna(all_df.iloc[i]['diag_2_desc']):
            indices.append(index)
        else:
            sentences.append(all_df.iloc[i]['diag_2_desc'])
        index += 1
        if pd.isna(all_df.iloc[i]['diag_3_desc']):
            indices.append(index)
        else:
            sentences.append(all_df.iloc[i]['diag_3_desc'])
        index += 1
    sentence_embs = list(session.run(embed(sentences)))

# Add nans back
for index in indices:
    sentence_embs.insert(index, np.nan)

with open('project2_data/10k_diabetes/uni_emb.pkl', 'wb') as output_file:
    pkl.dump(sentence_embs, output_file)
