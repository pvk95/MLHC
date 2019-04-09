from sklearn.metrics import roc_auc_score
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import multiprocessing as mp
import pandas as pd
import numpy as np
import csv

# load Data
df_train_1 = pd.read_csv(
    './project2_data/drugsCom_raw/drugsComTrain_raw.tsv', delimiter='\t', encoding='utf-8')
df_test = pd.read_csv(
    './project2_data/drugsCom_raw/drugsComTest_raw.tsv', delimiter='\t', encoding='utf-8')
df_train, df_valid = train_test_split(df_train_1, random_state=42)

# Some Drug-Names appear multiple time in a non-unique fashion.
# e.g. "Stalevo", "Stalevo 200"
drug_names_raw = df_train.drugName.unique()
drug_names = np.copy(drug_names_raw)

for idx1, drug1 in enumerate(drug_names_raw):
    for idx2, drug2 in enumerate(drug_names_raw):
        if drug1 in drug2 and drug1 != drug2 and drug2 in drug_names:
            idx_ = np.where(drug_names == drug2)
            drug_names = np.delete(drug_names, idx_)

print(f'Deleted {len(drug_names_raw) - len(drug_names)} duplicate drug_names')

# Find the mean rating of all drugs:
mean_rating = np.mean(df_train.rating)

# Find mean Rating for each Drugname
drug_ratings = {}
for drug_name in drug_names:
    idxs = [drug_name in name for name in df_train.drugName.values]
    ratings = df_train.rating[idxs]
    mean_rating = np.mean(ratings)
    drug_ratings[drug_name] = mean_rating
print('Found the mean ratings for each drug')

# Find ratings for df_valid
valid_ratings = []
num_not_found = 0
for valid_drug in df_valid.drugName:
    ratings = []
    # The drug listing could contain multiple drugs
    # e.g. 'drug A, drug B'
    # we take the average rating of each drug
    for key in drug_ratings.keys():
        if key in valid_drug:
            ratings.append(drug_ratings[key])

    # If no drug with that name is found then the rating is set to the mean rating
    if ratings == []:
        ratings.append(mean_rating)
        num_not_found += 1
    rating = np.mean(ratings)
    valid_ratings.append(rating)
print(f'Number of not found drugs {num_not_found}')
print(f'MSE: {mean_squared_error(df_valid.rating.values, valid_ratings)}')


def to_labels(ratings):
    labels = []
    for rating in ratings:
        if rating > 7:
            label = 0
        elif rating < 4:
            label = 2
        else:
            label = 1
        labels.append(label)
    return labels


valid_true_labels = to_labels(df_valid.rating.values)
valid_pred_labels = to_labels(valid_ratings)
print(f'Accuracy: {accuracy_score(valid_true_labels, valid_pred_labels):.3} - f1: {f1_score(valid_true_labels, valid_pred_labels, average="micro"):.3}')

# ! wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
# ! mkdir -p project2_data/glove
# ! unzip -d project2_data/glove/ glove.twitter.27B.zip
vocab_size = 10000
vector_length = 25
max_seq_length = 500
glove_embeddings = pd.read_csv('project2_data/glove/glove.twitter.27B.25d.txt',
                               header=None, delimiter=' ', encoding='utf-8', quoting=csv.QUOTE_NONE)
print('Loaded glove embeddings')
glove_embeddings = glove_embeddings[:vocab_size]

glove_word = glove_embeddings[0]

glove_vectors = glove_embeddings.iloc[:, 1:].values
M1 = np.zeros((vocab_size + 1, vector_length + 1))
M1[:-1, :-1] = glove_vectors
glove_vectors = M1
glove_vectors[-1, -1] = 1


def process_chunk(comments):
    processed_comments = []
    lookup = {}
    for comment in comments:
        idxs = []

        comment = comment.replace('&#39;', "'")
        comment = comment.replace('\n', ' ')
        comment = comment.replace('\r', ' ')
        comment = comment.replace('&amp;', ' and ')
        comment = comment.replace('&quot;', ' ')

        for i, word in enumerate(comment.split(' ')):
            # if the comment is longer than the max-seq-length break the  loop
            if i > max_seq_length:
                break

            # put everything into lowercase
            word = word.lower()

            # if empty word skip
            if word == '':
                continue
            # if word is in our lookup table
            if word in lookup.keys():
                idx = lookup[word]
            else:
                try:
                    idx = glove_word[glove_word == word].index[0]
                except IndexError:
                    idx = vocab_size
                # put word into lookup table
                lookup[word] = idx
            idxs.append(idx)
        processed_comments.append(idxs)
    processed_comments = keras.preprocessing.sequence.pad_sequences(
        processed_comments,
        maxlen=max_seq_length,
        padding='post',
        truncating='post',
        value=0
    )
    return processed_comments


def comments_to_idxs(comments):
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    chunks = np.array_split(comments, cores)
    results = pool.map(process_chunk, chunks)
    results = np.vstack(results)
    return results

#train_idxs = comments_to_idxs(df_train_1.review.values)
#test_idxs = comments_to_idxs(df_test.review.values)

#np.save('project2_data/glove/train_idxs.npy', train_idxs)
#np.save('project2_data/glove/test_idxs.npy', test_idxs)


train_idxs_1 = np.load('project2_data/glove/train_idxs.npy')
test_idxs = np.load('project2_data/glove/test_idxs.npy')

max_seq_length = 50
train_idxs_1 = train_idxs_1[:, :max_seq_length]
test_idxs = test_idxs[:, :max_seq_length]

train_idx, valid_idx, train_y, valid_y = train_test_split(
    train_idxs_1, df_train_1.rating.values, random_state=42)
train_y_cat = pd.get_dummies(to_labels(train_y)).values
valid_y_cat = pd.get_dummies(to_labels(valid_y)).values


# Bidiretional LSTM

batch_size = 64
hidden_units = 256
epochs = 20

sequence_input = keras.layers.Input(shape=(max_seq_length,))
embedding_sequence = keras.layers.Embedding(
    vocab_size + 1,
    vector_length + 1,
    weights=[glove_vectors],
    input_length=max_seq_length,
    trainable=False
)(sequence_input)

lstm = keras.layers.Bidirectional(
    keras.layers.LSTM(hidden_units))(embedding_sequence)
drop = keras.layers.Dropout(0.5)(lstm)
dense = keras.layers.Dense(128)(drop)
out = keras.layers.Dense(3, activation='softmax')(dense)
model = keras.models.Model(inputs=sequence_input, outputs=out)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# model.fit(
#   x = train_idx,
#   y = train_y_cat,
#   validation_data=(valid_idx, valid_y_cat),
#   epochs=epochs
#
# model.save('model_cat.h5')


model = keras.models.load_model('model_cat.h5')
print('loaded model')

lstm_predictions = model.predict(valid_idx)
print(f'ROC AUC {roc_auc_score(valid_y_cat, lstm_predictions):.3}  Acc: {accuracy_score(np.argmax(valid_y_cat, axis=1), np.argmax(lstm_predictions, axis=1)):.3} ')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

combined = np.hstack([pd.get_dummies(to_labels(valid_ratings)).values, lstm_predictions])

train_combined, valid_combined, train_cat, valid_cat = train_test_split(combined, np.argmax(valid_y_cat, axis=1))

clf = SVC()
clf.fit(train_combined, train_cat)
print(clf.score(valid_combined, valid_cat))