#from google.colab import drive
#drive.mount('/content/drive')

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
import multiprocessing as mp
import csv

# load Data
df_train = pd.read_csv(
    './project2_data/drugsCom_raw/drugsComTrain_raw.tsv', delimiter='\t', encoding='utf-8')
df_test = pd.read_csv(
    './project2_data/drugsCom_raw/drugsComTest_raw.tsv', delimiter='\t', encoding='utf-8')

df_train_model, df_valid = train_test_split(df_train, random_state=42)

df_train = df_train.iloc[:100,:]

# Some Drug-Names appear multiple time in a non-unique fashion.
# e.g. "Stalevo", "Stalevo 200"
drug_names_raw = df_train.drugName.unique()

for idx,drug in enumerate(drug_names_raw):
    drug = drug.lower()
    drug_names_raw[idx] = drug

drug_names = np.copy(drug_names_raw)
for idx1, drug1 in enumerate(drug_names_raw):
    for idx2, drug2 in enumerate(drug_names_raw):
        if drug1 in drug2 and drug1 != drug2 and drug2 in drug_names:
            idx_ = np.where(drug_names == drug2)
            drug_names = np.delete(drug_names, idx_)

print(f'Deleted {len(drug_names_raw) - len(drug_names)} duplicate drug_names')

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

#Pre process the comments and generate word2vec embeddings
def process_chunk(comments):
    processed_comments = []
    lookup = {}
    for comment in comments:
        idxs = []
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

def process_comments(comments):
    pre_comments = []
    for comment in comments:
        comment = comment.replace(',', '')
        comment = comment.replace('.', '')
        comment = comment.replace('(', '')
        comment = comment.replace(')', '')
        comment = comment.replace('"', '')
        comment = comment.replace('&amp;', 'and')
        comment = comment.replace('!', '')
        comment = comment.replace('/', ' ')
        comment = comment.replace('&quot;', ' ')
        comment = comment.replace('&#39;', ' ')
        comment = comment.replace('\n', ' ')
        comment = comment.replace('\r', ' ')
        pre_comments.append(comment)
    return np.array(pre_comments)

def comments_to_idxs(comments):
    cores = mp.cpu_count()
    pool = mp.Pool(cores)
    chunks = np.array_split(comments, cores)
    results = pool.map(process_chunk, chunks)
    results = np.vstack(results)
    return results

review_train = process_comments(df_train.review.values)
review_test = process_comments(df_test.review.values)

train_idxs = comments_to_idxs(review_train)
test_idxs = comments_to_idxs(review_test)

np.save('project2_data/glove/train_idxs.npy', train_idxs)
np.save('project2_data/glove/test_idxs.npy', test_idxs)

#train_idxs_1 = np.load('project2_data/glove/train_idxs.npy')
#test_idxs = np.load('project2_data/glove/test_idxs.npy')


def getDrugFeatures(drugName,df):
    drugs_prior = {}

    for i, curr_drug in enumerate(drugName):
        if curr_drug not in drugs_prior.keys():
            drugs_prior[curr_drug] = [df['rating'].iloc[i]]
        else:
            drugs_prior[curr_drug].append(df['rating'].iloc[i])

    for curr_drug in drugs_prior.keys():
        rating_arr = np.array(drugs_prior[curr_drug])
        n_rating_arr = rating_arr.shape[0]
        rating_counter = np.array([0, 0, 0])
        for curr_rating in rating_arr:
            if (curr_rating >= 7):
                rating_counter[0] = rating_counter[0] + 1
            elif (curr_rating >= 4):
                rating_counter[1] = rating_counter[1] + 1
            else:
                rating_counter[2] = rating_counter[2] + 1
        rating_counter = rating_counter / n_rating_arr
        drugs_prior[curr_drug] = rating_counter

    feat_drug = []
    for curr_drug in drugName:
        feat_drug.append(drugs_prior[curr_drug])

    feat_drug = np.stack(feat_drug, axis=0)
    return feat_drug


def getCommentFeatures(review,df,n_features=1000):
    tvec = TfidfVectorizer(max_features=n_features)
    X = tvec.fit_transform(review).todense().A
    return X

def getDiscreteLabels(ratings):
    temp_ratings = ratings.copy()
    for i, curr_rating in enumerate(ratings):
        if (curr_rating >= 7):
            temp_ratings[i] = 1
        elif (curr_rating >= 4):
            temp_ratings[i] = 0
        else:
            temp_ratings[i] = -1

    return temp_ratings

def undersample_data(y_in):

    np.random.seed(0)
    n_samples = y_in.shape[0]
    idxs_inclusion = [False]*n_samples
    for i in range(n_samples):
        if(y_in[i] == 1):
            if(np.random.choice(np.arange(5),size=1)==0):
                idxs_inclusion[i] = True
        else:
            idxs_inclusion[i] = True

    return np.array(idxs_inclusion)

#feat_drug = getDrugFeatures(drug_names)
X = getCommentFeatures(review_train,df_train,n_features=1000)
y = getDiscreteLabels(ratings = df_train['rating'].values)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

#Undersample the training data
idxs_inclusion = undersample_data(y_train)
X_train = X_train[idxs_inclusion,:]
y_train = y_train[idxs_inclusion]

#X = feat_drug[idxs_inclusion, :]
#X = np.concatenate((X_under, X), axis=-1)


metrics = pd.DataFrame(data=[], columns=["Name", "F1_score", "Accuracy"])
def getScores(model, model_name, metrics):
    print("Model: ", model_name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)

    metrics = metrics.append({"Name": model_name, "F1_score": f1, "Accuracy": acc}, ignore_index=True)

    return [y_pred, metrics]


rfc = RandomForestClassifier(class_weight='balanced', n_estimators=50)
lg = LogisticRegression(class_weight='balanced')
adb = ensemble.AdaBoostClassifier()
cb = naive_bayes.ComplementNB()

model_names = ["RandomForest", "LogisticRegression", "Adaboost", "ComplementNB"]
models = [rfc, lg, adb, cb]

all_models = zip(models, model_names)
for model, model_name in all_models:
    _, metrics = getScores(model, model_name, metrics)

print(metrics)

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Dropout

input_shape = X.shape[1:]

model = Sequential()
model.add(Dense(500, input_shape=input_shape, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])

def getCategorical(y):
    n_samples = y.shape[0]
    y_one_hot = np.zeros((n_samples, 3))
    for i in range(y.shape[0]):
        if (y[i] == 1):
            y_one_hot[i, 0] = 1
        elif (y[i] == 0):
            y_one_hot[i, 1] = 1
        else:
            y_one_hot[i, 2] = 1

    return y_one_hot

def flatten(y_to_flatten):
    ytemp = np.argmax(y_to_flatten, axis=1)
    y_pred_fl = ytemp.copy()
    y_pred_fl[ytemp == 2] = -1
    y_pred_fl[ytemp == 0] = 1
    y_pred_fl[ytemp == 1] = 0

    return y_pred_fl

cat_y_train = getCategorical(y_train)

history = model.fit(X_train, cat_y_train, epochs=20, validation_data=(X_test, y_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

metrics = pd.DataFrame(data=[], columns=["Name", "F1_score", "Accuracy"])
cat_y_pred = model.predict(X_test)

print(y_test.shape)
y_test = flatten(y_test)
y_pred = flatten(cat_y_pred)

f1 = f1_score(y_test, y_pred, average="weighted")
acc = accuracy_score(y_test, y_pred)

metrics = metrics.append({"Name": "ANN", "F1_score": f1, "Accuracy": acc}, ignore_index=True)

print(metrics)


###########################################
#Word Cloud
pos_tweets = df_train[df_train.rating >= 5]
pos_string = []
for t in pos_tweets.review:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_string)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Positive')
plt.axis("off")
plt.show()

neg_tweets = df_train[df_train.rating < 5]
neg_string = []
for t in neg_tweets.review:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(neg_string)
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Neg')
plt.axis("off")
plt.show()


##################################
#Martin code
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

####################
max_seq_length = 50
train_idxs = train_idxs[:, :max_seq_length]
test_idxs = test_idxs[:, :max_seq_length]

train_idx, valid_idx, train_y, valid_y = train_test_split(
    train_idxs, df_train.rating.values, random_state=42)
train_y_cat = pd.get_dummies(getDiscreteLabels(train_y)).values
valid_y_cat = pd.get_dummies(getDiscreteLabels(valid_y)).values


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



'''
os.getcwd()
os.chdir('/content/drive/My Drive/ML4HC/')

train_df = pd.read_csv('drugsComTrain_raw.tsv', sep='\t')

train_df = pd.read_csv('project2_data/drugsCom_raw/drugsComTrain_raw.tsv',sep='\t')
test_df = pd.read_csv('project2_data/drugsCom_raw/drugsComTest_raw.tsv',sep='\t')

spl_name = []
for name in train_df['drugName']:
    spl_name.append(name.split(sep='/')[-1])

train_df['drugName'] = spl_name
train_df = train_df.drop(['date', 'Unnamed: 0', 'usefulCount'], axis=1)

review = train_df['review'].values

for i, sentence in enumerate(review):
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('(', '')
    sentence = sentence.replace(')', '')
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('\r', '')
    sentence = sentence.replace('"', '')
    sentence = sentence.replace('&#039;', ' ')
    sentence = sentence.replace('&amp;', 'and')
    sentence = sentence.replace('!', '')
    sentence = sentence.replace('/', ' ')
    sentence = sentence.replace('&quot;', ' ')
    review[i] = sentence

train_df['review'] = review

'''

'''
lookup = {}
not_words = []
def process_chunk(chunk):
    processed_chunk = []
    for i,sentence in enumerate(chunk):
        print("Sample: ",i)
        sentence_vector = []
        sentence = sentence.replace(',','')
        sentence = sentence.replace('.','')
        sentence = sentence.replace('(','')
        sentence = sentence.replace(')','')
        sentence = sentence.replace('\n','')
        sentence = sentence.replace('\r','')
        sentence = sentence.replace('"','')
        for word in sentence.split(' '):
            word = word.lower()
            if word == '\n' or word == '':
                continue

            if word not in lookup.keys():
                idx = np.where(types_csv == word)
                # If the wor isn't found - use "unk"
                if len(idx[0]) == 0:
                    # idx = np.where(types_csv == 'unk')
                    not_words.append(word)
                    continue
                lookup[word] = idx
            else:
                idx = lookup[word]

            word_vec = vectors_csv.iloc[idx[0][0]].values
            sentence_vector.append(word_vec)
        # sentence_vector.append(vectors_csv.iloc[645536].values)
        processed_chunk.append(sentence_vector)
    return processed_chunk

'''

