import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import naive_bayes
plt.style.use('fivethirtyeight')
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('project2_data/drugsCom_raw/drugsComTrain_raw.tsv',sep='\t')
test_df = pd.read_csv('project2_data/drugsCom_raw/drugsComTest_raw.tsv',sep='\t')

spl_name =[]
for name in train_df['drugName']:
    spl_name.append(name.split(sep = '/')[-1])

train_df['drugName'] = spl_name
train_df = train_df.drop(['date','Unnamed: 0','usefulCount'],axis=1)

review = train_df['review'].values

for i,sentence in enumerate(review):
    sentence = sentence.replace(',', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('(', '')
    sentence = sentence.replace(')', '')
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('\r', '')
    sentence = sentence.replace('"', '')
    sentence = sentence.replace('&#039;',' ')
    sentence = sentence.replace('&amp;','and')
    sentence = sentence.replace('!','')
    sentence = sentence.replace('/' ,' ')
    sentence = sentence.replace('&quot;',' ')
    review[i] = sentence

train_df['review'] = review


def getDrugFeatures():
    drugs_prior={}

    for i,curr_drug in enumerate(spl_name):
        if curr_drug not in drugs_prior.keys():
            drugs_prior[curr_drug] = [train_df['rating'].iloc[i]]
        else:
            drugs_prior[curr_drug].append(train_df['rating'].iloc[i])

    for curr_drug in drugs_prior.keys():
        rating_arr = np.array(drugs_prior[curr_drug])
        n_rating_arr = rating_arr.shape[0]
        rating_counter = np.array([0,0,0])
        for curr_rating in rating_arr:
            if (curr_rating>=7):
                rating_counter[0] = rating_counter[0] + 1
            elif (curr_rating>=4):
                rating_counter[1] = rating_counter[1] + 1
            else:
                rating_counter[2] = rating_counter[2] + 1
        rating_counter = rating_counter/ n_rating_arr
        drugs_prior[curr_drug] = rating_counter

    feat_drug = []
    for curr_drug in spl_name:
        feat_drug.append(drugs_prior[curr_drug])

    feat_drug = np.stack(feat_drug,axis=0)
    return feat_drug

def getCommentFeatures(n_features = 10000):

    idxs_inclusion = np.array([False]*review.shape[0])
    tvec = TfidfVectorizer(max_features=n_features)
    X = tvec.fit_transform(review).todense().A
    y = train_df['rating'].values

    y_under = []
    X_under =[]

    for i,val in enumerate(y):
        if (val>=7 and val<=10):
            if np.random.randint(low=0,high=4,size=1) ==0:
                y_under.append(1)
                X_under.append(X[i,:])
                idxs_inclusion[i] = True
        elif(val>=4):
            y_under.append(0)
            X_under.append(X[i,:])
            idxs_inclusion[i] = True
        else:
            y_under.append(-1)
            X_under.append(X[i,:])
            idxs_inclusion[i] = True

    X_under = np.vstack(X_under)
    y_under = np.vstack(y_under).ravel()

    assert X_under.shape[0] == y_under.shape[0]

    return [X_under,y_under,idxs_inclusion]

def getDiscreteLabels(ratings):
    temp_ratings = ratings.copy()
    for i,curr_rating in enumerate(ratings):
        if (curr_rating>=7):
            temp_ratings[i] = 1
        elif (curr_rating>=4):
            temp_ratings[i] = 0
        else:
            temp_ratings[i] = -1

    return temp_ratings

feat_drug = getDrugFeatures()
X_under,y_under,idxs_inclusion = getCommentFeatures(n_features=1000)

X = feat_drug[idxs_inclusion,:]
y= getDiscreteLabels(ratings=train_df['rating'].values)[idxs_inclusion]

X = np.concatenate((X_under,X),axis=-1)
y = y_under
#X_train,X_test,y_train,y_test = train_test_split(X_under,y_under,test_size=0.25,random_state=0)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

metrics = pd.DataFrame(data=[],columns=["Name","F1_score","Accuracy"])

def getScores(model,model_name,metrics):
    print("Model: ",model_name)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test,y_pred,average="weighted")
    acc = accuracy_score(y_test,y_pred)

    metrics = metrics.append({"Name":model_name,"F1_score":f1,"Accuracy":acc},ignore_index=True)

    return [y_pred,metrics]

rfc = RandomForestClassifier(class_weight='balanced',n_estimators= 50)
lg = LogisticRegression(class_weight='balanced')
adb = ensemble.AdaBoostClassifier()
cb = naive_bayes.ComplementNB()

model_names = ["RandomForest","LogisticRegression","Adaboost","ComplementNB"]
models = [rfc,lg,adb,cb]

all_models = zip(models,model_names)
for model,model_name in all_models:
    _,metrics = getScores(model,model_name,metrics)

print(metrics)

from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras.layers import Dense

input_shape = X.shape[1:]

model = Sequential()
model.add(Dense(5000,input_shape=input_shape,activation='relu'))
model.add(Dense(1000,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(optimizer='adam',loss=tf.keras.losses.categorical_crossentropy,metrics=['acc'])

n_samples = y.shape[0]
y_one_hot = np.zeros((n_samples,3))
for i in range(y.shape[0]):
    if (y[i] == 1):
        y_one_hot[i,0] = 1
    elif (y[i]==0):
        y_one_hot[i,1] = 1
    else:
        y_one_hot[i,2] = 1

y = y_one_hot
del y_one_hot

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

model.fit(X_train,y_train,validation_data=(X_test,y_test))

y_pred = model.predict(X_train)

f1 = f1_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)

metrics = metrics.append({"Name":"ANN","F1_score":f1,"Accuracy":acc},ignore_index=True)

'''
pos_tweets = train_df[train_df.rating >= 5]
pos_string = []
for t in pos_tweets.review:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(pos_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Positive')
plt.axis("off")
plt.show()

neg_tweets = train_df[train_df.rating < 5]
neg_string = []
for t in neg_tweets.review:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('Neg')
plt.axis("off")
plt.show()

condition = train_df['condition'].values
for i,sentence in enumerate(condition):
    sentence = str(sentence)
    sentence.replace('<',' ')
    condition[i] = sentence

all_conditions = np.unique(condition)
spl_name = np.array(spl_name).reshape(1,-1)
from sklearn import preprocessing
one_hot = preprocessing.OneHotEncoder()
spl_name = one_hot.fit_transform(spl_name)

spl_name.shape

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

from sklearn.ensemble import ExtraTreesClassifier

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
imp_feat = forest.feature_importances_
print(imp_feat.shape)
