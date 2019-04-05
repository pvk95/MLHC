import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
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

n_features = 1000
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
    elif(val>=4):
        y_under.append(0)
        X_under.append(X[i,:])
    else:
        y_under.append(-1)
        X_under.append(X[i,:])

X_under = np.vstack(X_under)
y_under = np.vstack(y_under).ravel()

assert X_under.shape[0] == y_under.shape[0]
plt.hist(y_under,align='mid')

X_train,X_test,y_train,y_test = train_test_split(X_under,y_under,test_size=0.25,random_state=0)

metrics = pd.DataFrame(data=[],columns=["Name","F1_score","Accuracy"])
rfc = RandomForestClassifier(class_weight='balanced',n_estimators= 50)
lg = LogisticRegression(class_weight='balanced',n_jobs=-1)


rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

f1 = f1_score(y_test,y_pred,average="weighted")
acc = accuracy_score(y_test,y_pred)

metrics = metrics.append({"Name":'RandomForest',"F1_score":f1,"Accuracy":acc},ignore_index=True)

lg.fit(X_train,y_train)
y_pred = lg.predict(X_test)

f1 = f1_score(y_test,y_pred,average="weighted")
acc = accuracy_score(y_test,y_pred)

metrics = metrics.append({"Name":'LogisticRegression',"F1_score":f1,"Accuracy":acc},ignore_index=True)

print(metrics)

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
