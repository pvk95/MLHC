from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import multiprocessing
import linecache
import sklearn
import string
import time

# Importing dataframes
train_df = pd.read_csv('project2_data/10k_diabetes/diab_train.csv')
valid_df = pd.read_csv('project2_data/10k_diabetes/diab_validation.csv')
test_df = pd.read_csv('project2_data/10k_diabetes/diab_test.csv')

NaN_values = ['?', 'nan', 'Not Available', 'Not Mapped', 'None']

# Prediction
pred_columns = ['readmitted']

# All integer columns
int_columns = ['Unnamed: 0', 'time_in_hospital', 'num_lab_procedures',
               'num_procedures', 'num_medications', 'number_outpatient',
               'number_emergency', 'number_inpatient', 'number_diagnoses',
               'diag_1', 'diag_2', 'diag_3'
               ]

# All categorical columns
cat_columns = ['race', 'gender', 'age', 'weight',
               'max_glu_serum',
               'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
               'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
               'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
               'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
               'insulin', 'glyburide.metformin', 'glipizide.metformin',
               'glimepiride.pioglitazone', 'metformin.rosiglitazone',
               'metformin.pioglitazone', 'change', 'diabetesMed']


# All string columns
str_columns = ['diag_1_desc', 'diag_2_desc', 'diag_3_desc'
               ]
#               'admission_source_id', 'medical_specialty', 'admission_type_id',
#               'discharge_disposition_id', 'payer_code'
#               ]

category_map = {'race': {'AfricanAmerican': 0, 'Caucasian': 1, 'Asian': 2, 'Other': 3, 'Hispanic': 4, '?': np.NaN},
                'gender': {'Male': 0, 'Female': 1},
                'age': {'[60-70)': 0, '[70-80)': 1, '[80-90)': 2, '[50-60)': 3, '[20-30)': 4, '[40-50)': 5, '[30-40)': 6, '[10-20)': 7, '[90-100)': 8, '[0-10)': 9},
                'weight': {'?': np.NaN, '[75-100)': 1, '[100-125)': 2, '[50-75)': 3, '[125-150)': 4, '[25-50)': 5, '[150-175)': 6, '[0-25)': 7},
                'max_glu_serum': {'None': np.NaN, 'Norm': 1, '>200': 2, '>300': 3},
                'A1Cresult': {'None': 0, '>8': 1, 'Norm': 2, '>7': 3},
                'metformin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'repaglinide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'nateglinide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'chlorpropamide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'glimepiride': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'acetohexamide': {'No': 0},
                'glipizide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'glyburide': {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3},
                'tolbutamide': {'No': 0, 'Steady': 1},
                'pioglitazone': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'rosiglitazone': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'acarbose': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'miglitol': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'troglitazone': {'No': 0},
                'tolazamide': {'No': 0, 'Steady': 1},
                'examide': {'No': 0},
                'citoglipton': {'No': 0},
                'insulin': {'No': 0, 'Down': 1, 'Up': 2, 'Steady': 3},
                'glyburide.metformin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'glipizide.metformin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'glimepiride.pioglitazone': {'No': 0},
                'metformin.rosiglitazone': {'No': 0},
                'metformin.pioglitazone': {'No': 0},
                'change': {'No': 0, 'Ch': 1},
                'diabetesMed': {'No': 0, 'Yes': 1}
                }


def create_maps(df, cat_columns, nan_categories):
    maps = {}
    for column in cat_columns:
        maps[column] = {category: i for i,
                        category in enumerate(df[column].unique())}

    for key in maps.keys():
        for category in maps[key]:
            if category in nan_categories:
                maps[key][category] = np.NaN
    return maps


def fix_integers(df, int_columns):
    for column in int_columns:
        df[column] = df[column].where(
            lambda x: [str(i).isdigit() for i in x], np.NaN).astype(np.float)
    return df


def fix_categories(df, cat_columns, cat_map):
    for column in cat_columns:
        import pdb
        pdb.set_trace()
        df[column] = df[column].replace(cat_map[column])
        df[column] = df[column].astype('float64')
    return df


def set_NaN(df, cat_columns, nan_values):
    nan_map = {key: np.NaN for key in nan_values}
    for column in cat_columns:
        df[column] = df[column].replace(nan_map)
    return df


def create_missing_columns(df, unique_columns):
    for column in unique_columns:
        if column not in df.columns:
            df[column] = 0
    return df


# Fix the Integers - Set all the unknown to 0
train_df = fix_integers(train_df, int_columns)
valid_df = fix_integers(valid_df, int_columns)
test_df = fix_integers(test_df, int_columns)

# Find category map
# category_map = create_maps(train_df, cat_columns, NaN_values)
# print(category_map)

# Set all missing values to NaN
train_df = set_NaN(train_df, cat_columns, NaN_values)
valid_df = set_NaN(valid_df, cat_columns, NaN_values)
test_df = set_NaN(test_df, cat_columns, NaN_values)

# Create Dummy Variables for all the categories
train_cat_dummies = pd.get_dummies(train_df[cat_columns])
valid_cat_dummies = pd.get_dummies(valid_df[cat_columns])
test_cat_dummies = pd.get_dummies(test_df[cat_columns])

# Some dummy-variable get excluded because the category doesn't exist in a dataframe.
# In here we make it so that all 3 dataframes have the same columns
unique_columns = np.unique(np.concatenate(
    (train_cat_dummies.columns,
     valid_cat_dummies.columns,
     test_cat_dummies.columns)))


train_cat_dummies = create_missing_columns(train_cat_dummies, unique_columns)
valid_cat_dummies = create_missing_columns(valid_cat_dummies, unique_columns)
test_cat_dummies = create_missing_columns(test_cat_dummies, unique_columns)

# Merge dummy variables and integer variables to one dataframe
train_data_numerical = pd.concat(
    [train_cat_dummies, train_df[int_columns]], axis=1)
valid_data_numerical = pd.concat(
    [valid_cat_dummies, valid_df[int_columns]], axis=1)
test_data_numerical = pd.concat(
    [test_cat_dummies, test_df[int_columns]], axis=1)

# get y_values
train_y = train_df[pred_columns[0]]
valid_y = valid_df[pred_columns[0]]
test_y = test_df[pred_columns[0]]


# Fill missing values with median
train_data_numerical = train_data_numerical.fillna(
    train_data_numerical.median())
valid_data_numerical = valid_data_numerical.fillna(
    train_data_numerical.median())
test_data_numerical = test_data_numerical.fillna(
    train_data_numerical.median())

##************************************************** ##
#                Testing Linear Regression           ##
##************************************************** ##

reg = LogisticRegression(solver='lbfgs').fit(
    train_data_numerical.values, train_y)
score = reg.score(valid_data_numerical.values, valid_y)
f1 = f1_score(test_y.values, reg.predict(
    test_data_numerical.values))
print(f'The score on the validation set was {score:.3f} - \
    the f1-score was {f1:.3f}')

##************************************************** ##
#                Testing SVM                         ##
##************************************************** ##

svc = SVC(gamma='scale', class_weight='balanced', probability=True, kernel='poly').fit(
    train_data_numerical.values,
    train_y.values.ravel())
score = svc.score(valid_data_numerical.values, valid_y.values.ravel())
f1 = f1_score(test_y.values, svc.predict(
    test_data_numerical.values))
print(f'The score on the validation set was {score:.3f} - \
    the f1_score was {f1:.3f}')


##************************************************** ##
#                Word2Vec                            ##
##************************************************** ##

translator = str.maketrans('', '', string.punctuation)


def vector_to_string(vector):
    vector = [str(s) for s in vector]
    joined_string = " ".join(vector)
    string_no_punctuation = joined_string.translate(translator)
    return string_no_punctuation


# Join all strings together and remove punctuation
train_data_string = np.apply_along_axis(
    vector_to_string, 1, train_df[str_columns].values)
valid_data_string = np.apply_along_axis(
    vector_to_string, 1, valid_df[str_columns].values)
test_data_string = np.apply_along_axis(
    vector_to_string, 1, test_df[str_columns].values)

# Load Word2Vec model
# Downloaded from http://bioasq.org/news/bioasq-releases-continuous-space-word-vectors-obtained-applying-word2vec-pubmed-abstracts
# ! mkdir -p project2_data/word2vec/
# ! wget -O file.tar.gz  http://bioasq.lip6.fr/tools/BioASQword2vec/
# ! tar -C project2_data/word2vec -xvf file.tar.gz --strip 1
# ! rm file.tar.gz
types_csv = pd.read_csv(
    'project2_data/word2vec/types.txt', header=None, delimiter='\n')
vectors_csv = pd.read_csv(
    'project2_data/word2vec/vectors.txt', header=None, sep=' ',  delimiter='\n')
print('opened word2vec files')


def process_chunk(chunk):
    processed_chunk = []
    for sentence in chunk:
        sentence_vector = []
        for word in sentence.split(' '):
            word = word.lower()
            if word == '\n' or word == '':
                continue
            idx = np.where(types_csv == word)

            # If the wor isn't found - use "unk"
            if len(idx[0]) == 0:
                #idx = np.where(types_csv == 'unk')
                continue

            word_vec = vectors_csv.iloc[idx[0][0]].values
            sentence_vector.append(word_vec)
        # sentence_vector.append(vectors_csv.iloc[645536].values)
        processed_chunk.append(sentence_vector)
    return processed_chunk


def word2vec_parallel(data_string):
    cores = multiprocessing.cpu_count()
    chunks = np.array_split(data_string, cores)
    pool = multiprocessing.Pool(cores)
    results = pool.map(process_chunk, chunks)
    results = np.hstack(results)
    return results


# Transform the string data into word2vec
#valid_data_word2vec = word2vec_parallel(valid_data_string)
#train_data_word2vec = word2vec_parallel(train_data_string)
#test_data_word2vec = word2vec_parallel(test_data_string)
#print('Finished word2vec transformation')

# Save the data
#np.save('project2_data/word2vec/train_word2vec_1.npy', train_data_word2vec)
#np.save('project2_data/word2vec/valid_word2vec_1.npy', valid_data_word2vec)
#np.save('project2_data/word2vec/test_word2vec_1.npy', test_data_word2vec)

# Make space in the memory
del types_csv
del vectors_csv

train_data_word2vec = np.load('project2_data/word2vec/train_word2vec.npy')
valid_data_word2vec = np.load('project2_data/word2vec/valid_word2vec.npy')
test_data_word2vec = np.load('project2_data/word2vec/test_word2vec.npy')
print('Loaded word2vec transformation')


# Transform word2vec into floats
def transform(array):
    return [[[float(val) for val in word[0].split(" ")] for word in sentence] for sentence in array]


train_data_word2vec = transform(train_data_word2vec)
valid_data_word2vec = transform(valid_data_word2vec)
test_data_word2vec = transform(test_data_word2vec)
print("Transformed the string-vector into floats")


# Find the maximum sentence length
def find_max(array):
    return max([len(sentence) for sentence in array])


train_max = find_max(train_data_word2vec)
valid_max = find_max(valid_data_word2vec)
test_max = find_max(test_data_word2vec)
final_max = max([train_max, valid_max, test_max])
print(f"The maximum length of a sentence is {final_max}.")


# Zero pad the sentences - such that everything has has the same length
def zero_pad(array, max_length, word_vec_length):
    return [[sentence[idx] if idx < len(sentence) else np.zeros((word_vec_length)) for idx in range(max_length)] for sentence in array]


word_vec_length = 200
train_data_word2vec = zero_pad(train_data_word2vec, final_max, 200)
valid_data_word2vec = zero_pad(valid_data_word2vec, final_max, 200)
test_data_word2vec = zero_pad(test_data_word2vec, final_max, 200)
print(f"Zero padded all the word-vectors to length {word_vec_length}")


train_data_word2vec = np.array(train_data_word2vec)
valid_data_word2vec = np.array(valid_data_word2vec)
test_data_word2vec = np.array(test_data_word2vec)
print("Transformed everything into numpy array")

##************************************************** ##
##                    RNN-Training                   ##
##************************************************** ##


class Metrics(keras.callbacks.Callback):
    def __init__(self, valid_data, **kwargs):
        self.valid_data = valid_data
        super().__init__()

    def on_train_begin(self, logs={}):
        self.metrics = {
            'val_f1': [],
            'val_roc_auc': [],
            'val_prc_auc': [],
        }

    def on_epoch_end(self, epoch, logs={}):
        val_prob = np.asarray(self.model.predict(self.valid_data[0]))
        val_predict = (val_prob).round()
        val_targ = self.valid_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        fpr, tpr, _ = roc_curve(val_targ, val_prob)
        _val_roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(val_targ, val_predict)
        _val_prc_auc = auc(recall, precision)
        self.metrics['val_f1'].append(_val_f1)
        self.metrics['val_roc_auc'].append(_val_roc_auc)
        self.metrics['val_prc_auc'].append(_val_prc_auc)
        print(f' - val_f1: {_val_f1:.3} - val_rocauc: {_val_roc_auc:.3} - val_prcauc:{_val_prc_auc:.3}')
        return


metrics = Metrics((valid_data_word2vec, valid_y.values))


batch_size = 64
epochs = 20
hidden_layer = 32
class_weight = {
    0: 0.34,
    1: 0.66
}


model = keras.Sequential()
model.add(keras.layers.LSTM(hidden_layer,
                            input_shape=(final_max, word_vec_length)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["binary_accuracy"]
)
history = model.fit(train_data_word2vec, train_y.values, validation_data=(valid_data_word2vec, valid_y.values),
                    epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                    shuffle=True, verbose=1, callbacks=[metrics])

##************************************************** ##
##                    Evaluation                     ##
##************************************************** ##


def plot_metrics(metrics):
    plt.plot(metrics.metrics['val_f1'])
    plt.plot(metrics.metrics['val_roc_auc'])
    plt.plot(metrics.metrics['val_prc_auc'])
    plt.title('metrics')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['f1', 'roc-auc', 'prc-auc'], loc='upper left')
    plt.show()

def plot_history(history):
    # Plot the accuracy
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()


def print_scores(model, word2vec, true_values):
    prediction = model.predict(word2vec)
    y_test_pred = prediction > 0.5
    f1_test = f1_score(true_values, y_test_pred)
    fpr, tpr, threshold = roc_curve(true_values, prediction)
    precision, recall, _ = precision_recall_curve(true_values, prediction)
    auroc = auc(fpr, tpr)
    aurprc = auc(recall, precision)
    print(f"The f1_score on the test_set was {f1_test}")
    print(f"The auroc on the test_set was {auroc}")
    print(f"The auprc on the test_set was {aurprc}")


plot_metrics(metrics)
# plot_history(history)
print_scores(model, test_data_word2vec, test_y.values)


##************************************************** ##
##                    Attention                      ##
##************************************************** ##

hidden_layer = 32
epochs = 15

inputs = keras.layers.Input(shape=(final_max, word_vec_length))
sequences = keras.layers.Bidirectional(
    keras.layers.LSTM(hidden_layer, return_sequences=True))(inputs)
sequences = keras.layers.LSTM(hidden_layer, return_sequences=True)(sequences)
seq_last = keras.layers.Lambda(lambda x: x[:, -1, :])(sequences)
# Attention
drop1 = keras.layers.Dropout(0.5)(seq_last)
dense1 = keras.layers.Dense(32, activation='relu')(drop1)
attention = keras.layers.Dense(final_max, activation="softmax")(dense1)
context = keras.layers.dot([attention, sequences], axes=1)

dropout = keras.layers.Dropout(0.5)(context)
dense2 = keras.layers.Dense(32, activation='relu')(dropout)
output = keras.layers.Dense(1, activation='sigmoid')(dense2)
model = keras.models.Model(inputs=inputs, outputs=[output])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history = model.fit(train_data_word2vec, train_y.values, validation_data=(valid_data_word2vec, valid_y.values),
                    epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                    shuffle=True, verbose=1, callbacks=[metrics])

# plot_history(history)
print_scores(model, test_data_word2vec, test_y.values)



## Elmo Keras

epochs = 8
combined_string = np.hstack([train_data_string, valid_data_string])
combined_y = np.hstack([train_y, valid_y])

elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=False)
keras.backend.get_session().run(tf.global_variables_initializer())
input_text = keras.layers.Input(shape=(1,), dtype='string')
embedding = keras.layers.Lambda(lambda x: elmo(keras.backend.squeeze(x, axis=1)))(input_text)
drop1 = keras.layers.Dropout(0.5)(embedding)
dense = keras.layers.Dense(128, activation='relu')(drop1)
pred = keras.layers.Dense(1, activation='sigmoid')(dense)

model = keras.models.Model(inputs=[input_text], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
history = model.fit(combined_string, combined_y, validation_split=0.25,
                    epochs=epochs, batch_size=batch_size, class_weight=class_weight,
                    shuffle=True, verbose=1)

plot_history(history)
print_scores(model, test_data_string, test_y.values)
