from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
import string

# Importing dataframes
train_df = pd.read_csv('project2_data/10k_diabetes/diab_train.csv')
valid_df = pd.read_csv('project2_data/10k_diabetes/diab_validation.csv')
test_df = pd.read_csv('project2_data/10k_diabetes/diab_test.csv')

NaN_values = ['?', 'nan', 'Not Available', 'Not Mapped', 'None']

# Prediction
pred_columns = ['Unnamed: 0']

# All integer columns
int_columns = ['time_in_hospital', 'num_lab_procedures',
               'num_procedures', 'num_medications', 'number_outpatient',
               'number_emergency', 'number_inpatient', 'number_diagnoses',
               'readmitted', 'diag_1', 'diag_2', 'diag_3'
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
str_columns = ['diag_1_desc', 'diag_2_desc', 'diag_3_desc', 'payer_code',
               'admission_source_id', 'medical_specialty', 'admission_type_id',
               'discharge_disposition_id'
               ]

category_map = {'race': {'AfricanAmerican': 0, 'Caucasian': 1, 'Asian': 2, 'Other': 3, 'Hispanic': 4, '?': np.NaN},
                'gender': {'Male': 0, 'Female': 1},
                'age': {'[60-70)': 0, '[70-80)': 1, '[80-90)': 2, '[50-60)': 3, '[20-30)': 4, '[40-50)': 5, '[30-40)': 6, '[10-20)': 7, '[90-100)': 8, '[0-10)': 9},
                'weight': {'?': np.NaN, '[75-100)': 1, '[100-125)': 2, '[50-75)': 3, '[125-150)': 4, '[25-50)': 5, '[150-175)': 6, '[0-25)': 7},
                'max_glu_serum': {'None': np.NaN, 'Norm': 1, '>200': 2, '>300': 3},
                'A1Cresult': {'None': 0, '>8': 1, 'Norm': 2, '>7': 3},
                'metformin': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'repaglinide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'nateglinide': {'No': 0, 'Steady': 1, 'Up': 2},
                'chlorpropamide': {'No': 0, 'Steady': 1},
                'glimepiride': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'acetohexamide': {'No': 0},
                'glipizide': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'glyburide': {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3},
                'tolbutamide': {'No': 0, 'Steady': 1},
                'pioglitazone': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'rosiglitazone': {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3},
                'acarbose': {'No': 0, 'Steady': 1, 'Up': 2},
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
        df[column] = df[column].replace(cat_map[column])
    return df


# Fix the Integers - Set all the unknown to 0
train_df = fix_integers(train_df, int_columns)
valid_df = fix_integers(valid_df, int_columns)
test_df = fix_integers(test_df, int_columns)

# Find category map
# category_map = create_maps(train_df, cat_columns, NaN_values)
# print(category_map)

# Fix Categories
train_df = fix_categories(train_df, cat_columns, category_map)
valid_df = fix_categories(valid_df, cat_columns, category_map)
test_df = fix_categories(test_df, cat_columns, category_map)


# Create data for  later fixing
train_data_numerical = train_df[cat_columns + int_columns]
valid_data_numerical = valid_df[cat_columns + int_columns]
test_data_numerical = test_df[cat_columns + int_columns]

# get y_values
train_y = train_df[pred_columns[0]]
valid_y = valid_df[pred_columns[0]]
test_y = test_df[pred_columns[0]]


# Fill missing values with median
train_data_numerical = train_data_numerical.fillna(
    train_data_numerical.median())
valid_data_numerical = valid_data_numerical.fillna(
    train_data_numerical.median())
test_data_numerical = test_data_numerical.fillna(train_data_numerical.median())

##************************************************** ##
#                Testing Linear Regression           ##
##************************************************** ##

reg = LinearRegression().fit(train_data_numerical.values, train_y)
score = reg.score(valid_data_numerical.values, valid_y)
print(f'The score on the validation set was {score:.3f}')

##************************************************** ##
#                Testing SVM                         ##
##************************************************** ##

svc = SVR(gamma='scale').fit(
    train_data_numerical.values,
    train_y.values.ravel())
score = svc.score(valid_data_numerical.values, valid_y.values.ravel())
print(f'The score on the validation set was {score:.3f}')


##************************************************** ##
#                String                              ##
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
