import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf

## Importing dataframes
train_df = pd.read_csv('project2_data/10k_diabetes/diab_train.csv')
valid_df = pd.read_csv('project2_data/10k_diabetes/diab_validation.csv')
test_df  = pd.read_csv('project2_data/10k_diabetes/diab_test.csv')

NaN_values = ['?', 'nan', 'Not Available', 'Not Mapped', 'None']

int_columns = ['Unnamed: 0', 'time_in_hospital', 'num_lab_procedures',
       'num_procedures', 'num_medications', 'number_outpatient',
       'number_emergency', 'number_inpatient', 'number_diagnoses',
       'readmitted', 'diag_1', 'diag_2', 'diag_3']

cat_columns = ['race', 'gender', 'age', 'weight',
       'max_glu_serum',
       'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide.metformin', 'glipizide.metformin',
       'glimepiride.pioglitazone', 'metformin.rosiglitazone',
       'metformin.pioglitazone', 'change', 'diabetesMed']

str_columns = [ 'diag_1_desc', 'diag_2_desc', 'diag_3_desc', 'payer_code', 
        'admission_source_id', 'medical_speciality', 'admission_type_id',
        'discharge_disposition_id'
        ]

def create_maps(df, cat_columns, nan_categories):
    maps = {}
    for column in cat_columns:
        maps[column] = {category: i for i, category in enumerate(df[column].unique())}
    
    for key in maps.keys():
        for category in maps[key]:
            if category in nan_categories:
                maps[key][category] = np.NaN
    return maps

def fix_integers(df, int_columns):
    for column in int_columns:
        df[column] = df[column].where(lambda x: [str(i).isdigit() for i in x], np.NaN).astype(np.float)
    return df

def fix_categories(df, cat_columns, cat_map):
    for column in cat_columns:
        df[column] = df[column].replace(cat_map[column])
    return df

## Fix the Integers - Set all the unknown to 0
train_df = fix_integers(train_df, int_columns)
valid_df = fix_integers(valid_df, int_columns)
test_df = fix_integers(test_df, int_columns)

## Find category map
category_map = create_maps(train_df, cat_columns, NaN_values)

### Fix Categories
train_df = fix_categories(train_df, cat_columns, category_map)
valid_df = fix_categories(valid_df, cat_columns, category_map)
test_df = fix_categories(test_df, cat_columns, category_map)
