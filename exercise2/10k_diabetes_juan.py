import pandas as pd
import numpy as np
import pickle as pkl
from operator import itemgetter
from helper import EstimatorSelectionHelper
from sklearn.metrics import f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler

EMB_SIZE = 512
useDummy = True

# Importing dataframes
train_df = pd.read_csv('project2_data/10k_diabetes/diab_train.csv')
valid_df = pd.read_csv('project2_data/10k_diabetes/diab_validation.csv')
test_df = pd.read_csv('project2_data/10k_diabetes/diab_test.csv')

NaN_values = ['?', 'nan', 'Not Available', 'Not Mapped', 'None']

# The following columns provide no information or contain too many nans
to_remove = ['Unnamed: 0', 'weight', 'payer_code', 'discharge_disposition_id', 'admission_source_id',
             'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult']

# Prediction
pred_columns = ['readmitted']

# All integer columns
int_columns = ['time_in_hospital', 'num_lab_procedures',
               'num_procedures', 'num_medications', 'number_outpatient',
               'number_emergency', 'number_inpatient', 'number_diagnoses',
               ]

# All categorical columns
cat_columns = ['race', 'gender', 'age', 'metformin', 'repaglinide', 'nateglinide',
               'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
               'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
               'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
               'insulin', 'glyburide.metformin', 'glipizide.metformin',
               'glimepiride.pioglitazone', 'metformin.rosiglitazone',
               'metformin.pioglitazone', 'change', 'diabetesMed'
               ]


# All string columns
str_columns = ['diag_1_desc', 'diag_2_desc', 'diag_3_desc']


category_map = {'race': {'AfricanAmerican': 0, 'Caucasian': 1, 'Asian': 2, 'Other': 3, 'Hispanic': 4, '?': np.NaN},
                'gender': {'Male': 0, 'Female': 1},
                'age': {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9},
                'weight': {'?': np.NaN, '[0-25)': 1, '[25-50)': 2, '[50-75)': 3, '[75-100)': 4, '[100-125)': 5, '[125-150)': 6, '[150-175)': 7},
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
                'diabetesMed': {'No': 0, 'Yes': 1},
                'admission_type_id': {'Emergency': 0, 'Elective': 1, 'Urgent': 2, 'Newborn': 3, 'Not Available': np.NaN,
                                      'Not Mapped': np.NaN},
                'medical_specialty': {'?': np.NaN, 'Family/GeneralPractice': 1, 'Emergency/Trauma': 2, 'InternalMedicine': 3,
                                      'Psychiatry': 4, 'Surgery-Neuro': 5, 'Surgery-General': 6, 'Cardiology': 7,
                                      'Orthopedics': 8, 'Orthopedics-Reconstructive': 9, 'Nephrology': 10,
                                      'Pediatrics-Pulmonology': 11, 'Gastroenterology': 12,
                                      'Surgery-Cardiovascular/Thoracic': 13, 'Osteopath': 14,
                                      'PhysicalMedicineandRehabilitation': 15, 'Hematology/Oncology': 16,
                                      'Surgery-Vascular': 17, 'Pediatrics-Endocrinology': 18, 'Oncology': 19,
                                      'ObstetricsandGynecology': 20, 'Urology': 21, 'Neurology': 22, 'Pulmonology': 23,
                                      'Surgery-Cardiovascular': 24, 'Radiologist': 25, 'OutreachServices': 26,
                                      'Surgery-Plastic': 27, 'Endocrinology': 28, 'Ophthalmology': 29,
                                      'Obsterics&Gynecology-GynecologicOnco': 30, 'Radiology': 31,
                                      'Surgery-Thoracic': 32, 'Pediatrics': 33, 'Psychology': 34, 'Otolaryngology': 35,
                                      'InfectiousDiseases': 36, 'Pediatrics-CriticalCare': 37, 'Gynecology': 38,
                                      'Pediatrics-Hematology-Oncology': 39, 'Surgeon': 40, 'Podiatry': 41,
                                      'Obstetrics': 42, 'Anesthesiology-Pediatric': 43, 'Hospitalist': 44,
                                      'Hematology': 45, 'Pathology': 46, 'Surgery-Pediatric': 47,
                                      'Cardiology-Pediatric': 48, 'Surgery-Colon&Rectal': 49, 'PhysicianNotFound': 50,
                                      'Surgery-PlasticwithinHeadandNeck': 51, 'Pediatrics-EmergencyMedicine': 52}
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


def remove_columns(df, to_remove):
    return df.drop(columns=to_remove)


# remove uninteresting features
train_df = remove_columns(train_df, to_remove)
valid_df = remove_columns(valid_df, to_remove)
test_df = remove_columns(test_df, to_remove)

# Fix the Integers - Set all the unknown to 0
train_df = fix_integers(train_df, int_columns)
valid_df = fix_integers(valid_df, int_columns)
test_df = fix_integers(test_df, int_columns)

# Find category map
# category_map = create_maps(pd.concat((train_df, valid_df, test_df)), cat_columns, NaN_values)
# print(category_map)


if useDummy:
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
    train_data = pd.concat(
        [train_cat_dummies, train_df[int_columns]], axis=1)
    valid_data = pd.concat(
        [valid_cat_dummies, valid_df[int_columns]], axis=1)
    test_data = pd.concat(
        [test_cat_dummies, test_df[int_columns]], axis=1)
else:
    # Fix categories - Apply category map to integers
    train_data = fix_categories(train_df, cat_columns, category_map)[cat_columns + int_columns]
    valid_data = fix_categories(valid_df, cat_columns, category_map)[cat_columns + int_columns]
    test_data = fix_categories(test_df, cat_columns, category_map)[cat_columns + int_columns]


# get y_values
train_y = train_df[pred_columns[0]].values
valid_y = valid_df[pred_columns[0]].values
test_y = test_df[pred_columns[0]].values


# Fill missing values with median
train_data = train_data.fillna(
    train_data.median())
valid_data = valid_data.fillna(
    train_data.median())
test_data = test_data.fillna(
    train_data.median())


#################################################
# EVALUATION
################################################

models = {
    # 'AdaBoostClassifier': AdaBoostClassifier(base_estimator=LogisticRegression(solver='liblinear',
    #                                                                           class_weight='balanced')),
    'SVC': SVC(class_weight='balanced', gamma='auto')
    # 'LogisticRegression': LogisticRegression(solver='liblinear', class_weight='balanced'),
    # 'GaussianNB': GaussianNB(),
    # 'BernoulliNB': BernoulliNB(),
    # 'RandomForest': RandomForestClassifier(class_weight='balanced')
}

params = {
    # 'AdaBoostClassifier':  {'n_estimators': [8, 16, 32, 64, 128, 256]},
    'SVC': [
       {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
       {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100]},
    ],
    # 'LogisticRegression': {'C': [0.1, 1, 10, 50, 100]},
    # 'GaussianNB': {},
    # 'BernoulliNB': {},
    # 'RandomForest': {'n_estimators': [16, 32, 100]},
}

models2 = {
    'AdaBoostClassifier': Pipeline([('scaler', RobustScaler()),
                                    ('k_best', SelectKBest()),
                                    ('classifier', AdaBoostClassifier(base_estimator=LogisticRegression(solver='liblinear',
                                                                                                        class_weight='balanced')))]),
    'LogisticRegression': Pipeline([('scaler', RobustScaler()),
                                    ('k_best', SelectKBest()),
                                    ('classifier', LogisticRegression(solver='liblinear', class_weight='balanced'))]),

    'GaussianNB': Pipeline([('scaler', RobustScaler()),
                            ('k_best', SelectKBest()),
                            ('classifier', GaussianNB())]),

    'BernoulliNB': Pipeline([('scaler', RobustScaler()),
                            ('k_best', SelectKBest()),
                            ('classifier', BernoulliNB())]),

    'RandomForest': Pipeline([('scaler', RobustScaler()),
                              ('k_best', SelectKBest()),
                              ('classifier', RandomForestClassifier(class_weight='balanced'))])
}

params2 = {
    'AdaBoostClassifier':  {'k_best__k': [10, 20, 'all'],
                            'k_best__score_func': [f_classif, mutual_info_classif],
                            'classifier__n_estimators': [8, 16, 32, 64, 128, 256]},

    'LogisticRegression':  {'k_best__k': [10, 20, 'all'],
                            'k_best__score_func': [f_classif, mutual_info_classif],
                            'classifier__C': [0.1, 1, 10, 50, 100]},

    'GaussianNB':  {'k_best__k': [10, 20, 'all'],
                    'k_best__score_func': [f_classif, mutual_info_classif]},

    'BernoulliNB':  {'k_best__k': [10, 20, 'all'],
                     'k_best__score_func': [f_classif, mutual_info_classif]},

    'RandomForest':  {'k_best__k': [10, 20, 'all'],
                      'k_best__score_func': [f_classif, mutual_info_classif],
                      'classifier__n_estimators': [16, 32, 100]}
}


def evaluate(X_train, train_y, X_test, test_y, models, params):
    helper = EstimatorSelectionHelper(models, params)
    helper.fit(X_train, train_y, X_test, test_y,
               scoring=make_scorer(f1_score), n_jobs=-1)
    summary = helper.score_summary(sort_by='mean_score')
    # summary.to_pickle('summary')

    print('\nF1 test scores:')
    sortedKeysAndValues = sorted(helper.test_f1_scores.items(), key=lambda kv: -kv[1])
    for k, v in sortedKeysAndValues:
        print(k + ': ' + str(v))

    print('\nAUROC test scores:')
    sortedKeysAndValues = sorted(helper.test_auroc_scores.items(), key=lambda kv: -kv[1])
    for k, v in sortedKeysAndValues:
        print(k + ': ' + str(v))

    return summary


##################################################################

def flatten_diag(embs):
    diag_features = []
    embs = [np.zeros(EMB_SIZE) if np.isnan(x).all() else x for x in embs]
    for i in range(len(embs)):
        if i % 3 == 0:
            flattened = np.array(embs[i:i+3]).reshape(-1)
            diag_features.append(flattened)
    return diag_features


def average_diag(embs, weighted=False):
    diag_features = []
    weights = np.array([3, 2, 1])
    for i in range(len(embs)):
        if i % 3 == 0:
            aux = embs[i:i+3]
            nonans = []
            for j in range(3):
                if not np.isnan(aux[j]).all():
                    nonans.append(j)
            if weighted:
                averaged = np.dot(itemgetter(*nonans)(weights), itemgetter(*nonans)(aux))\
                           / np.sum(itemgetter(*nonans)(weights))
            else:
                averaged = np.average(itemgetter(*nonans)(aux), axis=0)
            diag_features.append(averaged)

    return diag_features


def avoid_bug(array):
    converted = np.zeros((array.shape[0], EMB_SIZE))
    for i in range(array.shape[0]):
        converted[i] = array[i]

    return converted


size_train = train_df.shape[0]
size_valid = valid_df.shape[0]
size_test = test_df.shape[0]

##########################################################
# Not using text features
#########################################################
X_train = train_data
X_valid = valid_data
X_test = test_data

X_train = np.concatenate((X_train, X_valid))
y_train = np.concatenate((train_y, valid_y))
y_test = test_y

summary = evaluate(X_train, y_train, X_test, y_test, models, params)
print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
AdaBoostClassifier: 0.5874532613211467
BernoulliNB: 0.5695266272189349
GaussianNB: 0.5656855707106964
LogisticRegression: 0.5455549420209829
RandomForest: 0.3197781885397412

AUROC test scores:
AdaBoostClassifier: 0.6367271203812147
RandomForest: 0.6338508761940385
LogisticRegression: 0.6293698695399159
BernoulliNB: 0.5884776801152588
GaussianNB: 0.497675915294452

             estimator min_score mean_score max_score   std_score
0   AdaBoostClassifier  0.530435   0.546866  0.573902    0.017315
1   AdaBoostClassifier  0.526161   0.538871  0.566092   0.0144569
5   AdaBoostClassifier  0.522197   0.534038  0.561765   0.0142826
6   LogisticRegression  0.521212   0.533173   0.56094   0.0143548
4   AdaBoostClassifier  0.513941   0.531505  0.563007   0.0168725
3   AdaBoostClassifier  0.511278   0.530767  0.560528   0.0162626
7   LogisticRegression  0.518803   0.529881  0.552632   0.0117573
2   AdaBoostClassifier  0.516272   0.528972  0.563277   0.0177625
8   LogisticRegression  0.514683   0.527916  0.550767   0.0120888
10  LogisticRegression  0.514286   0.527868  0.550512   0.0120769
9   LogisticRegression  0.514286   0.527688  0.549306   0.0115874
11          GaussianNB  0.266509   0.438355  0.566214   0.0969709
12         BernoulliNB  0.356053   0.436261  0.583465   0.0821994
15        RandomForest  0.407669   0.415605  0.429134  0.00757444
13        RandomForest  0.386386   0.393849  0.407692  0.00804888
14        RandomForest     0.374   0.392641  0.408357   0.0113138

MODELS2###############################################

F1 test scores:
GaussianNB: 0.5656855707106964
BernoulliNB: 0.5655105973025047
AdaBoostClassifier: 0.5654496883348175
LogisticRegression: 0.5220012055455092
RandomForest: 0.420249653259362

AUROC test scores:
LogisticRegression: 0.6376099486914812
BernoulliNB: 0.6219144105788951
AdaBoostClassifier: 0.6130119490028219
RandomForest: 0.5808404316560292
GaussianNB: 0.497675915294452

             estimator  min_score mean_score max_score   std_score
37  LogisticRegression   0.532934   0.543234  0.567661   0.0133698
43  LogisticRegression   0.521138   0.541518  0.568336   0.0167869
49  LogisticRegression   0.526233   0.539031  0.556104   0.0101656
55  LogisticRegression   0.508941   0.537529  0.557401   0.0169317
27  AdaBoostClassifier   0.522264   0.534745  0.566423   0.0161469
61  LogisticRegression   0.520486   0.534693   0.54454   0.0102694
0   AdaBoostClassifier   0.507486   0.534066  0.558209   0.0162626
35  AdaBoostClassifier   0.521148   0.533744  0.559647   0.0137997
34  AdaBoostClassifier   0.521148   0.533744  0.559647   0.0137997
3   AdaBoostClassifier   0.510574   0.532603  0.544693   0.0130306
41  LogisticRegression   0.519084    0.53225  0.559471    0.014172
40  LogisticRegression   0.519084    0.53225  0.559471    0.014172
24  AdaBoostClassifier     0.5152   0.532007  0.547826   0.0131447

# NO DUMMY #####################################################
F1 test scores:
LogisticRegression: 0.5248484848484849
AdaBoostClassifier: 0.5136417556346381
RandomForest: 0.43959469992205763
BernoulliNB: 0.41346906812842593
GaussianNB: 0.28185328185328185

AUROC test scores:
RandomForest: 0.6465468875861803
LogisticRegression: 0.6419718518812602
BernoulliNB: 0.6195015206587048
GaussianNB: 0.618141756107448
AdaBoostClassifier: 0.614639696348852

             estimator min_score mean_score max_score   std_score
0   AdaBoostClassifier  0.506523   0.531335  0.555024   0.0158465
7   LogisticRegression  0.506329   0.529763  0.565954   0.0196249
2   AdaBoostClassifier    0.5004   0.529359  0.557471   0.0214352
8   LogisticRegression  0.507911   0.528793   0.56448   0.0189867
9   LogisticRegression  0.509091   0.528346  0.564065   0.0187251
10  LogisticRegression  0.509091   0.528269  0.564065   0.0187583
3   AdaBoostClassifier  0.494364   0.527256  0.560115   0.0234676
6   LogisticRegression  0.506751   0.526992  0.560886   0.0185284
5   AdaBoostClassifier  0.501986   0.525509  0.564103   0.0209083
4   AdaBoostClassifier       0.5   0.525216  0.563093    0.021468
1   AdaBoostClassifier  0.498805   0.524516  0.555004   0.0197211
12         BernoulliNB  0.411321   0.432369  0.447059   0.0141014
15        RandomForest  0.396777   0.424449  0.459078   0.0200032
14        RandomForest   0.39204   0.411492  0.437438   0.0175911
13        RandomForest  0.389328   0.397299  0.414658  0.00916498
11          GaussianNB  0.157025    0.30612  0.451781    0.100223

# NO DUMMY, MODELS2 ############################################
F1 test scores:
GaussianNB: 0.5682551056968829
AdaBoostClassifier: 0.5411230856494612
LogisticRegression: 0.5177725118483413
RandomForest: 0.437094682230869
BernoulliNB: 0.4362264150943397

AUROC test scores:
AdaBoostClassifier: 0.6467929302691007
BernoulliNB: 0.6371612211657304
LogisticRegression: 0.6353804154203464
GaussianNB: 0.6253856497041741
RandomForest: 0.5750936895014476

             estimator  min_score mean_score max_score   std_score
0   AdaBoostClassifier   0.523625   0.542704  0.570803     0.01562
43  LogisticRegression   0.511111   0.539553  0.566667   0.0182596
25  AdaBoostClassifier   0.528361   0.538958  0.549823  0.00680492
49  LogisticRegression   0.530226   0.538069  0.555472  0.00915565
1   AdaBoostClassifier    0.52194   0.537559  0.551203   0.0113838
55  LogisticRegression   0.515879     0.5354  0.554539   0.0128568
6   AdaBoostClassifier   0.520416   0.535333  0.571643   0.0184685
33  AdaBoostClassifier   0.521671   0.533699  0.553623   0.0108682
2   AdaBoostClassifier   0.514062   0.533335  0.564364   0.0175183
63  LogisticRegression   0.516693   0.532942  0.560993   0.0157353
39  LogisticRegression    0.51913   0.532491  0.565627   0.0168375
3   AdaBoostClassifier   0.505564   0.530928  0.561295   0.0180779
26  AdaBoostClassifier   0.520588   0.530578  0.560816   0.0153137
36  LogisticRegression   0.513922   0.530461  0.576751   0.0233657
5   AdaBoostClassifier   0.511554   0.530387  0.557549   0.0151863
'''

##########################################################
# Only text features (unweighted average)
#########################################################
# with open('project2_data/10k_diabetes/uni_emb.pkl', 'rb') as input_file:
#     sentence_embs = pkl.load(input_file)
#
# embs = np.array(average_diag(sentence_embs, weighted=False))
# embs = avoid_bug(embs)
#
# embs_train_unweighted = embs[:size_train]
# embs_valid_unweighted = embs[size_train:(size_train + size_valid)]
# embs_test_unweighted = embs[(size_train + size_valid):]
#
# X_train = embs_train_unweighted
# X_valid = embs_valid_unweighted
# X_test = embs_test_unweighted
#
# X_train = np.concatenate((X_train, X_valid))
# y_train = np.concatenate((train_y, valid_y))
# y_test = test_y
#
# summary = evaluate(X_train, y_train, X_test, y_test, models, params)
# print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
AdaBoostClassifier: 0.5322503583373148
GaussianNB: 0.49640685461580975
LogisticRegression: 0.4954545454545455
BernoulliNB: 0.472663139329806
RandomForest: 0.3623082542001461

AUROC test scores:
LogisticRegression: 0.578699181215921
GaussianNB: 0.5638854266463703
BernoulliNB: 0.5638613970000554
AdaBoostClassifier: 0.5635834889165868
RandomForest: 0.5159645656745906

             estimator min_score mean_score max_score   std_score
0   AdaBoostClassifier  0.511515   0.530508  0.552359   0.0142265
1   AdaBoostClassifier  0.498024   0.510767  0.532992   0.0124417
7   LogisticRegression  0.479422   0.506101  0.532486    0.017845
8   LogisticRegression  0.486757   0.505047  0.520848   0.0128776
9   LogisticRegression  0.482759   0.501851  0.533049   0.0173668
10  LogisticRegression  0.482066   0.500527  0.536344   0.0190954
2   AdaBoostClassifier  0.484637   0.499114  0.520791   0.0130127
11          GaussianNB   0.48051   0.496723  0.509966   0.0114035
6   LogisticRegression  0.472464   0.495158  0.528622   0.0194232
5   AdaBoostClassifier  0.482051   0.493945  0.521117   0.0142447
3   AdaBoostClassifier  0.478105   0.493584  0.524245   0.0163821
4   AdaBoostClassifier  0.480811   0.492679   0.52242   0.0153321
12         BernoulliNB  0.475645   0.488597  0.511355   0.0147687
13        RandomForest  0.353266   0.362788  0.370776  0.00573769
14        RandomForest  0.346743   0.362565  0.377358   0.0119951
15        RandomForest  0.321285   0.338326   0.36811   0.0166037

MODELS2#####################################################
F1 test scores:
AdaBoostClassifier: 0.5061728395061729
LogisticRegression: 0.49944008958566627
GaussianNB: 0.49640685461580975
BernoulliNB: 0.4753416518122401
RandomForest: 0.32006010518407213

AUROC test scores:
AdaBoostClassifier: 0.5801211094174273
LogisticRegression: 0.5711089472820903
GaussianNB: 0.5638843818791393
BernoulliNB: 0.5610353016399712
RandomForest: 0.5098678264975955

             estimator min_score mean_score max_score   std_score
44  LogisticRegression  0.491709   0.510054   0.52669   0.0130148
62  LogisticRegression  0.491354   0.510051  0.527066   0.0133456
50  LogisticRegression  0.491354   0.510051  0.527066   0.0133456
56  LogisticRegression  0.491354   0.510051  0.527066   0.0133456
10  AdaBoostClassifier  0.488539   0.508716  0.531449   0.0146922
11  AdaBoostClassifier  0.488539   0.508716  0.531449   0.0146922
4   AdaBoostClassifier  0.479827    0.50862  0.533712   0.0196948
5   AdaBoostClassifier  0.479827    0.50862  0.533712   0.0196948
38  LogisticRegression  0.488825   0.508195   0.53041    0.014418
32  AdaBoostClassifier   0.48876   0.507263  0.526912   0.0134086
26  AdaBoostClassifier  0.487699   0.506356  0.527797   0.0141116
16  AdaBoostClassifier  0.488889   0.505354  0.523404    0.011908
17  AdaBoostClassifier  0.488889   0.505354  0.523404    0.011908
20  AdaBoostClassifier  0.483755   0.505102  0.528996   0.0163443
8   AdaBoostClassifier   0.48191   0.503339  0.521927   0.0133539
22  AdaBoostClassifier  0.485981   0.502375   0.53267   0.0161919
23  AdaBoostClassifier  0.485981   0.502375   0.53267   0.0161919
14  AdaBoostClassifier  0.483707   0.502066  0.526615    0.015813
45  LogisticRegression  0.490858   0.501402  0.516349  0.00938521
'''

##########################################################
# Only text features (weighted average)
#########################################################
# embs = np.array(average_diag(sentence_embs, weighted=True))
# embs = avoid_bug(embs)
#
# embs_train_weighted = embs[:size_train]
# embs_valid_weighted = embs[size_train:(size_train + size_valid)]
# embs_test_weighted = embs[(size_train + size_valid):]
#
# X_train = embs_train_weighted
# X_valid = embs_valid_weighted
# X_test = embs_test_weighted
#
# X_train = np.concatenate((X_train, X_valid))
# y_train = np.concatenate((train_y, valid_y))
# y_test = test_y
#
# summary = evaluate(X_train, y_train, X_test, y_test, models, params)
# print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
LogisticRegression: 0.49520586576424136
AdaBoostClassifier: 0.49229074889867847
GaussianNB: 0.48763657274295574
BernoulliNB: 0.4520123839009288
RandomForest: 0.3214013709063214

AUROC test scores:
LogisticRegression: 0.5732679587651269
GaussianNB: 0.5674402471501362
BernoulliNB: 0.5607411996644207
AdaBoostClassifier: 0.5565986975931697
RandomForest: 0.5139748064829897

             estimator min_score mean_score max_score   std_score
7   LogisticRegression  0.480922   0.508641  0.529774   0.0204529
8   LogisticRegression  0.473304   0.508299  0.538947   0.0225581
9   LogisticRegression  0.486061   0.506692  0.533708   0.0176008
10  LogisticRegression   0.48433   0.506331  0.535014   0.0193925
0   AdaBoostClassifier  0.481246   0.494639  0.514604   0.0122227
6   LogisticRegression  0.471698   0.492131  0.512563   0.0153815
1   AdaBoostClassifier  0.474576   0.488977  0.506762   0.0118357
5   AdaBoostClassifier  0.468773   0.485587  0.506182   0.0143926
2   AdaBoostClassifier  0.469653   0.481512  0.496741   0.0114713
4   AdaBoostClassifier  0.464259   0.480756  0.500366   0.0151722
3   AdaBoostClassifier  0.464364   0.480586   0.49744   0.0130239
11          GaussianNB  0.464023   0.480016  0.500734   0.0163299
12         BernoulliNB  0.439024   0.451054  0.473765   0.0147445
13        RandomForest  0.330171   0.341964  0.354478  0.00902822
14        RandomForest  0.317613   0.341919   0.36194   0.0171819
15        RandomForest  0.303091   0.325665  0.338028   0.0120941

MODELS2##############################################################
F1 test scores:
LogisticRegression: 0.505849582172702
AdaBoostClassifier: 0.5042492917847025
GaussianNB: 0.48763657274295574
BernoulliNB: 0.44458052663808934
RandomForest: 0.319634703196347

AUROC test scores:
LogisticRegression: 0.5771670300715352
AdaBoostClassifier: 0.5767470336446392
GaussianNB: 0.5674402471501362
BernoulliNB: 0.5494587583359365
RandomForest: 0.5154155404946555

             estimator min_score mean_score max_score   std_score
40  LogisticRegression  0.486141   0.510349  0.537439   0.0187109
41  LogisticRegression  0.486141   0.510349  0.537439   0.0187109
17  AdaBoostClassifier  0.483008   0.508863  0.539106   0.0192737
16  AdaBoostClassifier  0.483008   0.508863  0.539106   0.0192737
47  LogisticRegression  0.489796   0.507322  0.530441   0.0141216
46  LogisticRegression  0.489796   0.507322  0.530441   0.0141216
34  AdaBoostClassifier  0.483548   0.506887  0.535764   0.0195621
35  AdaBoostClassifier  0.483548   0.506887  0.535764   0.0195621
'''

##########################################################
# Only text features (concatenation)
#########################################################
# embs = np.array(flatten_diag(sentence_embs))
#
# embs_train_concat = embs[:size_train]
# embs_valid_concat = embs[size_train:(size_train + size_valid)]
# embs_test_concat = embs[(size_train + size_valid):]
#
# X_train = embs_train_concat
# X_valid = embs_valid_concat
# X_test = embs_test_concat
#
# X_train = np.concatenate((X_train, X_valid))
# y_train = np.concatenate((train_y, valid_y))
# y_test = test_y
#
# summary = evaluate(X_train, y_train, X_test, y_test, models, params)
# print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
AdaBoostClassifier: 0.5018607123870281
LogisticRegression: 0.4999999999999999
GaussianNB: 0.49914529914529915
BernoulliNB: 0.4936268829663963
RandomForest: 0.31145038167938927

AUROC test scores:
LogisticRegression: 0.5795255920957091
BernoulliNB: 0.5757915940118121
GaussianNB: 0.5708138005393087
AdaBoostClassifier: 0.5635594592702718
RandomForest: 0.5182442477728174

             estimator min_score mean_score max_score   std_score
0   AdaBoostClassifier  0.491136   0.512744  0.532985   0.0148231
1   AdaBoostClassifier  0.481559   0.501545  0.521438   0.0163978
7   LogisticRegression  0.483824   0.501165   0.52982   0.0165524
8   LogisticRegression   0.47509   0.498774  0.533239   0.0224748
6   LogisticRegression  0.473451     0.4975  0.530612   0.0188627
12         BernoulliNB  0.480995    0.49701  0.515991   0.0122793
11          GaussianNB  0.479238   0.496804  0.507358   0.0109422
4   AdaBoostClassifier  0.474378    0.49505  0.513958   0.0130647
2   AdaBoostClassifier  0.473272   0.494191  0.514768   0.0147421
5   AdaBoostClassifier  0.469978   0.494167  0.518728   0.0156175
3   AdaBoostClassifier   0.47619   0.493552  0.517045   0.0150273
9   LogisticRegression  0.476744   0.493412  0.514205   0.0169042
10  LogisticRegression  0.466276   0.491135  0.514286   0.0187179
13        RandomForest   0.35175   0.359461  0.363299  0.00402223
15        RandomForest  0.344423   0.354053  0.371212  0.00970245
14        RandomForest  0.346767   0.352162  0.362069  0.00526926
'''

##########################################################
# All features (unweighted average)
#########################################################
# X_train = np.concatenate((embs_train_unweighted, train_data), axis=1)
# X_valid = np.concatenate((embs_valid_unweighted, valid_data), axis=1)
# X_test = np.concatenate((embs_test_unweighted, test_data), axis=1)
#
# X_train = np.concatenate((X_train, X_valid))
# y_train = np.concatenate((train_y, valid_y))
# y_test = test_y
#
# summary = evaluate(X_train, y_train, X_test, y_test, models, params)
# print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
AdaBoostClassifier: 0.5867112411199331
GaussianNB: 0.5662953647143371
BernoulliNB: 0.5438596491228069
LogisticRegression: 0.32339656729900634
RandomForest: 0.2920426579163249

AUROC test scores:
AdaBoostClassifier: 0.6370102523008387
LogisticRegression: 0.6283292813777555
BernoulliNB: 0.5804873003319225
RandomForest: 0.5244099415870641
GaussianNB: 0.49891083016159415

             estimator min_score mean_score max_score  std_score
0   AdaBoostClassifier  0.533333   0.547032  0.572846  0.0164239
7   LogisticRegression  0.521061    0.54357  0.561224  0.0159811
9   LogisticRegression   0.49882   0.540563  0.582791  0.0275911
1   AdaBoostClassifier  0.526937   0.539376  0.569584  0.0157467
10  LogisticRegression  0.495645   0.539101  0.580877  0.0280951
6   LogisticRegression  0.522957   0.538692  0.564706  0.0145114
8   LogisticRegression  0.505114   0.537425  0.563991  0.0206706
5   AdaBoostClassifier  0.522481   0.535463  0.559471  0.0126349
4   AdaBoostClassifier  0.522214   0.534238  0.566814  0.0164504
3   AdaBoostClassifier  0.506042   0.531317  0.566837  0.0196752
2   AdaBoostClassifier   0.51742   0.530322  0.568928  0.0194902
11          GaussianNB  0.465625   0.507899  0.553237  0.0302297
12         BernoulliNB  0.473282   0.507594  0.537392  0.0240907
13        RandomForest  0.316872   0.342064  0.368794  0.0221013
14        RandomForest  0.296142   0.324106  0.362189  0.0217731
15        RandomForest   0.28328   0.319776  0.354037  0.0233832
'''

##########################################################
# All features (weighted average)
#########################################################
# X_train = np.concatenate((embs_train_weighted, train_data), axis=1)
# X_valid = np.concatenate((embs_valid_weighted, valid_data), axis=1)
# X_test = np.concatenate((embs_test_weighted, test_data), axis=1)
#
# X_train = np.concatenate((X_train, X_valid))
# y_train = np.concatenate((train_y, valid_y))
# y_test = test_y
#
# summary = evaluate(X_train, y_train, X_test, y_test, models, params)
# print(summary[['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']])

'''
F1 test scores:
AdaBoostClassifier: 0.5867112411199331
GaussianNB: 0.5656855707106964
BernoulliNB: 0.5355566454144188
LogisticRegression: 0.31444241316270566
RandomForest: 0.305532617671346

AUROC test scores:
AdaBoostClassifier: 0.637052042990082
LogisticRegression: 0.6276366007035462
BernoulliNB: 0.5803964055828181
RandomForest: 0.5330961363463028
GaussianNB: 0.497675915294452

             estimator min_score mean_score max_score  std_score
0   AdaBoostClassifier  0.532562    0.54747  0.572315  0.0163809
7   LogisticRegression  0.515128   0.545239  0.562682   0.017065
8   LogisticRegression   0.50428   0.542921  0.575977  0.0265765
9   LogisticRegression  0.509743   0.542494  0.574946  0.0237553
10  LogisticRegression   0.50625   0.542397  0.579023  0.0262075
6   LogisticRegression  0.532319   0.541015  0.569129  0.0140853
5   AdaBoostClassifier  0.530827    0.53971  0.565538  0.0130636
1   AdaBoostClassifier  0.526549   0.538227  0.567742  0.0152466
4   AdaBoostClassifier  0.521148   0.535088  0.568902  0.0173377
2   AdaBoostClassifier  0.516654   0.530567  0.566423  0.0183502
3   AdaBoostClassifier  0.508679   0.530252  0.565982  0.0191116
11          GaussianNB  0.453711   0.498499  0.573635  0.0419829
12         BernoulliNB  0.455556   0.481806  0.514324  0.0204377
13        RandomForest  0.319756   0.344364  0.357285  0.0136404
14        RandomForest  0.312185   0.324172  0.340122  0.0100427
15        RandomForest   0.29171   0.318466  0.330579  0.0150992
'''
