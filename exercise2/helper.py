import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, make_scorer

from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.test_f1_scores = {}
        self.test_auroc_scores = {}

    def fit(self, x_train, y_train, x_test, y_test, cv=5, n_jobs=-1, verbose=1, scoring=None, refit=True):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(x_train, y_train)
            self.grid_searches[key] = gs

            scoring = make_scorer(f1_score)
            test_f1_score = scoring(gs, x_test, y_test)
            self.test_f1_scores[key] = test_f1_score

            if key == 'SVC':
                predict_score = gs.decision_function(x_test)
            else:
                predict_score = gs.predict_proba(x_test)[:, 1]

            test_auroc_score = roc_auc_score(y_test, predict_score)
            self.test_auroc_scores[key] = test_auroc_score

    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params, **d})

        rows = []
        for k in self.grid_searches:
            # print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]
                scores.append(r.reshape(len(params), 1))

            all_scores = np.hstack(scores)
            for p, s in zip(params, all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1, sort=True).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
