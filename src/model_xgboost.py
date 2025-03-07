import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV


class ModelXGBoost:

    def __init__(self, n_estimators = 50, max_depth = 6,
                 n_jobs=1, early_stopping_rounds=10, eval_metric="mae"):
        """Initialized an instance of XGBoost model class with user chosen hyperparams.
        Parameters
        ------------
        n_estimators: int
            Number of trees in the ensemble.
        max_depth: int
            Maximum depth of each tree.
        n_jobs: int
            Number of parallel threads.
        early_stopping_rounds: int
            Evaluation metric on the validation set must go down at least every "early_stopping_rounds".
            If training is stopped early, the best performing model is returned.
        eval_metric: str
            Evaluation metric used to estimate loss.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

    def train_model(self, features = None, target = None):
        """Train an XGBoost model with known hyperparams.
        Parameters
        ------------
        features: pandas.DataFrame, array
            Input data (target must be removed).
        target: pandas.DataFrame, array
            Target variable.

        Returns
        ------------
        xgboost.sklearn.XGBRegressor object.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0,
                                                           test_size=0.2)

        clf = xgb.XGBRegressor(n_jobs=self.n_jobs, early_stopping_rounds=self.early_stopping_rounds,
                               eval_metric = self.eval_metric,
                               max_depth = self.max_depth, n_estimators = self.max_depth)

        return clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])

    def grid_search(self, features = None, target = None,
                    max_depth = None, n_estimators = None,
                    eval_metric = 'neg_mean_absolute_error'):

        """Use grid search to estimate optimal hyperparmeters.
        Parameters
        ------------
        features: pandas.DataFrame, array
            Input data (target must be removed).
        target: pandas.DataFrame, array
            Target variable.
        max_depth: int or list of int
            Potential values of max depth to try.
        n_estimators: int or list of int
            Potential number of trees in the ensemble.
        eval_metric: str
            Metric used to compare different hyperparam combinations.
            Refer to sklearn.model_selection.GridSearchCV "scoring" param.

        Returns
        ------------
        GridSearchCV object, x_test, y_test
        The .predict method will use the best performing model.
        Refer to sklearn.model_selection.GridSearchCV docs for full description.
        Test set is also returned for final model evaluation.
        """

        x_train, x_test, y_train, y_test = train_test_split(features,
                                                            target, random_state=0,
                                                           test_size=0.2)

        xgb_model = xgb.XGBRegressor(n_jobs=2)
        clf = GridSearchCV(
            xgb_model,
            {"max_depth": max_depth, "n_estimators": n_estimators},
            n_jobs=self.n_jobs,
            cv=3,
            scoring=eval_metric
        )

        return clf.fit(x_train, y_train), x_test, y_test

    def train_loc_models(self, features = None, target = None,
                         loc_list = None, use_grid_search = True,
                         max_depth=None, n_estimators=None,
                         eval_metric='neg_mean_absolute_error'
                         ):
        """Train a separate model for each location.
        Parameters
        ------------
        features: pandas.DataFrame
            Input data (target must be removed).
        target: pandas.DataFrame
            Target variable.
        loc_list: list of str
            List of locations to fit a model to.
            These must all be present in "features" and "target" DataFrames.

        use_grid_search: bool
            If True, run grid search for each location in loc_list.
            All params below are only used in this case, otherwise they have no effect.
        max_depth: int or list of int
            Potential values of max depth to try.
        n_estimators: int or list of int
            Potential number of trees in the ensemble.
        eval_metric: str
            Metric used to compare different hyperparam combinations.
            Refer to sklearn.model_selection.GridSearchCV "scoring" param.

        Returns
        ------------
        A dict containing xgboost.sklearn.XGBRegressor objects.
        If `use_grid_search` is True, then the test set for each location is also included in the list.
        """

        model_list = {}
        x_test_list = []
        y_test_list = []

        for loc in loc_list:
            print('LOCATION:', loc)
            features_loc = features[features['location'] == loc].drop(columns=['location'])
            target_loc = target[target['location'] == loc].drop(columns=['location'])

            if use_grid_search:
                model,x_test,y_test = self.grid_search(features = features_loc, target = target_loc,
                                 max_depth=max_depth, n_estimators=n_estimators,
                                 eval_metric=eval_metric)
                model_list[loc] = model.best_estimator_
                x_test_list.append(x_test)
                y_test_list.append(y_test)

                print('Best score:', model.best_score_)
                print('Best params:', model.best_params_)
            else:
                model = self.train_model(features=features_loc, target=target_loc)
                model_list[loc] = model.best_estimator_

                print('Best score:', model.best_score_)
                print('Best params:', model.best_params_)

            print('----------------------------------------')

        if use_grid_search:
            return model_list, x_test_list, y_test_list
        else:
            return model_list