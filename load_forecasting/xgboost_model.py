"""Minimal XGBoost model wrapper."""

import joblib
from xgboost import XGBRegressor


class XGBoostModel:
    """Wrapper around XGBRegressor with fit/predict/save/load only."""

    def __init__(self, **params):
        self.params = params
        self.model = XGBRegressor(**params)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path):
        instance = cls.__new__(cls)
        instance.model = joblib.load(path)
        instance.params = instance.model.get_params()
        return instance

    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        self.model.set_params(**params)
        self.params.update(params)
        return self

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

