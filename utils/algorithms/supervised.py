# supervised.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb


class SupervisedLearning:

    # Logistic Regression
    def logistic_regression(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def lasso_regression(self, X_train, y_train, alpha=1.0):
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        return model

    def elastic_net(self, X_train, y_train, alpha=1.0, l1_ratio=0.5):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        return model

    # Support Vector Machine
    def svm_classifier(self, X_train, y_train, kernel='linear', C=1.0):
        model = SVC(kernel=kernel, C=C)
        model.fit(X_train, y_train)
        return model

    # K-Nearest Neighbors Classifier
    def knn_classifier(self, X_train, y_train, n_neighbors=5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        return model

    # Decision Tree Classifier
    def decision_tree_classifier(self, X_train, y_train):
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        return model

    # Random Forest Classifier
    def random_forest_classifier(self, X_train, y_train, n_estimators=100):
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    # Gradient Boosting Classifier
    def gradient_boosting_classifier(self, X_train, y_train, n_estimators=100):
        model = GradientBoostingClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    # AdaBoost Classifier
    def adaboost_classifier(self, X_train, y_train, n_estimators=50):
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    # XGBoost Classifier
    def xgboost_classifier(self, X_train, y_train, n_estimators=100):
        model = xgb.XGBClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    # LightGBM Classifier
    def lightgbm_classifier(self, X_train, y_train, n_estimators=100):
        model = lgb.LGBMClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model








