import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
train = pd.read_csv('train_Dataset.csv')
test = pd.read_csv('test_Dataset.csv')

# Extract features and labels
X_train = train.drop(columns=['Class'])
y_train = train['Class']
X_test = test.drop(columns=['Class'])
y_test = test['Class']

# Feature Porject
clf_fs = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf_fs.fit(X_train, y_train)
selection = SelectFromModel(clf_fs, threshold=0.01, prefit=True)
X_train_selected = selection.transform(X_train)
X_test_selected = selection.transform(X_test)

# Poly Feature
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)


ros = RandomOverSampler()
X_train_ros, y_train_ros = ros.fit_resample(X_train_poly, y_train)

# XGB with grid search of parameters
clf = XGBClassifier(use_label_encoder=False)
parameters = {
    'n_estimators': [200, 500, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [8, 10, 12],
    'eval_metric': ['logloss', 'auc']
}
clf_gs = GridSearchCV(clf, parameters, cv=5)
clf_gs.fit(X_train_ros, y_train_ros)


print("Best Parameters: ", clf_gs.best_params_)


y_pred_proba = clf_gs.predict_proba(X_test_poly)
y_pred = (y_pred_proba[:,1] >= 0.4).astype(bool)

# Get Final Result
print("Classification Report: \n", classification_report(y_test, y_pred))
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

