import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

seed = 42
data = pd.read_csv("./Dataset.csv")

x = data.drop(columns=["Class"])  # Replace "target_column" with the name of your target column
y = data["Class"]
# print(x,y)

y = data.iloc[:,0:1]
x = data.iloc[:,1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# define scoring method
scoring = 'accuracy'
 
# Define models to train
names = ["Nearest Neighbors"
         , "Gaussian Process"
         ,"Decision Tree"
         , "Random Forest"
         , "Neural Net"
         , "AdaBoost"
         ,"Naive Bayes"
         , "SVM Linear"
         , "SVM RBF"
         , "SVM Sigmoid"]
 
 
 
classifiers = [
    KNeighborsClassifier(n_neighbors = 3)
#     ,GaussianProcessClassifier(1.0 * RBF(1.0))
    ,DecisionTreeClassifier(max_depth=5)
    ,RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    ,MLPClassifier(alpha=1)
#     ,AdaBoostClassifier()
    ,GaussianNB()
    ,SVC(kernel = 'linear')
    ,SVC(kernel = 'rbf')
    ,SVC(kernel = 'sigmoid')
]
 
 
 
models = zip(names, classifiers)
 
 
# evaluate each model in turn
results = []
names = []
 
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state = seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print('Test-- ',name,': ',accuracy_score(y_test, predictions))
    print()
    print(classification_report(y_test, predictions))