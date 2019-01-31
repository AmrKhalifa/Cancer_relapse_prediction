from sklearn.ensemble import RandomForestClassifier
import numpy as np
import data_reader
from sklearn.model_selection import cross_val_score as cvs, RandomizedSearchCV
from sklearn.metrics import roc_auc_score


xtrain, ytrain = data_reader.read_data()
xtrain = xtrain.T
ytrain = ytrain.reshape(-1,)
N, D = xtrain.shape

clf = RandomForestClassifier(n_estimators=600,random_state=0)
clf.fit(xtrain,ytrain)

ypredict = clf.predict(xtrain)
train_error = (ypredict != ytrain).mean()
score = cvs(clf, xtrain, ytrain, cv=5, scoring='accuracy')
print ('score = {0}'.format(score.mean()))
print ('train_error = {0}'.format(train_error))


#hyperparameters tunining
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 300, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(), param_distributions = random_grid, n_iter = 5, cv = 5, verbose=2, random_state=30, n_jobs = -1)

rf_random.fit(xtrain,ytrain)
ypredict = rf_random.predict(xtrain)
train_error = (ypredict != ytrain).mean()
roc_auc_score = cvs(rf_random, xtrain, ytrain, cv=5, scoring='roc_auc')
score  = cvs(rf_random, xtrain, ytrain, cv=5, scoring='accuracy')
print ('score = {0}'.format(score.mean()))
print ('roc_auc_score = {0}'.format(roc_auc_score.mean()))

# rf_random.fit(xtrain,ytrain)
# print(rf_random.best_params_)

#rf_best = RandomForestClassifier (bootstrap =  False, min_samples_leaf =  1, n_estimators =  488, max_features =  'sqrt', min_samples_split= 10, max_depth = 90)

#rf_best.fit(xtrain, ytrain)

#ypredict = rf_best.predict(xtrain)
#train_error = (ypredict != ytrain).mean()
#score = cvs(rf_best, xtrain, ytrain, cv=5, scoring='roc_auc')
#roc_auc_score = cvs(rf_best, xtrain, ytrain, cv=5, scoring='accuracy')
#print ('score = {0}'.format(score.mean()))
#print ('roc_auc_score = {0}'.format(roc_auc_score.mean()))


# probs = rf_best.predict_proba(xtrain)
# probs = probs[:, 1]
# auc = roc_auc_score(ytrain, probs)

# print(auc)
#6521416153