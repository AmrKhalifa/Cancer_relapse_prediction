import numpy as np
import data_reader
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as cvs, RandomizedSearchCV

xtrain, ytrain = data_reader.read_data()
xtrain = xtrain.T
ytrain = ytrain.reshape(-1,)
N, D = xtrain.shape

#hyperparameters
C = 3
kernel  = 'rbf'  # linear / sigmoid / poly
gamma = 1


#hyperparameters tunining
# kernels = ['rbf','linear','sigmoid','poly']
# C_range  = np.logspace(-3, 3, 100)
# gamma_range = np.logspace(-5, 0, 100)
#
# random_grid = {'kernel': kernels,
#                'C': C_range,
#                'gamma': gamma_range
#               }


# classifier = RandomizedSearchCV(estimator = SVC(), param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=40, n_jobs = 4)
#
# classifier = classifier.fit(xtrain, ytrain)
# ypredict = classifier.predict(xtrain)
# train_error = (ypredict != ytrain).mean()
# score = cvs(classifier, xtrain, ytrain, cv=5, scoring='accuracy')
# roc_auc_score = cvs(classifier, xtrain, ytrain, cv=5, scoring='roc_auc')
# print ('score = {0}'.format(score.mean()))
# print ('roc_auc_score = {0}'.format(roc_auc_score.mean()))
#
# print (classifier.best_params_)

C_range = np.logspace(-3, 3, 100)
C_scores = np.zeros(len(C_range))

for i,C in enumerate(C_range):
    classifier = SVC(C=C, kernel=kernel)
    scores = cvs(classifier, xtrain, ytrain, cv=5, scoring='accuracy')
    C_scores[i] = scores.mean()
C_best = C_range[np.argmax(C_scores)]
print('best C is {0}'.format(C_best))

C_best = 2.8
classifier = SVC(C=C_best, kernel=kernel)
classifier = classifier.fit(xtrain, ytrain)
ypredict = classifier.predict(xtrain)

score = cvs(classifier, xtrain, ytrain, cv=5, scoring='accuracy')
roc_auc_score = cvs(classifier, xtrain, ytrain, cv=5, scoring='roc_auc')
print('best score is {0}'.format(score.mean()))
print('best roc_auc_score is {0}'.format(roc_auc_score.mean()))
