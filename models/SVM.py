import numpy as np
import data_reader
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score as cvs

xtrain, ytrain = data_reader.read_data()
xtrain = xtrain.T
ytrain = ytrain.reshape(-1,)
N,D = xtrain.shape

#hyperparameters
C = 3
kernel  = 'rbf'  # linear / sigmoid / poly
gamma = 1


classifier = SVC(C=C, kernel=kernel)
classifier = classifier.fit(xtrain, ytrain)
ypredict = classifier.predict(xtrain)
train_error = (ypredict != ytrain).mean()
score = cvs(classifier,xtrain,ytrain,cv=5,scoring='accuracy')
print ('score = {0}'.format(score.mean()))
print ('train_error = {0}'.format(train_error))

C_range = np.logspace(-3, 3, 100)
C_scores = np.zeros(len(C_range))

for i,C in enumerate(C_range):
    classifier = SVC(C=C, kernel=kernel)
    scores = cvs(classifier, xtrain,ytrain,cv=5,scoring='accuracy')
    C_scores[i] = scores.mean()
C_best = C_range[np.argmax(C_scores)]
print('best C is {0}'.format(C_best))
classifier = SVC(C=C_best,kernel=kernel)
score = cvs(classifier,xtrain,ytrain,cv=5,scoring='accuracy')
print('best score is '.format(score.mean()))