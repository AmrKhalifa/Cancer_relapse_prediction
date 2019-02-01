import data_reader
from sklearn.metrics import*
from sklearn.model_selection import train_test_split
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.model_selection import cross_val_score as cvs, RandomizedSearchCV


x, y = data_reader.read_data()

x = (x-x.mean(axis = 0))/x.std(axis=0)
y = y.flatten()
x = x.T

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


bdt = GradientBoostingClassifier(n_estimators=100)


bdt.fit(x_train, y_train)

y_pred = bdt.predict(x_test)


score = cvs(bdt,x_train,y_train,cv = 5,scoring='accuracy')
score = score.mean()
auc1 =  cvs(bdt,x_train,y_train,cv = 5,scoring='roc_auc')
print('score = {0}'.format(score))
print('auc = {0}'.format(auc1.mean()))

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)


print(y_pred)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

print("AUC %.4f%%" %(auc*100.0))