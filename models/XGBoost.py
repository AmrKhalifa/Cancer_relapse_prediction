import xgboost as xgb
import data_reader
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

x, y = data_reader.read_data()

x= x.T

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0


x_train = x[:-50]
x_test = x[-50:]

y_train = y[:-50]
y_test = y[-50:]


print(x_train.shape)
print(y_train.shape)


dtrain = xgb.DMatrix(x_train, label=y_train)

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'reg:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

num_round = 10
#bst = xgb.train(param, dtrain, num_round)

bst = XGBClassifier()
bst.fit(x_train, y_train)

y_pred = bst.predict(x_test)

#predictions = [round(value) for value in y_train]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

