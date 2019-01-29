import data_reader
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

x, y = data_reader.read_data()

y = y.flatten()
x = x.T

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

num_round = 10

bst = XGBClassifier(max_depth=10000, learning_rate=0.1, n_estimators=1000, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=4, gamma=0.01, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)

bst.fit(x_train, y_train)

y_pred = bst.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(y_pred)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

