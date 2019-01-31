import data_reader
from sklearn.metrics import*
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.feature_selection import VarianceThreshold

x, y = data_reader.read_data()

x = (x-x.mean(axis = 0))/x.std(axis=0)
y = y.flatten()
x = x.T

print (x.shape)
sel = VarianceThreshold(threshold=(.05 * (1 - .05)))
x = sel.fit_transform(x)

print(x.shape)

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x, y = data_reader.read_data()

x = (x-x.mean(axis = 0))/x.std(axis=0)
y = y.flatten()
x = x.T

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)


# fit a model
model = LogisticRegression()
model.fit(x_train, y_train)
# predict probabilities
probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate roc auc
auc = roc_auc_score(y_test, probs)
print(auc)


fpr, tpr, thresholds = roc_curve(y_test, probs)
print(y_test)
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr, tpr)
pyplot.show()

