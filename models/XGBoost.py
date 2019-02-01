import data_reader
from sklearn.metrics import*
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


x, y = data_reader.read_data()

x = (x-x.mean(axis = 0))/x.std(axis=0)
y = y.flatten()
x = x.T

for i in range (y.shape[0]):
    if y[i] == -1:
        y[i] =0

embedding = Isomap(n_components=120,n_neighbors=100)
#x_transformed,error= locally_linear_embedding(x,n_components=100,n_neighbors=5)
# #embedding = SpectralEmbedding(n_components=100)
# #print(error)
x_transformed = embedding.fit_transform(x)



x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3)


bst = XGBClassifier(max_depth=10000, learning_rate=0.1, n_estimators=500, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=4, gamma=0.01, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None)

bst.fit(x_train, y_train)

y_pred = bst.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(y_pred)
print("Accuracy: %.4f%%" % (accuracy * 100.0))

print("AUC %.4f%%" %(auc*100.0))

