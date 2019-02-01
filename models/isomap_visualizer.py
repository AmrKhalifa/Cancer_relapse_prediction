from sklearn.manifold import Isomap
import data_reader
import matplotlib.pyplot as plt

x, y = data_reader.read_data()

y = y.flatten()
x = x.T

print(x.mean())
print(x.std())
# for i in range (y.shape[0]):
#     if y[i] == -1:
#         y[i] =0
# y = y[::-1]

embedding = Isomap(n_components=2,n_neighbors=1)

x_transformed = embedding.fit_transform(x)

print(x_transformed.shape)

#plt.scatter(x_transformed[:,0],x_transformed[:,1],c=y)
#plt.show()

plt.hist(x.T)
plt.show()
