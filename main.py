from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from modelScratch import MLP
import numpy as np
import matplotlib.pyplot as plt

def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

data = load_iris()
X = data.data
# print(X.shape)
# print(X.shape[1])
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = to_one_hot(y_train, 3)
y_test = to_one_hot(y_test, 3)

model = MLP(X.shape[1], 10, 3)
md = model.train(X_train, y_train, 1_000, 0.01)


plt.plot(md["mse"])
plt.title('Mean Squared Error')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()