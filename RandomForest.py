import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
import matplotlib.pyplot as plt



#Load data
train_path = r"data\train.npy"
labels_path = r"data\labels.npy"
train = np.load(train_path, allow_pickle = True)[0:800]
labels = np.load(labels_path, allow_pickle = True)[0:800]
#train = (train > 0).astype(float)

test = np.load(train_path, allow_pickle = True)[800:1000]
test_labels = np.load(labels_path, allow_pickle = True)[800:1000]
#test = (test > 0).astype(float)

def classification_error(num_trees):
    tree = RandomForestClassifier(n_estimators = num_trees)
    tree.fit(train, labels)
    pred = tree.predict(test)
    acc = np.mean(np.equal(pred, test_labels).astype(int))
    return 1 - acc

rf = RandomForestClassifier()
min_num = 1
min_err = 1
err_list = []
for i in range(1, 300):
    err = classification_error(i)
    print(i, " : ", err)
    err_list.append(err)
    if err < min_err:
        min_num = i
print(min_num)
plt.plot(err_list)
plt.title("Misclassification error vs. Number of trees")
plt.show()




print("Error: ", classification_error(rf))
