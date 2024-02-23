import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import nn

# methods: DT (Decision Tree), KNN (K Nearest Neighbours), GNB (Gaussian Naive Bayes), NN (Sequential)

method = "KNN"
epoch = 50
learning_rate = 0.001
k = 11

df = pd.read_csv("waterQuality1.csv", na_values=['#NUM!'])
df = df.dropna()

from sklearn.model_selection import train_test_split

(train_set, test_set) = train_test_split(df.values, train_size=0.85, random_state=23215)

train_set = np.array(train_set)
test_set = np.array(test_set)

train_inputs = train_set[:, 0:20]
train_classes = train_set[:, 20]
test_inputs = test_set[:, 0:20]
test_classes = test_set[:, 20]

if method == "DT":
    from sklearn import tree
    model = nn.DecisionTree(train_inputs, train_classes)

    tree.plot_tree(model)
    plt.savefig("decisionTree.png", dpi=1200)
    plt.clf()

elif method == "KNN":
    model = nn.KNN(train_inputs, train_classes, k)

elif method == "GNB":
    model = nn.GNB(train_inputs, train_classes)

elif method == "NN":
    model, history = nn.NN(train_inputs, train_classes, epoch, learning_rate)

if method == "NN":
    from sklearn.metrics import accuracy_score

    y_pred = model.predict(test_inputs)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_pred, test_classes)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('learning curve')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('learning curve.png')
    plt.show()

    plt.clf()

else:
    y_pred = model.predict(test_inputs)
    acc = model.score(test_inputs, test_classes)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_classes, y_pred)

plt.imshow(cm, cmap=plt.cm.Blues)

tick_marks = np.arange(len(set(test_classes)))
plt.xticks(tick_marks, set(test_classes))
plt.yticks(tick_marks, set(test_classes))

for i in range(len(set(test_classes))):
    for j in range(len(set(test_classes))):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.text(1.7, 1.7, f'Accuracy: {acc:.2f}')

plt.xlabel('Predicted label')
plt.ylabel('True label')

if (method == "KNN"):
    plt.title(f'Confusion matrix - {k}' + 'NN')
    plt.colorbar()
    plt.savefig(f'confusion_matrix - {k}' + 'NN' + '.png', dpi=300)
else:
    plt.title('Confusion matrix - ' + method)
    plt.colorbar()
    plt.savefig('confusion_matrix - ' + method + '.png', dpi=300)
