
def DecisionTree(train_inputs, train_classes):
    from sklearn import tree

    model = tree.DecisionTreeClassifier()
    model = model.fit(train_inputs, train_classes)

    return model

def KNN(train_inputs, train_classes, k):
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model = model.fit(train_inputs, train_classes)

    return model

def GNB(train_inputs, train_classes):
    from sklearn.naive_bayes import GaussianNB

    model = GaussianNB()
    model = model.fit(train_inputs, train_classes)

    return model

def NN(train_inputs, train_classes, epoch, learning_rate):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam

    model = Sequential()

    model.add(Dense(8, activation="relu", input_dim=20))
    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    opt = Adam(learning_rate=learning_rate)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    history = model.fit(train_inputs, train_classes, validation_split=0.1, epochs=epoch, batch_size=32)

    return model, history
