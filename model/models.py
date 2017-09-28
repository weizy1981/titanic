from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam
from keras import regularizers
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU, ThresholdedReLU
from keras.layers.core import Activation

def create_LR():
    model = LogisticRegression()
    return model

def create_LDA():
    model = LinearDiscriminantAnalysis()
    return model

def create_KNN():
    model = KNeighborsClassifier()
    return model

def create_CART():
    model = DecisionTreeClassifier()
    return model

def create_GB():
    model = GaussianNB()
    return model

def create_SVM():
    model = SVC()
    return model

def create_RF():
    model = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)
    return model

def create_AB():
    model = AdaBoostClassifier()
    return model

def create_MLP_():
    return create_MLP_1()

def create_MLP_1():
    #88.44%
    init = 'normal'
    lr = 0.2
    decay = 0.001
    momentum = 0.9

    model = Sequential()
    model.add(Dense(units=27, activation='relu', input_dim=9, kernel_initializer=init))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))

    optimizer = SGD(lr=lr, decay=decay, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_MLP_base():
    # 84.85%
    init = 'normal'
    lr = 0.1
    decay = 0.01
    momentum = 0.9

    model = Sequential()
    model.add(Dense(units=27, activation='relu', input_dim=9, kernel_initializer=init))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Activation('relu'))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))

    optimizer = SGD(lr=lr, decay=decay, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #model.fit(X, y, epochs=100, batch_size=5, verbose=1)
    return model


def create_MLP():
    model = KerasClassifier(build_fn=create_MLP_, epochs=100, batch_size=5, verbose=0)
    return model