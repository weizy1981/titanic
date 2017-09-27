from model.data_prepare import get_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
from model.models import create_LR, create_CART, create_KNN, create_LDA, create_AB, create_RF, create_SVM, create_GB, create_MLP
from model.validation_fold import validation_model
from model.feature_selection import select_feature
import numpy as np
from model.turn_model import turn_RF, turn_MLP

np.random.seed(seed=7)

#pd.set_option('display.height', 1000)

training_dataset, names = get_data(filename='data/titanic_train.csv')

print(training_dataset.groupby('Survived').size())

array = training_dataset.values
#print(array.shape)
X = array[:, 1:].astype(float)
y = array[:, 0].astype(float)

X = StandardScaler().fit_transform(X)

#model = create_MLP()
#turn_MLP(model, X, y)
#validation_model(model, X, y)

#print(X.shape)

#model = create_LR()
#validation_model(model, X, y)

#model = create_LDA()
#validation_model(model, X, y)

#model = create_KNN()
#validation_model(model, X, y)

#model = create_CART()
#validation_model(model, X, y)

#model = create_GB()
#validation_model(model, X, y)

#model = create_SVM()
#validation_model(model, X, y)

#model = create_AB()
#validation_model(model, X, y)

#model = create_RF()
#validation_model(model, X, y)

# 特征选择
#select_feature(X, y)
# [ 0.06486463  0.25325018  0.17271201  0.04292124  0.02638373  0.19974153 0.20668297  0.01068576  0.02275796]
#print(training_dataset.keys())
#del(training_dataset['Embarked'])
#del(training_dataset['Cabin'])
#del(training_dataset['Parch'])
#del(training_dataset['SibSp'])
#del(training_dataset['SibSp'])
#del(training_dataset['Pclass'])
#print(training_dataset)

#array = training_dataset.values
#y = array[:, 0].astype(float)
#X = array[:, 1:].astype(float)
#X = StandardScaler().fit_transform(X)
# print(X.shape)

model = create_RF()
#turn_RF(model, X, y)
validation_model(model, X, y)

#print(X)