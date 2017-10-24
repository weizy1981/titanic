from keras.utils.np_utils import to_categorical
import numpy as np

def one_hot_encodeX(x):
    pclass = to_categorical(x['Pclass'], num_classes=4)
    print(pclass.shape)
    sex = to_categorical(x['Sex'], num_classes=2)
    print(sex.shape)
    sibSp = to_categorical(x['SibSp'], num_classes=9)
    print(sibSp.shape)
    parch = to_categorical(x['Parch'], num_classes=10)
    print(parch.shape)
    cabin = to_categorical(x['Cabin'], num_classes=2)
    print(cabin.shape)
    embarked = to_categorical(x['Embarked'], num_classes=3)
    print(embarked.shape)
    title = to_categorical(x['Title'], num_classes=13)
    print(title.shape)
    age = x['Age'].values.reshape(len(x), 1)
    fare = x['Fare'].values.reshape(len(x), 1)
    value = np.concatenate((pclass, sex, sibSp, parch, cabin, embarked, title, age, fare), axis=1)

    print(value.shape)
    return value

