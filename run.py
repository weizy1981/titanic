from preprocessing.data_init import get_data, splitx_y
from keras.utils.np_utils import to_categorical
from preprocessing.one_hot_encode import one_hot_encodeX
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

init = 'normal'
lr = 0.1
decay = 0.001
momentum = 0.9
batch_size = 5
epochs = 200

def build_model():
    model = Sequential()
    model.add(Dense(units=45, activation='relu', input_dim=45, kernel_initializer=init))
    model.add(Dense(units=90, activation='relu', kernel_initializer=init))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init))

    optimizer = SGD(lr=lr, decay=decay, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    training_dataset, passengerId = get_data(filename='data/titanic_train.csv')

    x, y = splitx_y(training_dataset)

    # one-hot
    x = one_hot_encodeX(x)
    # y = to_categorical(y)

    model = build_model()
    model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)


    # 预测结果
    predict_X, predict_passengerId = get_data(filename='data/test.csv')
    predict_X = one_hot_encodeX(predict_X)
    predict_y = model.predict_classes(predict_X)
    #predict_y = model.predict_classes(scaler.transform(predict_X))

    # 生成预测文件
    with open('data/result.csv', 'w') as file:
        lines = []
        lines.append('PassengerId,Survived\n')
        for passengerId, predict in zip(predict_passengerId.values, predict_y.astype(int)):
            predict = str(predict).replace('[', '').replace(']', '')
            s = (str(passengerId) + ',' + str(predict) + '\n')
            lines.append(s)

        file.writelines(lines)