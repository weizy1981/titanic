from model2.data_prepare import get_data
from sklearn.preprocessing import StandardScaler, Normalizer
from model2.models import create_MLP_, create_MLP
import numpy as np
from model2.validation_fold import validation_model

np.random.seed(seed=7)

training_dataset, passengerId = get_data(filename='data/titanic_train.csv')

array = training_dataset.values
X = array[:, 1:].astype(float)
y = array[:, 0].astype(int)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
#print(X)

batch_size = 5
epochs = 400

val_model = create_MLP(epochs=epochs, batch_size=batch_size)
validation_model(val_model, X, y)

model = create_MLP_()
#Model_base
#model.fit(X, y, epochs=100, batch_size=5, verbose=1)
#Model_2
#model.fit(X, y, epochs=300, batch_size=20, verbose=1)
model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

#result = model.evaluate(X, y)
#print('\n')
#print('%s: %.2f%%' % ('acc', result[1] * 100))

# 预测结果
predict_X, predict_passengerId = get_data(filename='data/test.csv')
predict_y = model.predict_classes(StandardScaler().fit_transform(predict_X))
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

