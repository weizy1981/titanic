from model.data_prepare import get_data
from sklearn.preprocessing import StandardScaler, Normalizer
from model.models import create_MLP_
import numpy as np

np.random.seed(seed=7)

training_dataset, passengerId = get_data(filename='data/titanic_train.csv')

array = training_dataset.values
X = array[:, 1:].astype(float)
y = array[:, 0].astype(int)

scaler = StandardScaler().fit(X)
X = scaler.transform(X)

model = create_MLP_()
model.fit(X, y, epochs=100, batch_size=5, verbose=1)

result = model.evaluate(X, y)
print('\n')
print('%s: %.2f%%' % ('acc', result[1] * 100))

# 预测结果
predict_X, predict_passengerId = get_data(filename='data/test.csv')
#predict_y = model.predict_classes(StandardScaler().fit_transform(predict_X))
predict_y = model.predict_classes(scaler.transform(predict_X))

# 生成预测文件
with open('data/result.csv', 'w') as file:
    lines = []
    lines.append('PassengerId,Survived\n')
    for passengerId, predict in zip(predict_passengerId.values, predict_y.astype(int)):
        predict = str(predict).replace('[', '').replace(']', '')
        s = (str(passengerId) + ',' + str(predict) + '\n')
        lines.append(s)

    file.writelines(lines)
