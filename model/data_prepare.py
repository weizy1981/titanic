from pandas import read_csv


def exhange(cabin):
    value = ''
    if cabin != 0:
        for char in cabin:
            if char == ' ':
                char = ''
            if char == 'A':
                char = '1'
            if char == 'B':
                char = '2'
            if char == 'C':
                char = '3'
            if char == 'D':
                char = '4'
            if char == 'E':
                char = '5'
            if char == 'F':
                char = '6'
            if char == 'G':
                char = '7'

            if char == 'T':
                char = '8'
            value = value + char
    else:
        value = cabin
    return value

def name_to_tile(name):
    title = name.split(',')[1].split('.')[0].replace(' ', '')
    titles = ['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'theCountess']
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    index = titles.index(title)

    return values[index]

def get_data(filename):
    data = read_csv(filename)
    passengerId = data['PassengerId']
    del (data['PassengerId'])
    # del(data['Name'])
    data['Name'] = data['Name'].apply(name_to_tile)
    #data['Name'] = data['Name'].replace(to_replace=['Capt', 'Col', 'Don', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev', 'Sir', 'the Countess'], value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

    data['Sex'] = data['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
    #print(data['Name'])
    # print(data.groupby('Embarked').size())
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].replace(to_replace=['C', 'Q', 'S'], value=[0, 1, 2])

    # print(data.isnull())
    # Age and Cabin has null value
    age_average = round(data['Age'].sum() / data['Age'].size)
    data['Age'] = data['Age'].fillna(age_average)

    # Ticket keep after space
    data['Ticket'] = data['Ticket'].apply(lambda x: x.split(' ')[len(x.split(' ')) - 1])
    data['Ticket'] = data['Ticket'].apply(lambda x: '0' if x == 'LINE' else x)

    data['Cabin'] = data['Cabin'].fillna(0)
    data['Cabin'] = data['Cabin'].apply(exhange)
    # data['Cabin'] = data['Cabin'].apply(lambda x: 0 if x==0 else 1)

    # fare_average = round(data['Fare'].sum() / data['Fare'].size)
    data['Fare'] = data['Fare'].fillna(0)

    # print(data.isnull())

    # 根据特征工程删除如下项目
    # print(data.keys())
    # del (data['Embarked'])
    # del (data['Cabin'])
    # del (data['Parch'])
    # del(data['Ticket'])

    return data, passengerId
