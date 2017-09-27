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

def get_data(filename):

    data = read_csv(filename)
    passengerId = data['PassengerId']
    del(data['PassengerId'])
    del(data['Name'])
    data['Sex'] = data['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
    #print(data.groupby('Embarked').size())
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].replace(to_replace=['C', 'Q', 'S'], value=[0, 1, 2])

    #print(data.isnull())
    # Age and Cabin has null value
    age_average = round(data['Age'].sum() / data['Age'].size)
    data['Age'] = data['Age'].fillna(age_average)

    # Ticket keep after space
    data['Ticket'] = data['Ticket'].apply(lambda x: x.split(' ')[len(x.split(' ')) - 1])
    data['Ticket'] = data['Ticket'].apply(lambda x: '0' if x=='LINE' else x)

    data['Cabin'] = data['Cabin'].fillna(0)
    data['Cabin'] = data['Cabin'].apply(exhange)

    fare_average = round(data['Fare'].sum() / data['Fare'].size)
    data['Fare'] = data['Fare'].fillna(fare_average)

    #print(data.isnull())

    # 根据特征工程删除如下项目
    #print(data.keys())
    #del (data['Embarked'])
    #del (data['Cabin'])
    #del (data['Parch'])

    return data, passengerId
