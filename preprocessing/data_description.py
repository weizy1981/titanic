from preprocessing.data_init import get_data
from matplotlib import pyplot as plt

filename = '../data/titanic_train.csv'

if __name__ == '__main__':

    dataset, passengerId = get_data(filename=filename)

    print(dataset.head(5))
    #print(dataset.describe())

    # 年龄获救比例
    age_hist = dataset[['Age', 'Survived']]

    var = age_hist.groupby(['Age', 'Survived']).size()
    print(var)
    var.unstack().plot(kind='bar', stacked=False, color=['red', 'blue'])
    plt.title('Age Distrubution')
    plt.show()

    # 性别获救比例
    sex_hist = dataset[['Sex', 'Survived']]

    var = sex_hist.groupby(['Sex', 'Survived']).size()
    print(var)
    var.unstack().plot(kind='bar', stacked=False, color=['red', 'blue'])
    plt.title('Sex Distrubution')
    plt.show()

    # 贵族获救比例
    sex_hist = dataset[['Title', 'Survived']]

    var = sex_hist.groupby(['Title', 'Survived']).size()
    print(var)
    var.unstack().plot(kind='bar', stacked=False, color=['red', 'blue'])
    plt.title('Nobility Distrubution')
    plt.show()

    # 三等舱的获救比例
    sex_hist = dataset[['Pclass', 'Survived']]

    var = sex_hist.groupby(['Pclass', 'Survived']).size()
    print(var)
    var.unstack().plot(kind='bar', stacked=False, color=['red', 'blue'])
    plt.title('Pclass Distrubution')
    plt.show()

