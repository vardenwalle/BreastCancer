import pandas as pd
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
# Logistic regression
from sklearn.linear_model import LogisticRegression
# Загрузка файла и вывод на экран
dataset = pd.read_csv('data.csv')
pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
print(dataset.head(5))
# инициализация Х и У
X = dataset.drop(columns=(['diagnosis','id','Unnamed: 32']))
Y = dataset['diagnosis']
print(Y.head(5))
# Определение опухолей
# 0 - злокачественная
# 1 - доброкачественная
for i in range(len(Y)):
    if Y[i] == 'M':
        Y[i] = float(0)
    else:
        Y[i] = float(1)
print(Y.head(5))
# приведение к типу данных с плавающей точкой
X = X.to_numpy(dtype='float32')
Y = Y.to_numpy(dtype='float32')
# Метод К-ближайших соседей
# Создание генератора разбиений
kfold = KFold(n_splits=5, shuffle=True, random_state=13)
scores = []
for i in range(1,50):
    clf = KNeighborsClassifier(n_neighbors=i)
    score = cross_val_score(clf, X, Y, scoring='accuracy', cv=kfold).mean()
    scores.append(score)

plt.plot(range(1,50), scores)
print('X=',X)
print('Y=',Y)
# Построение графика
plt.xlabel('Number of neighbours')
plt.ylabel('Accuracy')
plt.show()