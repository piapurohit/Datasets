import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings(action='ignore')
iris=pd.read_csv("C:/Users/Pia Purohit/PycharmProjects/Datasets/iris.csv")
print(iris)
print(iris.shape)
print(iris.describe())
print(iris.isna().sum())
print(iris.describe())
print(iris.head())
print(iris.head(150))
print(iris.tail(100))
n = len(iris[iris['variety'] == 'Versicolor'])
print("No of Versicolor in Dataset:",n)

n1 = len(iris[iris['variety'] == 'Virginica'])
print("No of Virginica in Dataset:",n1)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()

import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([iris['sepal.length']])
plt.figure(2)
plt.boxplot([iris['sepal.width']])
plt.show()

iris.hist()
plt.show()

iris.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)
plt.show()

X = iris['sepal.length'].values.reshape(-1,1)
print(X)
Y = iris['sepal.width'].values.reshape(-1,1)
print(X)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)

train_X = train[['sepal.length', 'sepal.width', 'petal.length']]
train_y = train.variety
test_X = test[['sepal.length', 'sepal.width', 'petal.length']]
test_y = test.variety
train_X.head()
test_y.head()

#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))

#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))

#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))

#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))

#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))

#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))

results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)