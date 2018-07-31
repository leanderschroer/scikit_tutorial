import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
iris = sns.load_dataset('iris')
#sns.pairplot(iris, hue='species', height=1.5)
#plt.show()

X_iris = iris.drop('species', axis = 1)
y_iris = iris['species']

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
#plt.scatter(x, y)

model = LinearRegression(fit_intercept=True)

X = x[:,np.newaxis]
model.fit(X, y)
print (model.coef_)
print (model.intercept_)
xfit = np.linspace(-1,11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
#plt.plot(xfit,yfit)


Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
print (train_test_split(X_iris, y_iris,random_state=1))
from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data

from sklearn.metrics import accuracy_score
print (accuracy_score(ytest, y_model))

#plt.plot()
