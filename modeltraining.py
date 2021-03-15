import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sparse

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
# import module to calculate model perfomance metrics
from sklearn import metrics


data = pd.read_csv("data.csv")

X = data[['square_feet']]
Y = data[['price']]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# Linear Regression Model
# Create linear regression object
linreg = LinearRegression()

print(X_test)
print(y_test)

# fit the model to the training data (learn the coefficients)
linreg.fit(X_train, y_train)

# make predictions on the testing set
y_pred = linreg.predict(X_test)

print (X_test, y_test)



############################################################## Plot data

plt.scatter(X_train, y_train, color='black')
plt.title('Test Data')
plt.xlabel('square_feet')
plt.ylabel('price')
plt.xticks(())
plt.yticks(())

# Plot outputs
plt.plot(X_test, linreg.predict(X_test), color='red',linewidth=3)
plt.show()

