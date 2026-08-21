# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize the weights and bias with small/random values.
2. Calculate the predicted probability using the **sigmoid function**.
3. Compute the error and update weights and bias using **gradient descent**.
4. Repeat the process for the given number of iterations until the model converges.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MYVIZHI S
RegisterNumber:  212224040209
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
 
# Dataset
X = np.array([[30], [35], [40], [45], [50], [55], [60], [65]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])   # 0 = Fail, 1 = Pass
 
# Train model
model = LogisticRegression()
model.fit(X, y)
 
# Predict for a new student
marks = [[48]]
result = model.predict(marks)
prob = model.predict_proba(marks)
 
print("Prediction (0=Fail, 1=Pass):", result)
print("Probability of Fail :", prob[0][0])
print("Probability of Pass :", prob[0][1])
 
# Model parameters
w = model.coef_[0][0]
b = model.intercept_[0]
 
# Generate values for plotting
X_test = np.linspace(25, 70, 200).reshape(-1, 1)
 
# Pass sigmoid
pass_prob = 1 / (1 + np.exp(-(w * X_test + b)))
 
# Fail sigmoid
fail_prob = 1 - pass_prob
 
# Plot original data
plt.scatter(X[y==0], y[y==0], color='red', marker='x', s=100, label='Fail Data')
plt.scatter(X[y==1], y[y==1], color='green', marker='o', s=100, label='Pass Data')
 
# Plot Pass sigmoid
plt.plot(X_test, pass_prob, color='blue', linewidth=2,
         label='Pass Sigmoid')
 
# Plot Fail sigmoid
plt.plot(X_test, fail_prob, color='orange', linewidth=2,
label='Fail Sigmoid')
# Mark predicted point
plt.scatter(marks, prob[0][1], color='black', s=120,
label='Prediction (48 Marks)')
plt.xlabel("Marks")
plt.ylabel("Probability")
plt.title("Logistic Regression - Pass & Fail Sigmoid Curves")
plt.grid(True)
plt.legend()
plt.show()
```

## Output:
<img width="807" height="786" alt="image" src="https://github.com/user-attachments/assets/82aa41bd-5a7b-4b23-bcc4-665c227d0e20" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

