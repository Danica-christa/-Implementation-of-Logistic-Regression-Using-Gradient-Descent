# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.

2. Print the placement data and salary data.

3. Find the null and duplicate values.

4. Using logistic regression find the predicted values of accuracy , confusion matrices.



## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Danica Christa
RegisterNumber: 212223240022
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
*/
```

## Output:

Dataset:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/e8eaca43-7f14-4e1b-ae00-19a4796539d6)


Dataset.dtypes:

![alt text](image-1.png)

Status data

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/6816ef11-621b-45f8-92b2-0e465e278671)


Labeled_dataset:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/5bfb5018-9c42-4e48-82a4-79c507650701)


Dependent variable Y:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/64e2e384-ba5b-4781-849d-5b40d0dc4046)


Accuracy:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/581011bd-da64-4c77-818b-b2f33753ac38)


y_pred:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/e1a0722b-7380-496e-b010-2d007f747a45)


Y:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/2529b38d-8b58-4d53-9e85-d4be59973ebb)


y_pred:

![image](https://github.com/Danica-christa/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/151514009/f6f2ebb2-7a62-4647-90d6-048581160d30)

 


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

