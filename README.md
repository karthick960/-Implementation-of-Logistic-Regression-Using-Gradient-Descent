# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GANESH S
RegisterNumber: 212222040042
```
```PYTHON
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:


### Array Value of x
![Screenshot 2024-05-07 144152](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/41ffe09a-f978-4950-a0fe-4d6f39c4e4f2)


### Array Value of y
![Screenshot 2024-05-07 144234](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/458f7ffa-4a55-4b34-8bd7-bf0174800e2a)


### Exam 1 - score graph
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/65b32461-92ec-491d-863a-452186cc8267)


### Sigmoid function graph
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/6e2f85d3-b687-4372-bbfd-a568706642b3)


### X_train_grad value
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/97796422-5b58-49bc-92c1-dcfdd8875a63)



### Y_train_grad value
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/64984c13-a857-45c2-8e4e-556c6ce5b701)


### Print res.x

![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/beec9f2f-c320-49cb-9e03-f4cc18a8eae5)




### Decision boundary - graph for exam score
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/36b176d0-a9b0-4c14-8ab6-9aaf86318f3d)




### Proability value

![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/08efa57a-2e86-4fe2-bd63-6660258b1aa0)



### Prediction value of mean
![image](https://github.com/karthick960/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121215938/309f949a-835a-4655-80f9-805d228555ab)

## RESULT
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

