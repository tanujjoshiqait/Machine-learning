import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

costStore=np.zeros(2000);
# test data prepare
data=pd.read_csv('train.csv')
X_train=np.array([np.ones(data.shape[0]),data['x']])
X_train=np.transpose(X_train);

Y_train=np.array(data['y'])
theta=np.zeros(data.shape[1])

# hypothesis
def hypothesis(theta, x):
    return np.dot(x, theta);


# Cost Function
def costFunction(theta, X, Y):
    m=Y.size;
    temp=np.square(hypothesis(theta, X)-Y);
    cost=1/(2*m)*np.nansum(temp);
    return cost;
       
def gradientDescent(X,Y, theta, iterations, alpha):
    m=Y.size;
    for count in range(iterations):
        theta=theta- alpha*(1/m)* np.transpose(X).dot(hypothesis(theta, X)-Y);
#        print("Cost : ",costFunction(theta, X, Y));        
        costStore[count]=costFunction(theta, X, Y);

    print("Training Cost : ", costFunction(theta, X,Y));
    print("Theta Obtained : ",theta);        
    return theta;
        
        
theta=gradientDescent( X_train, Y_train,theta, 2000, 0.0001);
testdata=pd.read_csv('test.csv');
X_test=np.array([np.ones(testdata.shape[0]),testdata['x']]);
X_test=np.transpose(X_test);
Y_test=np.array(testdata['y']);

print("Test Cost: ",costFunction(theta, X_test,Y_test));

print(costStore)
plt.plot( range(2000), costStore ,linewidth=2.0);
#plt.show();
print(theta)
print(hypothesis(theta, X_test))


    


