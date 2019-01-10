import numpy as np
import itertools
joint_prob = np.array([[0.3 , 0.3 , 0] ,[0.1 , 0.2 , 0.1]])
X = [1, 2]
Y = [-1 , 0, 5]

marginal_X = joint_prob.sum(axis = 1)
marginal_Y = joint_prob.sum(axis = 0) 

#Expectations
expectation_X  = sum([X[i]*marginal_X[i] for i in range(len(marginal_X))])
expectation_Y  = sum([Y[i]*marginal_Y[i] for i in range(len(marginal_Y))])
prob_y_given_x = joint_prob / marginal_X.reshape(-1,1)
prob_x_given_y = joint_prob / marginal_Y

expectation_y_given_x = sum([X[i]*Y[j]*prob_y_given_x[i][j] for i,j in itertools.product(range(len(X)),range(len(Y)))])
expectation_x_given_y = sum([X[i]*Y[j]*prob_x_given_y[i][j] for i,j in itertools.product(range(len(X)),range(len(Y)))])
cov = sum([(X[i]-expectation_X)*(Y[j]-expectation_Y)*joint_prob[i][j] for i,j in itertools.product(range(len(X)),range(len(Y)))])

#%%
#Entropy
entropy_x_y = sum([-np.log(j)*j if j!=0 else 0 for i in joint_prob for j in i])
entropy_x = sum([-np.log(j)*j for j in marginal_X])
entropy_y = sum([-np.log(j)*j for j in marginal_Y])
entropy_y_given_x = sum([-np.log(prob_y_given_x[i][j])* joint_prob[i][j] if prob_y_given_x[i][j]!=0 else 0 for i in range(len(prob_y_given_x)) for j in range(len(prob_y_given_x[i]))])
entropy_x_given_y = sum([-np.log(prob_x_given_y[i][j])* joint_prob[i][j] if prob_x_given_y[i][j]!=0 else 0 for i in range(len(prob_x_given_y)) for j in range(len(prob_x_given_y[i]))])

#%%
#Mutual Information
MI_x_y = entropy_x - entropy_x_given_y

#Verification
if entropy_x_y <= entropy_x + entropy_y:
    print('H[x,y] < H[x] + H[y]' )
    print('H[x,y]: ', entropy_x_y)
    print('H[x]: ',entropy_x)
    print('H[y]: ',entropy_y)
    
print('H[y]:', entropy_y) 
print('I[X,Y] + H[Y|X]: ',MI_x_y + entropy_y_given_x)     
