import numpy as np
from sklearn.metrics import r2_score
from config import config


def sigmoid(theta0,theta,X):
    
    return 1/(1 + np.exp(-(theta0 + np.matmul(X,theta))))


def nll(theta0,theta,X,y):

    p_class_eq_pos_on_X = sigmoid(theta0,theta,X)
    p_class_eq_neg_on_X = (1 - p_class_eq_pos_on_X)

    
    log_p_class_eq_pos_on_X = np.log(p_class_eq_pos_on_X)
    log_p_class_eq_neg_on_X = np.log(p_class_eq_neg_on_X)

    one_minus_y = (1 - y)

    first_term = np.matmul(y.T,log_p_class_eq_pos_on_X)
    second_term = np.matmul(one_minus_y.T,log_p_class_eq_neg_on_X)

    return -1/(y.shape[0])*(first_term + second_term)




def del_by_del_theta(theta0, theta, X, y ):
    
    p_class_eq_pos_on_X = sigmoid(theta0, theta, X)
    del_by_del_theta0 = -np.mean(y - p_class_eq_pos_on_X)
    del_by_del_theta = (-1/y.shape[0])*(np.matmul((y-p_class_eq_pos_on_X).T,X)).T
    
    return [del_by_del_theta0, del_by_del_theta]




def training(epsilon, X_train, y_train, tol):
    
    epoch_counter = 0
    theta0_initial = 0
    theta_initial = np.zeros((X_train.shape[1],1))
    
    while True:
    
        del_by_del_theta_initial = del_by_del_theta(theta0_initial, theta_initial, X_train, y_train)
    
        theta0_final = theta0_initial- (epsilon*del_by_del_theta_initial[0])
        theta_final = theta_initial - (epsilon*del_by_del_theta_initial[1])
    
        nll_initial = nll(theta0_initial, theta_initial, X_train, y_train)[0]
        nll_final = nll(theta0_final, theta_final,  X_train, y_train)[0]
    
        if abs(nll_initial - nll_final) < tol:
          break
    
        theta0_initial = theta0_final
        theta_initial = theta_final
    
        print("Epoch #{}, Binary Cross entropy loss = {}".format(epoch_counter,nll_initial))
    
        epoch_counter +=1
        
        
    return [theta0_final,theta_final]