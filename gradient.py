# module:   gradient_descent.py
# author:   Jin Yeom
# since:    01/12/17

import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(input_mat, output_vec, learning_rate=0.1, 
                    threshold=0.0001, plot=False, limit=100000):
   
    input_shape = input_mat.shape
    output_shape = output_vec.shape
    assert input_shape[1] == output_shape[0], 'invalid shape of input/output data'
    
    n, m = input_shape[0], input_shape[1]
    h_func = lambda theta, x: np.dot(theta.T, x)
   
    losses = [] 
    theta_vec = np.random.rand(n, 1)
    
    for i in range(limit):
        err = h_func(theta_vec, input_mat) - output_vec
        loss = np.sum(err ** 2, axis=1) / (2 * m)
        deriv_term = np.sum(input_mat * err, axis=1) / m
        gradient = (learning_rate * deriv_term).reshape(n, 1)
        theta_vec -= gradient

        losses.append(loss)
        if np.sum(loss)/n < threshold:
            break

    if plot:
        plt.plot(losses, 'rs')
        plt.show()

    return theta_vec.T
