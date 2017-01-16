import numpy as np
from gadient import gradient_descent

n = 8
m = 500

target = np.random.randint(1, 10, size=(n, 1)).T
fn = lambda x: np.dot(target, x)

input_mat = np.random.rand(n, m)
output_vec = np.array([fn(vec) for vec in input_mat.T]).reshape(m)

param = gradient_descent(input_mat, output_vec, threshold=0.001, plot=True) 

print target
print param
