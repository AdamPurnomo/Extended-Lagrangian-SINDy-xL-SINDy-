from HLsearch import *
from scipy.integrate import solve_ivp
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

import itertools
import scipy.io as sio
data = sio.loadmat('./Data/springmass2_k_is_25_and_9.mat')

X = np.real(data['y'])
Xdot = np.real(data['dy'])
data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)

expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=False)
print(len(expr),' terms are: ',expr)

Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)


u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]

print(xi)

# print('{} coefficients ='.format(xi.shape[0]),xi)

def countNumberOfElementsLargerThanThreshold(x,threshold = 1e-8):
    count = 0
    for i in range(len(x)):
        if abs(x[i]) > threshold:
            count = count +1
    return count

print(countNumberOfElementsLargerThanThreshold(xi))

print('Now drop off small coefficients')
Hamiltonian = generateExpression(xi,expr)

print('H = ',Hamiltonian)