from HLsearch import *
from scipy.integrate import solve_ivp
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

import itertools

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81

def doublePendulum(t, y):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return z1, z1dot, z2, z2dot

t = np.arange(0,10,0.01)
y0=np.array([3*np.pi/7, 0, 3*np.pi/4, 0])
X,Xdot = generate_data(doublePendulum,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
# print(data_description_sym)
# print([data_description[i] for i in range(round(len(data_description)/2))])

expr_new0 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2))],use_sine=True)
expr_new1 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2),len(data_description))],use_sine=True)

print(expr_new0[1:])
print(expr_new1[1:])
expr_new = expr_new0[1:]+expr_new1[1:]
expr = buildFunctionExpressions(4,len(expr_new),expr_new)

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