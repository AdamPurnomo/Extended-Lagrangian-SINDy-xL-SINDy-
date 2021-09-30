from HLsearch import *
from scipy.integrate import solve_ivp
from scipy import stats
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

import itertools

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

g = 9.81
m = 1
L = 1

def spherePend(t,y,Moment=0.0):
    theta, theta_t, phi, phi_t = y
    theta_2t, phi_2t = (L**2*m*np.sin(theta)*np.cos(theta)*phi_t**2 + L*g*m*np.sin(theta))/(L**2*m),(-2.0*L**2*m*np.sin(theta)*np.cos(theta)*phi_t*theta_t + Moment)/(L**2*m*np.sin(theta)**2)
    return theta_t,theta_2t,phi_t,phi_2t

t = np.arange(0,2.5,0.01)
y0=np.array([np.pi/2, 0,0, 0.5])
X,Xdot = generate_data(spherePend,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)

expr_new0 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2))],use_sine=True)
expr_new1 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2),len(data_description))],use_sine=False)

print(expr_new0[1:])
print(expr_new1)
expr_new = expr_new0[1:]+expr_new1
expr = buildFunctionExpressions(4,len(expr_new),expr_new)

print(len(expr),' terms are: ',expr)

Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)


u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]

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