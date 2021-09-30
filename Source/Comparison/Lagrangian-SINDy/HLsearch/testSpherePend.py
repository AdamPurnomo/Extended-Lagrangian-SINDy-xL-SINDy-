from HLsearch import *
from scipy.integrate import solve_ivp
from scipy import stats
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

import itertools

import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

g = 9.81
m = 1
L = 1

def spherePend(t,y,Moment=1.0):
    theta, theta_t, phi, phi_t = y
    theta_2t,phi_2t = (-L**2*m*np.sin(theta)*np.cos(theta)*theta_t**2 + L*g*m*np.sin(theta) + L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*np.sin(phi)*np.cos(theta)*phi_t + L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*np.sin(theta)*np.cos(phi)*theta_t + L*m*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)*np.sin(phi)*np.sin(theta)*theta_t - L*m*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)*np.cos(phi)*np.cos(theta)*phi_t - L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t**2 - L*np.sin(phi)*np.sin(theta)*theta_t**2 + 2*L*np.cos(phi)*np.cos(theta)*phi_t*theta_t)*np.sin(phi)*np.cos(theta) - L*m*(-2*L*np.sin(phi)*np.cos(theta)*phi_t*theta_t - L*np.sin(theta)*np.cos(phi)*phi_t**2 - L*np.sin(theta)*np.cos(phi)*theta_t**2)*np.cos(phi)*np.cos(theta) + m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*(-2*L*np.sin(phi)*np.cos(theta)*phi_t - 2*L*np.sin(theta)*np.cos(phi)*theta_t)/2 + m*(-2*L*np.sin(phi)*np.sin(theta)*theta_t + 2*L*np.cos(phi)*np.cos(theta)*phi_t)*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)/2)/(L**2*m*np.sin(phi)**2*np.cos(theta)**2 + L**2*m*np.sin(theta)**2 + L**2*m*np.cos(phi)**2*np.cos(theta)**2),(L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*np.sin(phi)*np.cos(theta)*theta_t + L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*np.sin(theta)*np.cos(phi)*phi_t + L*m*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)*np.sin(phi)*np.sin(theta)*phi_t - L*m*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)*np.cos(phi)*np.cos(theta)*theta_t - L*m*(-L*np.sin(phi)*np.sin(theta)*phi_t**2 - L*np.sin(phi)*np.sin(theta)*theta_t**2 + 2*L*np.cos(phi)*np.cos(theta)*phi_t*theta_t)*np.sin(theta)*np.cos(phi) + L*m*(-2*L*np.sin(phi)*np.cos(theta)*phi_t*theta_t - L*np.sin(theta)*np.cos(phi)*phi_t**2 - L*np.sin(theta)*np.cos(phi)*theta_t**2)*np.sin(phi)*np.sin(theta) + Moment + m*(-2*L*np.sin(phi)*np.sin(theta)*phi_t + 2*L*np.cos(phi)*np.cos(theta)*theta_t)*(L*np.sin(phi)*np.cos(theta)*theta_t + L*np.sin(theta)*np.cos(phi)*phi_t)/2 + m*(-L*np.sin(phi)*np.sin(theta)*phi_t + L*np.cos(phi)*np.cos(theta)*theta_t)*(-2*L*np.sin(phi)*np.cos(theta)*theta_t - 2*L*np.sin(theta)*np.cos(phi)*phi_t)/2)/(L**2*m*np.sin(phi)**2*np.sin(theta)**2 + L**2*m*np.sin(theta)**2*np.cos(phi)**2)
    return theta_t,theta_2t,phi_t,phi_2t

t = np.arange(0,2.5,0.01)
y0=np.array([np.pi/4, 0,0, 0.5])
X,Xdot = generate_data(spherePend,t,y0)

def energy(X):
    theta, theta_t, phi, phi_t = np.hsplit(X,4)
    return L**2*m*(np.sin(theta)**2*phi_t**2 + theta_t**2 )/2 + L*m*g*np.cos(theta)

data_description = symbols('theta theta_t phi phi_t')
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

energyChange = 1.0*X[:,3]

stored_indices = ()
elements = tuple(x for x in range(len(expr)) if x not in stored_indices)
indices = itertools.combinations(elements, 3)

def countNumberOfElementsLargerThanThreshold(x,threshold = 1e-8):
    count = 0
    for i in range(len(x)):
        if abs(x[i]) > threshold:
            count = count +1
    return count
start = time.time()
for count,index in enumerate(indices):
    index_tup = index + stored_indices
    xi, sumResidual = np.linalg.lstsq(Gamma[:,index_tup], energyChange,rcond=None)[:2]
    if sumResidual.size==0 or sumResidual>1e-8: continue
    if countNumberOfElementsLargerThanThreshold(xi)<=2: continue
    expr_temp = [expr[i] for i in index_tup]
    Hamiltonian = generateExpression(xi,expr_temp,threshold=1e-8)
    print('Total Energy = ',Hamiltonian)
    print('Found result after ',time.time()-start,'s')
print('Elapsed time: ',time.time()-start)

