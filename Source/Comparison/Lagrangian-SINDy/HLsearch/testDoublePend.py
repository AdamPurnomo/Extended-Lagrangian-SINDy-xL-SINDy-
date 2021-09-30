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

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81

def doublePendulumForce(t,y,M=1.0):
    q1,q1_t,q2,q2_t = y
    q1_2t = (-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(2*(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)))/(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)
    q2_2t = (-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)
    return q1_t,q1_2t,q2_t,q2_2t

t = np.arange(0,1,0.01)
y0=np.array([3*np.pi/7, 0, 3*np.pi/4, 0])
X,Xdot = generate_data(doublePendulumForce,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)

expr_new0 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2))],use_sine=True)
expr_new1 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2),len(data_description))],use_sine=True)

print(expr_new0[1:])
print(expr_new1[1:])
expr_new = expr_new0[1:]+expr_new1[1:]
expr = buildFunctionExpressions(4,len(expr_new),expr_new)

# expr = buildFunctionExpressions(3,len(expr_new),expr_new)
for i,expr_new_item in enumerate(expr):
    print(i,expr_new_item)

print(len(expr),' terms are: ',expr)

Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)

energyChange = 1.0*X[:,1]

stored_indices = (2,5,6,15)
elements = tuple(x for x in range(len(expr)) if x not in stored_indices)
indices = itertools.combinations(elements, 2)

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
    # if np.var(Gamma[:,index_tup]@xi-energyChange) > 1e-5: continue
    xi = np.around(xi,decimals=6)
    expr_temp = [expr[i] for i in index_tup]
    Hamiltonian = generateExpression(xi,expr_temp,threshold=1e-8)
    print('Total Energy = ',Hamiltonian)
    # u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

    # xi = vh.T[:,-1]

    # print('{} coefficients ='.format(xi.shape[0]),xi)
# print('Total Energy = ',Hamiltonian)
print('Elapsed time: ',time.time()-start)
print(simplify(Hamiltonian))

