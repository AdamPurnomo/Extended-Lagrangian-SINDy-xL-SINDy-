#%%
from HLsearch import *
from scipy.integrate import solve_ivp
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

import itertools
import scipy.io as sio

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

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

"""Pendulum 1"""
t = np.arange(0,1,0.01)
y0=np.array([np.pi/4, 0])
X,Xdot = generate_data(pendulum,t,y0)

data_description  = symbols('x0, x0_t')
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=True)
print(expr)
Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)


u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]

# print('{} coefficients ='.format(xi.shape[0]),xi)
print('Now drop off small coefficients')
Hamiltonian,terms = generateSimplifiedExpression(xi,expr)

print('H = ',Hamiltonian)
Hamiltonian_old1 = Hamiltonian
xi_old1 = xi
terms1 = terms
expr1 = expr
#%%

"""Pendulum 2"""
t = np.arange(0,1,0.01)
y0=np.array([np.pi/4, 0])
X,Xdot = generate_data(pendulum,t,y0)

data_description = symbols('x1, x1_t')
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=True)
print(expr)
Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)


u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]
# print('{} coefficients ='.format(xi.shape[0]),xi)
print('Now drop off small coefficients')
Hamiltonian,terms = generateSimplifiedExpression(xi,expr)

print('H = ',Hamiltonian)
Hamiltonian_old2 = Hamiltonian
xi_old2 = xi
terms2 = terms
expr2 = expr
#%%

print('Now start calculating Double Pendulum')

t = np.arange(0,4,0.01)
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
#%%
expr_new0 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2))],use_sine=True)
expr_new1 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2),len(data_description))],use_sine=True)
#%%
print(expr_new0[1:])
print(expr_new1[1:])
expr_new = expr_new0[1:]+expr_new1[1:]
#%%
expr_new = buildFunctionExpressions(4,len(expr_new),expr_new)
# for i,expr_new_item in enumerate(expr_new):
#     print(i,expr_new_item) 

#%%
Theta_new = buildTimeSerieMatrixFromFunctions(X,expr_new, data_description)
Gamma_old1 = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr1    ,data_description)
Gamma_old2 = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr2    ,data_description)
Gamma_new = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr_new,data_description)
#%%
stored_indices = tuple(expr_new.index(str(f)) for f in terms1+terms2)

terms = [expr_new[i] for i in stored_indices]
print('Keeping terms: ',terms)
elements = tuple(x for x in range(len(expr_new)) if x not in stored_indices)
Hamiltonian_old = Hamiltonian_old1+Hamiltonian_old2

print(Hamiltonian_old)

combi_number = 2
indices = itertools.combinations(elements, combi_number)
zero_in_xi = np.array([collect(Hamiltonian_old,term).coeff(term) for term in terms],dtype=np.float32)
for _ in itertools.repeat(None, combi_number):
    zero_in_xi = np.insert(zero_in_xi,0,0)
print(zero_in_xi)

goodHamiltonian={}

'''trial and error with only k terms
Regression of y = X*xi while y = 0 might not give a good result. Therefore
the term H is divided into three parts where some of the parts are known from a simple system.
H total = Hc(theta_1,theta_2) + Ha(theta_1) + Hb(theta2).
In this case, the last two terms are already known from a simple system. 
dE/dt = 0 = Gamma_new*xi_new + Gamma_old1*xi_old1 + Gamma_old2*xi_old2
-Gamma_old1*xi_old1 -Gamma_old2*xi_old2 = Gamma_new*xi_new
'''

for count,index in enumerate(indices):
# index_tup = (2,5,16,25)
    index_tup = index + stored_indices
    xi_new = STRidge(Gamma_new[:,index_tup], -1*np.dot(Gamma_old1,xi_old1)-1*np.dot(Gamma_old2,xi_old2),0.0,1000, 10**-12, print_results=False)
    Hamiltonian = Hamiltonian_old+generateExpression(xi_new,[expr_new[i] for i in index_tup],threshold=1e-6)
    if np.linalg.norm(xi_new+zero_in_xi,2) < 1e-5: continue
    energyFunc = lambdify(data_description,Hamiltonian,'numpy')
    energy = np.array([energyFunc(*X[i,:]) for i in range(X.shape[0])])
    if np.var(energy)<1e-11 and abs(np.mean(energy))>1e-3:
        temp_terms = [sympify(expr_new[i]) for j,i in enumerate(index_tup) if abs(xi_new[j])>1e-8]
        Lagrangian = findLagrangianFromHamiltonian(Hamiltonian,temp_terms,data_description_sym,threshold=1e-6)
        if Lagrangian is not None and Lagrangian != '':
            goodHamiltonian[str(simplify(Hamiltonian))] = Lagrangian
            print('Found good result at ',count,'th trial: ',index_tup)

for H,L in goodHamiltonian.items():
    print('Hamiltonian is ',H)
    print('Lagrangian is ',L)
    print('')