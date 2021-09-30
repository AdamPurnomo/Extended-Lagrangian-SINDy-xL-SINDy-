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

t = np.arange(0,10,0.01)
y0=np.array([np.pi/4, 0])
X,Xdot = generate_data(pendulum,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
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

print('Now start calculating CartPole')

g=9.81
mp=0.1
mc=1
l=1

def cartpole(t,y,f=0):
    theta,thetadot,x,xdot = y
    xdotdot = (f+mp*np.sin(theta)*(l*thetadot**2+g*np.cos(theta)))/(mc+mp*np.sin(theta)**2)
    thetadotdot = (-f*np.cos(theta)-mp*l*thetadot**2*np.cos(theta)*np.sin(theta)-(mc+mp)*g*np.sin(theta))/(l*(mc+mp*np.sin(theta)**2))
    return thetadot,thetadotdot,xdot,xdotdot

t = np.arange(0,10,0.01)
y0=np.array([np.pi/4, 0,0, 0])
X,Xdot = generate_data(cartpole,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
# print(data_description_sym)
# print([data_description[i] for i in range(round(len(data_description)/2))])

expr_new0 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2))],use_sine=True)
expr_new1 = buildFunctionExpressions(1,round(X.shape[1]/2),[data_description[i] for i in range(round(len(data_description)/2),len(data_description))],use_sine=False)

print(expr_new0[1:])
print(expr_new1)
expr_new = expr_new0[1:]+expr_new1
expr_new = buildFunctionExpressions(3,len(expr_new),expr_new)
# for i,expr_new_item in enumerate(expr_new):
#     print(i,expr_new_item) 

Theta_new = buildTimeSerieMatrixFromFunctions(X,expr_new, data_description)
Gamma_old = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr    ,data_description)
Gamma_new = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr_new,data_description)

stored_indices = tuple(expr_new.index(str(f)) for f in terms)

print('Keeping terms: ',[expr_new[i] for i in stored_indices])
elements = tuple(x for x in range(len(expr_new)) if x not in stored_indices)
Hamiltonian_old = Hamiltonian

combi_number = 2
indices = itertools.combinations(elements, combi_number)
zero_in_xi = np.array([collect(Hamiltonian_old,term).coeff(term) for term in terms],dtype=np.float32)
for _ in itertools.repeat(None, combi_number):
    zero_in_xi = np.insert(zero_in_xi,0,0)
# print(zero_in_xi)

goodHamiltonian={}

for count,index in enumerate(indices):
# index_tup = (2,5,16,25)
    index_tup = index + stored_indices
    xi_new = STRidge(Gamma_new[:,index_tup], -1*np.dot(Gamma_old,xi),0.0,1000, 10**-12, print_results=False)
    Hamiltonian = Hamiltonian_old+generateExpression(xi_new,[expr_new[i] for i in index_tup],threshold=1e-6)
    if np.linalg.norm(xi_new+zero_in_xi,2) < 1e-5: continue
    energyFunc = lambdify(data_description,Hamiltonian,'numpy')
    energy = np.array([energyFunc(*X[i,:]) for i in range(X.shape[0])])
    # var = np.var(energy)
    if np.var(energy)<1e-13 and abs(np.mean(energy))>1e-3:
        temp_terms = [sympify(expr_new[i]) for j,i in enumerate(index_tup) if abs(xi_new[j])>1e-8]
        Lagrangian = findLagrangianFromHamiltonian(Hamiltonian,temp_terms,data_description_sym,threshold=1e-6)
        if Lagrangian is not None and Lagrangian != '':
            goodHamiltonian[str(simplify(Hamiltonian))] = Lagrangian
            print('Found good result at ',count,'th trial: ',index_tup)

for H,L in goodHamiltonian.items():
    print('')
    print('Hamiltonian is ',H)
    print('Lagrangian is ',L)
