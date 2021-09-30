# %%
from HLsearch import *
from scipy.integrate import solve_ivp
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

t = np.arange(0,10,0.01)
y0=np.array([np.pi/4, 0])
X,Xdot = generate_data(pendulum,t,y0)
# %%
#producing candidate terms for the library
data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=True)
print(expr)
#%%
#building library matrix
Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

#building time derivative of library matrxix
Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)

#solving xi if no energy dissipation exists (searching for null space)
u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]
#%%

print('Now drop off small coefficients')
Hamiltonian,terms = generateSimplifiedExpression(xi,expr)

print('H = ',Hamiltonian)
print('terms = ',terms)
Lagrangian = findLagrangianFromHamiltonian(Hamiltonian,terms,data_description_sym)
print('L = ',Lagrangian)