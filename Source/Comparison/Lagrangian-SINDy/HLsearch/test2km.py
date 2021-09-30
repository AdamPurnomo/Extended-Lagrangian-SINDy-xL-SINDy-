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

g=9.81
mp=0.1
mc=1.0
l=1.0
def twoSpringMassForced(t,y,f=1.0,k1=25,k2=9):
    x1,x1_t,x2,x2_t = y
    x1_2t = -(k1+k2)*x1 +k2*x2
    x2_2t = -k2*(x2-x1) + f
    return x1_t,x1_2t,x2_t,x2_2t

t = np.arange(0,0.5,0.01)
y0=np.array([0.2, 0,0.5, 0])
X,Xdot = generate_data(twoSpringMassForced,t,y0)

data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description)
data_description_sym = data_description
print(data_description_sym)
data_description = list(str(descr) for descr in data_description)
expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=False)
# expr = buildFunctionExpressions(3,len(expr_new),expr_new)
for i,expr_new_item in enumerate(expr):
    print(i,expr_new_item)

print(len(expr),' terms are: ',expr)

Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)

energyChange = 1.0*X[:,3]

stored_indices = (4,6)
elements = tuple(x for x in range(len(expr)) if x not in stored_indices)
indices = itertools.combinations(elements, 3)

goodHamiltonian={}

def countNumberOfElementsLargerThanThreshold(x,threshold = 1e-8):
    count = 0
    for i in range(len(x)):
        if abs(x[i]) > threshold:
            count = count +1
    return count
start = time.time()

for count,index in enumerate(indices):
# index_tup = (2,5,16,25)
    index_tup = index + stored_indices
    xi, sumResidual = np.linalg.lstsq(Gamma[:,index_tup], energyChange,rcond=None)[:2]
    # print(xi)
    if sumResidual.size==0 or sumResidual>1e-3: continue
    if countNumberOfElementsLargerThanThreshold(xi)<=2: continue
    # if np.var(Gamma[:,index_tup]@xi-energyChange) > 1e-5: continue
#     xi = np.around(xi,decimals=12)
    expr_temp = [sympify(expr[i]) for j,i in enumerate(index_tup) if abs(xi[j])>1e-8]
    Hamiltonian = generateExpression(xi,expr_temp,threshold=1e-8)
    if Hamiltonian=='': continue
    # print('Total Energy = ',Hamiltonian)
    # print(expr_temp)
    Lagrangian = findLagrangianFromHamiltonian(Hamiltonian,expr_temp,data_description_sym,threshold=1e-8)
    # print(Lagrangian)
    if Lagrangian is not None and Lagrangian != '':
        goodHamiltonian[Hamiltonian] = Lagrangian
        print('Found good result at ',count,'th trial: ',index_tup)
print('Elapsed time: ',np.around(time.time()-start,2), 's')

for H,L in goodHamiltonian.items():
    dLdq_expr = diff(L, 'x1')
    dLdqdot_expr = diff(L, 'x1_t')
    dLdq = buildTimeSerieFromFunction(X,dLdq_expr,data_description_sym)
    d_dLdqdot_dt = buildTimeDerivativeSerieFromFunction(X,Xdot,dLdqdot_expr,data_description_sym)
    fCal = d_dLdqdot_dt-dLdq
    if np.var(fCal)>1e-10: continue
    print('')
    print('Hamiltonian is ',H)
    print('Lagrangian is ',L)  

