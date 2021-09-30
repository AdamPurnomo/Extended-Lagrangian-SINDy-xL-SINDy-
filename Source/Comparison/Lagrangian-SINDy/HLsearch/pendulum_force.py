#%%
from HLsearch import *
from scipy.integrate import solve_ivp
import numpy as np
from sympy import symbols, var, diff, simplify, collect,solve
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.physics.mechanics import *

from operator import add,sub,mul

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])

#generating data for each state
t = np.arange(0,10,0.01)
y0=np.array([np.pi/4, 0])
X,Xdot = generate_data(pendulum,t,y0)

#producing candidate terms for the library
data_description = ()
for i in range(round(X.shape[1]/2)):
    data_description = data_description + symbols('x{}, x{}_t'.format(i,i))
print('Variables are:',data_description, '\n')
data_description_sym = data_description
data_description = list(str(descr) for descr in data_description)
expr = buildFunctionExpressions(2,X.shape[1],data_description,use_sine=True)
print('Library Function: ', expr, '\n')

#building library matrix
Theta = buildTimeSerieMatrixFromFunctions(X,expr, data_description)

#building time derivative of library matrix
Gamma = buildTimeDerivativeMatrixFromFunctions(X,Xdot,expr,data_description)

#solving xi if no energy dissipation exists (searching for null space)
u, s, vh = np.linalg.svd(0*Theta-Gamma,full_matrices=False)

xi = vh.T[:,-1]

print('Now drop off small coefficients')
Hamiltonian,terms = generateSimplifiedExpression(xi,expr)

print('H = ',Hamiltonian)
Lagrangian = findLagrangianFromHamiltonian(Hamiltonian,terms,data_description_sym)

#This lagrangian is not the accurate lagrangian yet. It needs to be multiplied by constant k.
#Let's call this lagrangian L*
print ('L = ', Lagrangian)
print('terms = ',terms, '\n' )
#%%

#Finding the constant k to find the true xi by introducing new data with actuation
def pendulum_force(t,x,m=1):
    return x[1],-9.81*np.sin(x[0])+m

t = np.arange(0,0.1,0.01)
y0=np.array([np.pi/4, 0])
Xf,Xfdot = generate_data(pendulum_force,t,y0)

#Constant k is the ratio of actuation force and estimated actuation force.
#Estimated actuation force is obtained from L*. 
# m = d(dL/dqdot)dt - dL/dq ; m* = d(dL*/dqdot)dt - dL*/dq ; L = kL* ; k = m/m*
dLdq_expr = diff(Lagrangian, data_description_sym[0])
dLdqdot_expr = diff(Lagrangian, data_description_sym[1])
dLdq = buildTimeSerieFromFunction(Xf,dLdq_expr,data_description_sym)
d_dLdqdot_dt = buildTimeDerivativeSerieFromFunction(Xf,Xfdot,dLdqdot_expr,data_description_sym)
fCal = d_dLdqdot_dt-dLdq
k = 1.0/np.mean(fCal)
#%%
Lagrangian = k*Lagrangian

print('L = ',Lagrangian, '\n')


dyn_x0 = dynamicsymbols('x0')
dyn_x0d = dynamicsymbols('x0',1)
new_data = (dyn_x0,dyn_x0d)

r = dynamicsymbols('r')

# print(list(zip(data_description_sym,new_data)))

Lagrangian = Lagrangian.subs(list(zip(data_description_sym,new_data)))
# Lagrangian.subs(data_description_sym[1],dyn_x0d)

print(Lagrangian)

N = ReferenceFrame('N')
A = N.orientnew('A', 'axis', [dyn_x0, N.z])
A.set_ang_vel(N, dyn_x0d*N.z)
moment = (A,r*A.z)

LM = LagrangesMethod(Lagrangian, [dyn_x0], forcelist=[moment], frame= N)
mprint(LM.form_lagranges_equations())
# mprint(LM.rhs())

op_point = {dyn_x0:np.pi/2,dyn_x0d:0}
# # A,B,inp_vec = LM.linearize(q_ind=[dyn_x0], qd_ind=[dyn_x0d], A_and_B=True, op_point=op_point)

linearizer = LM.to_linearizer(q_ind=[dyn_x0], qd_ind=[dyn_x0d])

A, B = linearizer.linearize(A_and_B=True, op_point=op_point)

print('A = ',A)
print('B = ',B)


# %%
