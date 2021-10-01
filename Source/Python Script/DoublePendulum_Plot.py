import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
import torch
import HLsearch as HL
import matplotlib.pyplot as plt


import time

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
tau = 0

def doublePendulum(t,y,M=0.0):
    q1,q2,q1_t,q2_t = y
    q1_2t = (-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(2*(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)))/(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)
    q2_2t = (-L2*g*m2*np.sin(q2) + m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t)/2 - m2*(2*L2*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q2)*q2_t + 2*L2*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q2) - 2*L2*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q2)*q2_t + 2*L2*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q2))/2 - m2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))*(-L1*g*m1*np.sin(q1) - L1*g*m2*np.sin(q1) + M + m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t)/2 - m2*(2*L1*(L1*np.sin(q1)*q1_t + L2*np.sin(q2)*q2_t)*np.cos(q1)*q1_t + 2*L1*(-L1*np.sin(q1)*q1_t**2 - L2*np.sin(q2)*q2_t**2)*np.cos(q1) - 2*L1*(L1*np.cos(q1)*q1_t + L2*np.cos(q2)*q2_t)*np.sin(q1)*q1_t + 2*L1*(L1*np.cos(q1)*q1_t**2 + L2*np.cos(q2)*q2_t**2)*np.sin(q1))/2)/(2*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)))/(-m2**2*(2*L1*L2*np.sin(q1)*np.sin(q2) + 2*L1*L2*np.cos(q1)*np.cos(q2))**2/(4*(m1*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2 + m2*(2*L1**2*np.sin(q1)**2 + 2*L1**2*np.cos(q1)**2)/2)) + m2*(2*L2**2*np.sin(q2)**2 + 2*L2**2*np.cos(q2)**2)/2)
    return q1_t,q2_t,q1_2t,q2_2t



#Saving Directory
rootdir = "../Double Pendulum/"

num_sample = 100
create_data = False
training = False

a = 2
b = 2
noiselevel = a*10**(-b)


if(create_data):
    num_sample = 100
    X, Xdot = [], []
    for i in range(num_sample):
        t = np.arange(0,5,0.01)
        theta1 = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(0,0)
        theta2 = np.random.uniform(-np.pi, np.pi)
        
        y0=np.array([theta1, theta2, thetadot, thetadot])
        x,xdot = generate_data(doublePendulum,t,y0)
        X.append(x)
        Xdot.append(xdot)
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    np.save(rootdir + "Data/X.npy", X)
    np.save(rootdir + "Data/Xdot.npy",Xdot)
else:
    X = np.load(rootdir + "Data/X.npy")
    Xdot = np.load(rootdir + "Data/Xdot.npy")



states_dim = 4
states = ()
states_dot = ()
for i in range(states_dim):
    if(i<states_dim//2):
        states = states + (symbols('x{}'.format(i)),)
        states_dot = states_dot + (symbols('x{}_t'.format(i)),)
    else:
        states = states + (symbols('x{}_t'.format(i-states_dim//2)),)
        states_dot = states_dot + (symbols('x{}_tt'.format(i-states_dim//2)),)
print('states are:',states)
print('states derivatives are: ', states_dot)


#Turn from sympy to str
states_sym = states
states_dot_sym = states_dot
states = list(str(descr) for descr in states)
states_dot = list(str(descr) for descr in states_dot)



#identifier = "no"
identifier = str(a) + "e-0" + str(b)
if(training==False):
    with open(rootdir + "Data/lagrangian_" + identifier + "_noise.txt") as f:
        L = f.readline()
        print(L)
        expr = L.split(" +")
        device = "cuda:0"
        xi_L = torch.ones(len(expr),device=device)
        mask = torch.ones(len(expr),device=device)


#obtaining equations of motions
x0 = dynamicsymbols(states[0])
x1 = dynamicsymbols(states[1])
x0_t = dynamicsymbols(states[0],1)
x1_t = dynamicsymbols(states[1],1)
L = eval(str(L))
LM = LagrangesMethod(L, [x0,x1])


LM.form_lagranges_equations()
eom = LM.rhs()


for i in range(len(eom)):
    eom[i] = eom[i].subs(x0_t, states_sym[2])
    eom[i] = eom[i].subs(x1_t, states_sym[3])
    eom[i] = eom[i].subs(x0, states_sym[0])
    eom[i] = eom[i].subs(x1, states_sym[1])

def predicteddoublependulum(t,y):
    from numpy import sin, cos
    x0,x1,x0_t,x1_t = y
    x0_tt = eval(str(eom[2]))
    x1_tt = eval(str(eom[3]))
    return x0_t,x1_t,x0_tt,x1_tt


## SINDy-PI ##

'''file name : Double Pendulum/Data/SINDy-PI_2e-02_noise.txt'''

def SINDyPI(t,y):
    from numpy import sin, cos
    z1,z2,z3,z4 = y
    dz1=1.0*z3
    dz2=z4
    dz3=(2.292e+7*sin(2.0*z1 - 2.0*z2) - 2.093e+7*sin(2.0*z1 - 1.0*z2) + 2.005e+7*sin(z1 - z2) + 1.238e+7*sin(z1 - 2.0*z2) + 7.062e+7*sin(z1) - 5.678e+6*sin(z2) + 1.477e+6*z3**2*sin(2.0*z1 - 1.0*z2) + 1.06e+6*z4**2*sin(2.0*z1 - 1.0*z2) - 1.609e+6*z4**2*sin(2.0*z1 - 2.0*z2) - 1.887e+6*z3**2*sin(z1 - z2) + 3.678e+6*z4**2*sin(z1 - z2) + 1.541e+6*z3**2*sin(z1 - 2.0*z2) + 9.951e+5*z4**2*sin(z1 - 2.0*z2))/(6.543e+6*cos(z1 - 1.0*z2)**2 - 1.0e+7)
    dz4=-(1.0*(8.16*sin(2.0*z1 - 1.0*z2) + 3.179*sin(2.0*z1 - 2.0*z2) - 2.906*sin(z1 - z2) + 1.408*sin(z1) - 9.208*sin(z2) + 0.8598*z3**2*sin(z1 - z2)))/(cos(z1 - 1.0*z2)**2 - 1.307)
    return dz1, dz2, dz3, dz4



## Training Results ##
import matplotlib.pyplot as plt

#forward
t = np.arange(0,5,0.01)

idx = 0
y_0 = X[idx,:]

X, Xdot = generate_data(doublePendulum, t, y_0) 

q_tt_true_wn = (Xdot[:,states_dim//2:].T).copy()
q_t_true_wn = (Xdot[:,:states_dim//2].T).copy()
q_true_wn = (X[:,:states_dim//2].T).copy()

#adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i]+noise
    Xdot[:,i] = Xdot[:,i]+noise

q_tt_true = Xdot[:,states_dim//2:].T
q_t_true = Xdot[:,:states_dim//2].T
q_true = X[:,:states_dim//2].T

#ours
Xpred, Xdotpred = generate_data(predicteddoublependulum, t, y_0) 

q_tt_pred = Xdotpred[:,states_dim//2:].T
q_t_pred = Xdotpred[:,:states_dim//2].T
q_pred = Xpred[:,:states_dim//2].T

#SINDy PI
Xspi, Xdotspi = generate_data(SINDyPI, t, y_0) 

q_tt_spi = Xdotspi[:,states_dim//2:].T
q_t_spi = Xdotspi[:,:states_dim//2].T
q_spi = Xspi[:,:states_dim//2].T



## Test Results ##
ttest = np.arange(5,10,0.01)
y_0 = np.array([q_true_wn[0,-1], q_true_wn[1,-1], q_t_true_wn[0,-1], q_t_true_wn[1,-1]])
Xtest,Xdottest = generate_data(doublePendulum, ttest, y_0)

y_0 = np.array([q_pred[0,-1], q_pred[1,-1], q_t_pred[0,-1], q_t_pred[1,-1]])
Xtestpred,Xdottestpred = generate_data(predicteddoublependulum, ttest, y_0)

y_0 = np.array([q_spi[0,-1], q_spi[1,-1], q_t_spi[0,-1], q_t_spi[1,-1]])
Xtestspi,Xdottestspi = generate_data(SINDyPI, ttest, y_0)



# Concatenante training and test data
t = np.concatenate((t, ttest))
q_tt_true = np.concatenate((q_tt_true, Xdottest[:,states_dim//2:].T), axis=1)
q_t_true = np.concatenate((q_t_true, Xtest[:,states_dim//2:].T), axis=1)
q_true = np.concatenate((q_true,Xtest[:,:states_dim//2].T), axis=1)

q_tt_pred = np.concatenate((q_tt_pred,Xdottestpred[:,states_dim//2:].T), axis=1)
q_t_pred = np.concatenate((q_t_pred, Xtestpred[:,states_dim//2:].T), axis=1)
q_pred = np.concatenate((q_pred,Xtestpred[:,:states_dim//2].T), axis=1)

q_tt_spi = np.concatenate((q_tt_spi,Xdottestspi[:,states_dim//2:].T), axis=1)
q_t_spi = np.concatenate((q_t_spi, Xtestspi[:,states_dim//2:].T), axis=1)
q_spi = np.concatenate((q_spi,Xtestspi[:,:states_dim//2].T), axis=1)



fig, ax = plt.subplots(6,1,figsize=(5,8))

ax[0].plot(t, q_tt_true[0,:], label='True Model')
ax[0].plot(t, q_tt_pred[0,:], 'r--',label='Ours')
ax[0].plot(t, q_tt_spi[0,:], ':g', label='SINDy-PI')
ax[0].set_ylabel('$\ddot{\\theta_1}$ (rad/s$^2$)')
ax[0].vlines(5,0,1,transform=ax[0].get_xaxis_transform(),colors='k')
ax[0].set_xlim([0,10])

ax[1].plot(t, q_tt_true[1,:], label='True Model')
ax[1].plot(t, q_tt_pred[1,:], 'r--',label='Ours')
ax[1].plot(t, q_tt_spi[1,:], ':g',label='SINDy-PI')
ax[1].set_ylabel('$\ddot{\\theta_2}$ (rad/s$^2$)')
ax[1].vlines(5,0,1,transform=ax[1].get_xaxis_transform(),colors='k')
ax[1].set_xlim([0,10])

ax[2].plot(t, q_t_true[0,:], label='True Model')
ax[2].plot(t, q_t_pred[0,:], 'r--',label='Ours')
ax[2].plot(t, q_t_spi[0,:], ':g',label='SINDy-PI')
ax[2].set_ylabel('$\dot{\\theta_1}$ (rad/s)')
ax[2].vlines(5,0,1,transform=ax[2].get_xaxis_transform(),colors='k')
ax[2].set_xlim([0,10])

ax[3].plot(t, q_t_true[1,:], label='True Model')
ax[3].plot(t, q_t_pred[1,:], 'r--',label='Ours')
ax[3].plot(t, q_t_spi[1,:], ':g',label='SINDy-PI')
ax[3].set_ylabel('$\dot{\\theta_2}$ (rad/s)')
ax[3].vlines(5,0,1,transform=ax[3].get_xaxis_transform(),colors='k')
ax[3].set_xlim([0,10])

ax[4].plot(t, q_true[0,:], label='True Model')
ax[4].plot(t, q_pred[0,:], 'r--',label='Ours')
ax[4].plot(t, q_spi[0,:], ':g',label='SINDy-PI')
ax[4].set_ylabel('$\\theta_1$ (rad)')
ax[4].vlines(5,0,1,transform=ax[4].get_xaxis_transform(),colors='k')
ax[4].set_xlim([0,10])
ax[4].set_ylim([-1.8,1.8])

ax[5].plot(t, q_true[1,:], label='True Model')
ax[5].plot(t, q_pred[1,:], 'r--',label='Predicted Model')
ax[5].plot(t, q_spi[1,:], ':g',label='SINDy-PI')
ax[5].set_ylabel('$\\theta_2$ (rad)')
ax[5].vlines(5,0,1,transform=ax[5].get_xaxis_transform(),colors='k')
ax[5].set_xlim([0,10])
ax[5].set_ylim([-4,4])

Line, Label = ax[0].get_legend_handles_labels()
fig.legend(Line, Label, loc='upper right', bbox_to_anchor=(1.5, 0.98))

fig.tight_layout()
fig.savefig(rootdir + "Images/training_" + identifier + "_noise.png", bbox_inches='tight',dpi=400)





