import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
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

g=9.81
mp=0.5

mc=1
l=1

def cartpole(t,y,f=0.0):
    theta,x,thetadot,xdot = y
    xdotdot = (f+mp*np.sin(theta)*(l*thetadot**2+g*np.cos(theta)))/(mc+mp*np.sin(theta)**2)
    thetadotdot = (-f*np.cos(theta)-mp*l*thetadot**2*np.cos(theta)*np.sin(theta)-(mc+mp)*g*np.sin(theta))/(l*(mc+mp*np.sin(theta)**2))
    return thetadot,xdot,thetadotdot,xdotdot


#Saving Directory
rootdir = "../Cart Pendulum/Data/"

num_sample = 100
create_data = False
training = True
save = False
noiselevel = 2e-2


if(create_data):
    print("Creating data ...")
    X, Xdot = [], []
    for i in range(num_sample):
        t = np.arange(0,5,0.01)
        theta = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(0,0)

        y0=np.array([theta, thetadot,0, 0])
        x,xdot = generate_data(cartpole,t,y0)
        X.append(x)
        Xdot.append(xdot)
    X = np.vstack(X)
    Xdot = np.vstack(Xdot)
    if(save==True):
        np.save(rootdir + "X.npy", X)
        np.save(rootdir + "Xdot.npy",Xdot)

  
else:
    X = np.load(rootdir + "X.npy")
    Xdot = np.load(rootdir + "Xdot.npy")


#adding noise
mu, sigma = 0, noiselevel
noise = np.random.normal(mu, sigma, X.shape[0])
for i in range(X.shape[1]):
    X[:,i] = X[:,i]+noise
    Xdot[:,i] = Xdot[:,i]+noise


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


#Separating states of pendulum and cart
pendulum_states = []
cartpole_states = []
for i in range(states_dim):
    if(i%2==0):
        pendulum_states.append(states[i])
    else:
        cartpole_states.append(states[i])


#build function expression for the library in str
pend_terms = HL.buildFunctionExpressions(1,states_dim//2,pendulum_states,use_sine=True)
cartpole_terms = HL.buildFunctionExpressions(1,states_dim//2,cartpole_states,use_sine=False)

#Assuming we get a prior knowledge about a single pendulum equations
temp = pend_terms[1:] + cartpole_terms
expr = HL.buildFunctionExpressions(3,len(temp),temp)


#Creating library tensor
Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot, scaling=True)


#separating known and unknown terms
expr = np.array(expr)
i1 = np.where(expr == 'x0_t**2')[0]

#Deleting unused terms
idx = np.arange(0,len(expr))
idx = np.delete(idx,i1)
known_expr = expr[i1].tolist()
expr = np.delete(expr,i1)

#non-penalty index from prev knowledge
i2 = np.where(expr == 'cos(x0)')[0][0]
nonpenaltyidx = [i2]

expr = expr.tolist()

Zeta_ = Zeta[:,:,i1,:].clone().detach()
Eta_ = Eta[:,:,i1,:].clone().detach()
Delta_ = Delta[:,i1,:].clone().detach()

Zeta = Zeta[:,:,idx,:]
Eta = Eta[:,:,idx,:]
Delta = Delta[:,idx,:]


#Moving to Cuda
device = 'cuda:0'

Zeta = Zeta.to(device)
Eta = Eta.to(device)
Delta = Delta.to(device)

Zeta_ = Zeta_.to(device)
Eta_ = Eta_.to(device)
Delta_ = Delta_.to(device)



xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
prevxi_L = xi_L.clone().detach()
c = torch.ones(len(known_expr), device=device)



def loss(pred, targ):
    loss = torch.mean((pred - targ)**2) 
    return loss 


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha, nonpenaltyidx):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    for idx in nonpenaltyidx:
        w[idx] = w_hat[idx]
    return w


def training_loop(c, coef, prevcoef, RHS, LHS, xdot, bs, lr, lam, momentum=True):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]

    Zeta_, Eta_, Delta_ = LHS
    Zeta, Eta, Delta = RHS

    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()
    
    v = coef.clone().detach().requires_grad_(True)
    prev = v
    
    for i in range(tl//bs):
                
        #computing acceleration with momentum
        if(momentum==True):
            vhat = (v + ((i-1)/(i+2))*(v - prev)).clone().detach().requires_grad_(True)
        else:
            vhat = v.requires_grad_(True).clone().detach().requires_grad_(True)
   
        prev = v

        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]

        zeta_ = Zeta_[:,:,:,i*bs:(i+1)*bs]
        eta_ = Eta_[:,:,:,i*bs:(i+1)*bs]
        delta_ = Delta_[:,:,i*bs:(i+1)*bs]
        
        x_t = xdot[i*bs:(i+1)*bs,:]

        #forward
        pred = -ELforward(vhat,zeta,eta,delta,x_t,device)
        targ = ELforward(c,zeta_,eta_,delta_,x_t,device)
        
        lossval = loss(pred, targ)
        
        #Backpropagation
        lossval.backward()

        with torch.no_grad():
            v = vhat - lr*vhat.grad
            v = (proxL1norm(v,lr*lam,nonpenaltyidx))
            
            # Manually zero the gradients after updating weights
            vhat.grad = None
        
        
    
        
        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return v, prevcoef, torch.tensor(loss_list).mean().item()

## First Round Selection ##
Epoch = 100
i = 0
lr = 1e-5
lam = 1
temp = 1000
RHS = [Zeta, Eta, Delta]
LHS = [Zeta_, Eta_, Delta_]
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L, prevxi_L, lossitem= training_loop(c, xi_L,prevxi_L,RHS,LHS,Xdot,128,lr=lr,lam=lam)
    temp = lossitem
    if(i>50):
        lr = 1e-5
    i+=1


## Thresholding
threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-3)
print("Result stage 1: ", simplify(L))


## Next round selection ##
for stage in range(3):

    #Redefine computation after thresholding
    expr.append(known_expr[0])
    Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot, scaling=False)

    expr = np.array(expr)
    i1 = np.where(expr == 'x0_t**2')[0]
    idx = np.arange(0,len(expr))
    idx = np.delete(idx,i1)
    known_expr = expr[i1].tolist()
    expr = np.delete(expr,i1).tolist()
    nonpenaltyidx = []

    Zeta_ = Zeta[:,:,i1,:].clone().detach()
    Eta_ = Eta[:,:,i1,:].clone().detach()
    Delta_ = Delta[:,i1,:].clone().detach()

    Zeta = Zeta[:,:,idx,:]
    Eta = Eta[:,:,idx,:]
    Delta = Delta[:,idx,:]

    Zeta = Zeta.to(device)
    Eta = Eta.to(device)
    Delta = Delta.to(device)
    Zeta_ = Zeta_.to(device)
    Eta_ = Eta_.to(device)
    Delta_ = Delta_.to(device)

    #Training
    Epoch = 100
    i = 0
    lr += lr
    if(stage==3):
        lam = 0
    else:
        lam = 0.1
    temp = 1000
    RHS = [Zeta, Eta, Delta]
    LHS = [Zeta_, Eta_, Delta_]
    while(i<=Epoch):
        print("\n")
        print("Stage " + str(stage+2))
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L, prevxi_L, lossitem= training_loop(c, xi_L,prevxi_L,RHS,LHS,Xdot,128,lr=lr,lam=lam)
        temp = lossitem
        i+=1
        if(temp <2e-3):
            break
    
    ## Thresholding
    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
    print("Result stage " + str(stage+2) + ":" , simplify(L))


## Adding known terms
L = str(simplify(L)) + " + " + known_expr[0]
print("Obtained Lagrangian:" ,L)

expr = expr + known_expr
xi_L = torch.cat((xi_L, c))
mask = torch.ones(len(xi_L),device=device)

if(save==True):
    #Saving Equation in string
    text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
    text_file.write(L)
    text_file.close()


