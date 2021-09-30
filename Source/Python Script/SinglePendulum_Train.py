import numpy as np
import sys 


from sympy import symbols, simplify, derive_by_array
from scipy.integrate import solve_ivp
from xLSINDy import *
from sympy.physics.mechanics import *
from sympy import *
import sympy
import torch
import sys 
sys.path.append(r'../../../HLsearch/')
import HLsearch as HL


import time

def generate_data(func, time, init_values):
    sol = solve_ivp(func,[time[0],time[-1]],init_values,t_eval=time, method='RK45',rtol=1e-10,atol=1e-10)
    return sol.y.T, np.array([func(0,sol.y.T[i,:]) for i in range(sol.y.T.shape[0])],dtype=np.float64)

def pendulum(t,x):
    return x[1],-9.81*np.sin(x[0])


#Saving Directory
rootdir = "../Single Pendulum/Data/"

num_sample = 100
create_data = False
training = True
save = False
noiselevel = 2e-2


if(create_data):
    X, Xdot = [], []
    for i in range(num_sample):
        t = np.arange(0,5,0.01)

        theta = np.random.uniform(-np.pi, np.pi)
        thetadot = np.random.uniform(-2.1,2.1)
        cond = 0.5*thetadot**2 - np.cos(theta)
        #checking condition so that it does not go full loop
        while(cond>0.99):
            theta = np.random.uniform(-np.pi, np.pi)
            thetadot = np.random.uniform(-2.1,2.1)
            cond = 0.5*thetadot**2 - np.cos(theta)
        
        y_0 = np.array([theta, thetadot])
        x,xdot = generate_data(pendulum, t, y_0)
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

states_dim = 2
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


#build function expression for the library in str
expr= HL.buildFunctionExpressions(2,states_dim,states,use_sine=True)
expr.pop(5)


device = 'cuda:0'


Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=True)
Eta = Eta.to(device)
Zeta = Zeta.to(device)
Delta = Delta.to(device)


mask = torch.ones(len(expr),device=device)
xi_L = torch.ones(len(expr), device=device).data.uniform_(-10,10)
prevxi_L = xi_L.clone().detach()


def loss(pred, targ, coef):
    loss = torch.mean((pred - targ)**2) 
    return loss 


def clip(w, alpha):
    clipped = torch.minimum(w,alpha)
    clipped = torch.maximum(clipped,-alpha)
    return clipped

def proxL1norm(w_hat, alpha):
    if(torch.is_tensor(alpha)==False):
        alpha = torch.tensor(alpha)
    w = w_hat - clip(w_hat,alpha)
    return w


def training_loop(coef, prevcoef, Zeta, Eta, Delta,xdot, bs, lr, lam):
    loss_list = []
    tl = xdot.shape[0]
    n = xdot.shape[1]

    if(torch.is_tensor(xdot)==False):
        xdot = torch.from_numpy(xdot).to(device).float()
    
    for i in range(tl//bs):
        
        #computing acceleration with momentum
        v = (coef + ((i-1)/(i+2))*(coef - prevcoef)).clone().detach().requires_grad_(True)
        
        prevcoef = coef.clone().detach()


        #Computing loss
        zeta = Zeta[:,:,:,i*bs:(i+1)*bs]
        eta = Eta[:,:,:,i*bs:(i+1)*bs]
        delta = Delta[:,:,i*bs:(i+1)*bs]
        x_t = xdot[i*bs:(i+1)*bs,:]

        #forward
        q_tt_pred = lagrangianforward(v,mask,zeta,eta,delta,x_t,device)
        q_tt_true = xdot[i*bs:(i+1)*bs,n//2:].T
        

        
        lossval = loss(q_tt_pred, q_tt_true, coef)
        
        #Backpropagation
               
        lossval.backward()
        with torch.no_grad():
            vhat = v - lr*v.grad
            coef = (proxL1norm(vhat,lr*lam)).clone().detach()

        
    
        
        loss_list.append(lossval.item())
    print("Average loss : " , torch.tensor(loss_list).mean().item())
    return coef, prevcoef, torch.tensor(loss_list).mean().item()


Epoch = 100
i = 1
lr = 1e-4
lam = 0.1
temp = 1000
while(i<=Epoch):
    print("\n")
    print("Stage 1")
    print("Epoch "+str(i) + "/" + str(Epoch))
    print("Learning rate : ", lr)
    xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
    if(temp <=5e-3):
        break
    if(temp <=1e-1):
        lr = 1e-5
    temp = lossitem
    i+=1


## Thresholding small indices ##
threshold = 1e-2
surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
expr = np.array(expr)[surv_index].tolist()

xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
prevxi_L = xi_L.clone().detach()
mask = torch.ones(len(expr),device=device)

## obtaining analytical model
xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=2)
L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
print(simplify(L))



## Next round Selection ##
for stage in range(2):
    
    #Redefine computation after thresholding
    Zeta, Eta, Delta = LagrangianLibraryTensor(X,Xdot,expr,states,states_dot,scaling=False)
    Eta = Eta.to(device)
    Zeta = Zeta.to(device)
    Delta = Delta.to(device)

    #Training
    Epoch = 100
    i = 1
    lr = 3e-5
    if(stage==1):
        lam = 0
    else:
        lam = 0.1
    temp = 1000
    while(i<=Epoch):
        print("\n")
        print("Stage " + str(stage+2))
        print("Epoch "+str(i) + "/" + str(Epoch))
        print("Learning rate : ", lr)
        xi_L , prevxi_L, lossitem= training_loop(xi_L,prevxi_L,Zeta,Eta,Delta,Xdot,128,lr=lr,lam=lam)
        temp = lossitem
        if(temp <=1e-6):
            break
        i+=1
    
    ## Thresholding small indices ##
    threshold = 1e-1
    surv_index = ((torch.abs(xi_L) >= threshold)).nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expr = np.array(expr)[surv_index].tolist()

    xi_L =xi_L[surv_index].clone().detach().requires_grad_(True)
    prevxi_L = xi_L.clone().detach()
    mask = torch.ones(len(expr),device=device)

    ## obtaining analytical model
    xi_Lcpu = np.around(xi_L.detach().cpu().numpy(),decimals=3)
    L = HL.generateExpression(xi_Lcpu,expr,threshold=1e-2)
    print("Result stage " + str(stage+2) + ":" , simplify(L))


if(save==True):
    #Saving Equation in string
    text_file = open(rootdir + "lagrangian_" + str(noiselevel)+ "_noise.txt", "w")
    text_file.write(L)
    text_file.close()


# %%



