# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import os


# %%
rootdir = "Spherical Pendulum/Data/"
imdir = "Spherical Pendulum/Images/"


# %%
q_true = np.load(rootdir + "q_true.npy")


# %%
q_pred_1e03_noise = np.load(rootdir + "q_pred_1e-03_noise.npy")
q_pred_2e02_noise = np.load(rootdir + "q_pred_2e-02_noise.npy")
q_pred_6e02_noise = np.load(rootdir + "q_pred_6e-02_noise.npy")
q_pred_1e01_noise = np.load(rootdir + "q_pred_1e-01_noise.npy")


# %%
t = np.arange(0,10,0.01)


# %%
def showSphericalPend(fig,ax,q,i,title, trace=True,start_trace=0):
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    theta = q[0,:i+1]
    phi = q[1,:i+1]
    # print(theta.shape)
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    # print(z.shape)
    ax.plot((x[-1],0),(y[-1],0),(z[-1],0),color='k')
    ax.scatter3D(x[-1],y[-1],z[-1],color='k')
    if(trace==True):
        ax.plot3D(x[start_trace:], y[start_trace:], z[start_trace:])
    #ax.text2D(0.1, 0.95, title, transform=ax.transAxes, fontsize=12)
    ax.title.set_text(title)


# %%
def show(i):
    fig= plt.figure(figsize=(9,7))
    
    ax0 = fig.add_subplot(2,3,1, projection='3d')
    ax0.axis('off')

    ax1 = fig.add_subplot(2,3,2, projection='3d')
    #ax1.set_aspect(aspect=1)

    ax2 = fig.add_subplot(2,3,4, projection='3d')
    #ax2.set_aspect(aspect=1)

    ax3 = fig.add_subplot(2,3,5, projection='3d')
    #ax3.set_aspect(aspect=1)

    ax4 = fig.add_subplot(2,3,6, projection='3d')
    #ax4.set_aspect(aspect=1)


    if(i>=500):
        showSphericalPend(fig,ax1,q_true,i,"True Model",trace=True, start_trace=500)
        showSphericalPend(fig,ax2,q_pred_1e03_noise,i,"xL-SINDy at $\sigma=10^{-3}$", trace=True,start_trace=500)
        showSphericalPend(fig,ax3,q_pred_2e02_noise,i,"xL-SINDy at $\sigma=2 \\times 10^{-2}$", trace=True,start_trace=500)
        showSphericalPend(fig,ax4,q_pred_6e02_noise,i,"xL-SINDy at $\sigma=6 \\times 10^{-2}$", trace=True,start_trace=500)
        ax0.text(0.5, 0.5, 0.5, 'Validation Period', horizontalalignment='center',verticalalignment='center', fontsize=15)
    else:
        showSphericalPend(fig,ax1,q_true,i,"True Model",trace=False)
        ax0.text(0.5, 0.5, 0.5, 'Training Period', horizontalalignment='center',verticalalignment='center', fontsize=15)
    
        ax1.set_xlim3d(-1, 1)
        ax1.set_ylim3d(-1,1)
        ax1.set_zlim3d(-1,1)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax2.set_xlim3d(-1, 1)
        ax2.set_ylim3d(-1,1)
        ax2.set_zlim3d(-1,1)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')


        ax3.set_xlim3d(-1, 1)
        ax3.set_ylim3d(-1,1)
        ax3.set_zlim3d(-1,1)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('z')


        ax4.set_xlim3d(-1, 1)
        ax4.set_ylim3d(-1,1)
        ax4.set_zlim3d(-1,1)
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('z')

        #ax2.text2D(0.1, 0.95, "xL-SINDy at $\sigma=10^{-3}$", transform=ax2.transAxes, fontsize=12)
        #ax3.text2D(0.1, 0.95, "xL-SINDy at $\sigma=2\\times10^{-2}$", transform=ax3.transAxes, fontsize=12)
        #ax4.text2D(0.1, 0.95, "xL-SINDy at $\sigma=6\\times10^{-2}$", transform=ax4.transAxes, fontsize=12)
        ax2.title.set_text("xL-SINDy at $\sigma=10^{-3}$")
        ax3.title.set_text("xL-SINDy at $\sigma=2\\times10^{-2}$")
        ax4.title.set_text("xL-SINDy at $\sigma=6\\times10^{-2}$")

    fig.tight_layout()
    fig.savefig(os.path.join(imdir, '{:03d}.png'.format(i)), dpi = 400)
    plt.close(fig)


# %%
for i in range(t.shape[0]):
    show(i)


# %%



