from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import os


### Double Pendulum ###
print("Saving Double Pendulum ...")
rootdir = "../Double Pendulum/Data 1/"
imdir = "../Double Pendulum/Images/"

q_true = np.load(rootdir + "q_true.npy")
q_pred_1e03_noise = np.load(rootdir + "q_pred_1e-03_noise.npy")
q_pred_2e02_noise = np.load(rootdir + "q_pred_2e-02_noise.npy")
q_pred_6e02_noise = np.load(rootdir + "q_pred_6e-02_noise.npy")
q_pred_1e01_noise = np.load(rootdir + "q_pred_1e-01_noise.npy")


t = np.arange(0,10,0.01)

def showDoublePend(fig,ax,q,i,title,trace=True,start_trace=0):
    l1,l2=1.0,1.0
    ax.set_xlim((-2.5, 2.5))
    ax.set_ylim((-2.5, 2.5))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    theta0 = q[0,:i+1]
    theta1 = q[1,:i+1]
    c0,s0 = np.cos(theta0), np.sin(theta0)
    c1,s1 = np.cos(theta1), np.sin(theta1)
    x0,y0 = l1*s0,-l1*c0
    x1,y1 = x0+l2*s1,y0-l2*c1
    ax.plot((0,x0[-1]), (0, y0[-1]), color='k')
    ax.plot((x0[-1],x1[-1]), (y0[-1], y1[-1]), color='k')
    if(trace==True):
        ax.plot(x0[start_trace:],y0[start_trace:])
        ax.plot(x1[start_trace:],y1[start_trace:])
    circle1 = plt.Circle((x0[-1],y0[-1]),0.1,color='k')
    circle2 = plt.Circle((x1[-1],y1[-1]),0.1,color='k')
    ax.add_artist(circle1)
    ax.add_artist(circle2)
    ax.title.set_text(title)


def saveDoublePend(i):
    fig= plt.figure(figsize=(9,6))
    
    ax0 = fig.add_subplot(2,3,1)
    ax0.axis('off')

    ax1 = fig.add_subplot(2,3,2)
    ax1.set_aspect(aspect=1)

    ax2 = fig.add_subplot(2,3,4)
    ax2.set_aspect(aspect=1)

    ax3 = fig.add_subplot(2,3,5)
    ax3.set_aspect(aspect=1)

    ax4 = fig.add_subplot(2,3,6)
    ax4.set_aspect(aspect=1)


    if(i>=500):
        showDoublePend(fig,ax1,q_true,i,"True Model",trace=True, start_trace=500)
        showDoublePend(fig,ax2,q_pred_1e03_noise,i,"xL-SINDy at $\sigma=10^{-3}$", trace=True,start_trace=500)
        showDoublePend(fig,ax3,q_pred_2e02_noise,i,"xL-SINDy at $\sigma=2 \\times 10^{-2}$", trace=True,start_trace=500)
        showDoublePend(fig,ax4,q_pred_6e02_noise,i,"xL-SINDy at $\sigma=6 \\times 10^{-2}$", trace=True,start_trace=500)
        ax0.text(0.5, 0.5, 'Validation Period', horizontalalignment='center',verticalalignment='center', fontsize=15)
    else:
        showDoublePend(fig,ax1,q_true,i,"True Model",trace=False)
        ax0.text(0.5, 0.5, 'Training Period', horizontalalignment='center',verticalalignment='center', fontsize=15)
    
        ax2.set_xlim((-2.5, 2.5))
        ax2.set_ylim((-2.5, 2.5))
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
    
        ax3.set_xlim((-2.5, 2.5))
        ax3.set_ylim((-2.5, 2.5))
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        ax4.set_xlim((-2.5, 2.5))
        ax4.set_ylim((-2.5, 2.5))
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')

        ax2.title.set_text("xL-SINDy at $\sigma=10^{-3}$")
        ax3.title.set_text("xL-SINDy at $\sigma=2\\times10^{-2}$")
        ax4.title.set_text("xL-SINDy at $\sigma=6\\times10^{-2}$")

    fig.tight_layout()
    fig.savefig(os.path.join(imdir, '{:03d}.png'.format(i)), dpi = 400)
    plt.close(fig)



for i in range(t.shape[0]):
    saveDoublePend(i)




for i in range(t.shape[0]):
    saveCartPole(i)
