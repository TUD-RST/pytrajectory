# acrobot

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
import numpy as np
from sympy import cos, sin

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x
    u1, = u
    
    m = 1.0             # masses of the rods [m1 = m2 = m]
    l = 0.5             # lengths of the rods [l1 = l2 = l]
    
    I = 1/3.0*m*l**2    # moments of inertia [I1 = I2 = I]
    g = 9.81            # gravitational acceleration
    
    lc = l/2.0
    
    d11 = m*lc**2+m*(l**2+lc**2+2*l*lc*cos(x1))+2*I
    h1 = -m*l*lc*sin(x1)*(x2*(x2+2*x4))
    d12 = m*(lc**2+l*lc*cos(x1))+I
    phi1 = (m*lc+m*l)*g*cos(x3)+m*lc*g*cos(x1+x3)

    ff = np.array([     x2,
                        u1,
                        x4,
                -1/d11*(h1+phi1+d12*u1)
                ])
    
    return ff


# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [  0.0,
        0.0,
        3/2.0*np.pi,
        0.0]

xb = [  0.0,
        0.0,
        1/2.0*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

# create System
first_guess = {'seed' : 1529} # choose a seed which leads to quick convergence
S = ControlSystem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=ua, ub=ub, use_chains=True, first_guess=first_guess)

# alter some method parameters to increase performance
S.set_param('su', 10)

# run iteration
S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xti, image):
    phi1, phi2 = xti[0], xti[2]
    
    L=0.5
    
    x1 = L*cos(phi2)
    y1 = L*sin(phi2)
    
    x2 = x1+L*cos(phi2+phi1)
    y2 = y1+L*sin(phi2+phi1)
    
    # rods
    rod1 = mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0)
    rod2 = mpl.lines.Line2D([x1,x2],[y1,y2],color='0.3',zorder=0,linewidth=2.0)
    
    # pendulums
    sphere1 = mpl.patches.Circle((x1,y1),0.01,color='k')
    sphere2 = mpl.patches.Circle((0,0),0.01,color='k')
    
    image.lines.append(rod1)
    image.lines.append(rod2)
    image.patches.append(sphere1)
    image.patches.append(sphere2)
    
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex5_Acrobot.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(0,'phi1'),(2,'phi2')], plotinputs=[(0,'u')])
    A.set_limits(xlim=(-1.1,1.1), ylim=(-1.1,1.1))
    
if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex5_Acrobot.gif')
