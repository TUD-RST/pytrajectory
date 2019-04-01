# underactuated manipulator

# import trajectory class and necessary dependencies
from pytrajectory import TransitionProblem
import numpy as np
from sympy import cos, sin


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2, x3, x4  = xx     # state variables
    u1, = uu                 # input variable
    
    e = 0.9     # inertia coupling
    
    s = sin(x3)
    c = cos(x3)
    
    ff = np.array([         x2,
                            u1,
                            x4,
                    -e*x2**2*s-(1+e*c)*u1
                    ])
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 1.8 [s]
xa = [  0.0,
        0.0,
        0.4*np.pi,
        0.0]

xb = [  0.2*np.pi,
        0.0,
        0.2*np.pi,
        0.0]

# boundary values for the inputs
ua = [0.0]
ub = [0.0]

# create trajectory object
S = TransitionProblem(f, a=0.0, b=1.8, xa=xa, xb=xb, ua=ua, ub=ub)

# also alter some method parameters to increase performance
S.set_param('su', 20)
S.set_param('kx', 3)

# run iteration
S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xti, image):
    phi1, phi2 = xti[0], xti[2]
    
    L =0.4
    
    x1 = L*cos(phi1)
    y1 = L*sin(phi1)
    
    x2 = x1+L*cos(phi2+phi1)
    y2 = y1+L*sin(phi2+phi1)
    
    # rods
    rod1 = mpl.lines.Line2D([0,x1],[0,y1],color='k',zorder=0,linewidth=2.0)
    rod2 = mpl.lines.Line2D([x1,x2],[y1,y2],color='k',zorder=0,linewidth=2.0)
    
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
    S.save(fname='ex4_UnderactuatedManipulator.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(0,'phi1'), (2,'phi2')], plotinputs=[(0,'u')])
    A.set_limits(xlim= (-0.1,0.6), ylim=(-0.4,0.65))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex4_UnderactuatedManipulator.gif')
