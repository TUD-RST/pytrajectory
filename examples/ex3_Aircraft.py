# vertical take-off aircraft

# import trajectory class and necessary dependencies
import sys
from pytrajectory import TransitionProblem, log
from sympy import sin, cos
import numpy as np
from numpy import pi


if "log" in sys.argv:
    log.console_handler.setLevel(10)


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2, x3, x4, x5, x6 = xx  # system state variables
    u1, u2 = uu                  # input variables
    
    # coordinates for the points in which the engines engage [m]
    l = 1.0
    h = 0.1

    g = 9.81    # graviational acceleration [m/s^2]
    M = 50.0    # mass of the aircraft [kg]
    J = 25.0    # moment of inertia about M [kg*m^2]

    alpha = 5/360.0*2*pi    # deflection of the engines

    sa = sin(alpha)
    ca = cos(alpha)

    s = sin(x5)
    c = cos(x5)
    
    ff = np.array([             x2,
                    -s/M*(u1+u2) + c/M*(u1-u2)*sa,
                                x4,
                    -g+c/M*(u1+u2) +s/M*(u1-u2)*sa ,
                                x6,
                    1/J*(u1-u2)*(l*ca+h*sa)])
    
    return ff 

# system state boundary values for a = 0.0 [s] and b = 3.0 [s]
xa = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
xb = [10.0, 0.0, 5.0, 0.0, 0.0, 0.0]

# boundary values for the inputs
ua = [0.5*9.81*50.0/(cos(5/360.0*2*pi)), 0.5*9.81*50.0/(cos(5/360.0*2*pi))]
ub = [0.5*9.81*50.0/(cos(5/360.0*2*pi)), 0.5*9.81*50.0/(cos(5/360.0*2*pi))]

# create trajectory object
S = TransitionProblem(f, a=0.0, b=3.0, xa=xa, xb=xb, ua=ua, ub=ub)

# don't take advantage of the system structure (integrator chains)
# (this will result in a faster solution here)
S.set_param('use_chains', False)

# also alter some other method parameters to increase performance
S.set_param('kx', 5)

# run iteration
S.solve()

# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xti, image):
    x, y, theta = xti[0], xti[2], xti[4]
    
    S = np.array( [   [0,     0.3],
                      [-0.1,  0.1],
                      [-0.7,  0],
                      [-0.1,  -0.05],
                      [ 0,    -0.1],
                      [0.1,   -0.05],
                      [ 0.7,  0],
                      [ 0.1,  0.1]])

    xx = S[:,0].copy()
    yy = S[:,1].copy()

    S[:,0] = xx*cos(theta)-yy*sin(theta)+x
    S[:,1] = yy*cos(theta)+xx*sin(theta)+y

    aircraft = mpl.patches.Polygon(S, closed=True, facecolor='0.75')
    image.patches.append(aircraft)
    
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex3_Aircraft.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(4,'theta')], plotinputs=[(0,'F1'),(1,'F2')])
    A.set_limits(xlim=(-1,11), ylim=(-1,7))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex3_Aircraft.gif')
