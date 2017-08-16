"""
This example of the inverted pendulum demonstrates how to handle possible state constraints.
"""

# import all we need for solving the problem
from pytrajectory import TransitionProblem
import numpy as np
from sympy import cos, sin

from pytrajectory import log
log.console_handler.setLevel(30)


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2, x3, x4 = xx  # system variables
    u1, = uu             # input variable
    
    l = 0.5     # length of the pendulum
    g = 9.81    # gravitational acceleration
    
    # this is the vectorfield
    ff = [          x2,
                    u1,
                    x4,
            (1/l)*(g*sin(x3)+u1*cos(x3))]
    
    return ff

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, np.pi, 0.0]

b = 3.0
xb = [0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

# next, this is the dictionary containing the constraints
con = { 'x1': [-0.8, 0.3],
        'x2': [-2.0, 2.0] }

first_guess = {'seed': 50}
# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = TransitionProblem(f, a, b, xa, xb, ua, ub, constraints=con, kx=2, eps=5e-2,
                      first_guess=first_guess, use_chains=False, sol_steps=1300,
                      reltol=2e-5)

# time to run the iteration
S.solve(tcpport=5006)
# S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation


def draw(xt, image):
    x = xt[0]
    phi = xt[2]
    
    car_width = 0.05
    car_heigth = 0.02
    
    rod_length = 0.5
    pendulum_size = 0.015
    
    x_car = x
    y_car = 0
    
    x_pendulum = -rod_length * sin(phi) + x_car
    y_pendulum = rod_length * cos(phi)
    
    pendulum = mpl.patches.Circle(xy=(x_pendulum, y_pendulum), radius=pendulum_size, color='black')
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')
    rod = mpl.lines.Line2D([x_car,x_pendulum], [y_car,y_pendulum],
                            color='black', zorder=1, linewidth=2.0)
    
    image.patches.append(pendulum)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod)
    
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex7_ConstrainedInvertedPendulum.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                        plotsys=[(0,'x'), (1,'dx')],
                        plotinputs=[(0,'u1')])
    xmin = np.min(S.sim_data[1][:,0])
    xmax = np.max(S.sim_data[1][:,0])
    A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-0.6,0.6))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex7_ConstrainedInvertedPendulum.gif')
    
