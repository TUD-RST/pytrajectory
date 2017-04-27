'''
This example of the double integrator demonstrates how to pass constraints to PyTrajectory.
'''
# imports
from pytrajectory import TransitionProblem
import numpy as np

# define the vectorfield
def f(x,u):
    x1, x2 = x
    u1, = u
    
    ff = [x2,
          u1]
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0]
xb = [1.0, 0.0]

# constraints dictionary
con = {1 : [-0.1, 0.65]}

# create the trajectory object
S = TransitionProblem(f, a=0.0, b=2.0, xa=xa, xb=xb, constraints=con, use_chains=False)

# start
x, u = S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xt, image):
    x = xt[0]
    
    car_width = 0.05
    car_heigth = 0.02
    
    x_car = x
    y_car = 0
    
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)
    
    image.patches.append(car)
    
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex6_ConstrainedDoubleIntegrator.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                        plotsys=[(0,'x'), (1,'dx')],
                        plotinputs=[(0,'u')])
    xmin = np.min(S.sim_data[1][:,0])
    xmax = np.max(S.sim_data[1][:,0])
    A.set_limits(xlim=(xmin - 0.1, xmax + 0.1), ylim=(-0.1,0.1))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex6_ConstrainedDoubleIntegrator.gif')
