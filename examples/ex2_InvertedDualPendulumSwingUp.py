# swing up of the inverted dual pendulum with partial linearization

# import trajectory class and necessary dependencies
from pytrajectory import ControlSystem
from sympy import cos, sin
import numpy as np

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4, x5, x6 = x  # system variables
    u, = u                      # input variable
    
    # length of the pendulums
    l1 = 0.7
    l2 = 0.5
    
    g = 9.81    # gravitational acceleration
    
    ff = np.array([         x2,
                            u,
                            x4,
                (1/l1)*(g*sin(x3)+u*cos(x3)),
                            x6,
                (1/l2)*(g*sin(x5)+u*cos(x5))
                    ])
    
    return ff

# system state boundary values for a = 0.0 [s] and b = 2.0 [s]
xa = [0.0, 0.0,  np.pi, 0.0,  np.pi, 0.0]
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# boundary values for the input
ua = [0.0]
ub = [0.0]

# create trajectory object
S = ControlSystem(f, a=0.0, b=2.0, xa=xa, xb=xb, ua=ua, ub=ub)

# alter some method parameters to increase performance
S.set_param('su', 10)
S.set_param('eps', 8e-2)

# run iteration
S.solve()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xti, image):
    x, phi1, phi2 = xti[0], xti[2], xti[4]
    
    l1 = 0.7
    l2 = 0.5

    car_width = 0.05
    car_heigth = 0.02
    pendel_size = 0.015


    x_car = x
    y_car = 0

    x_pendel1 = -l1*sin(phi1)+x_car
    y_pendel1 = l1*cos(phi1)

    x_pendel2 = -l2*sin(phi2)+x_car
    y_pendel2 = l2*cos(phi2)

    
    # pendulums
    sphere1 = mpl.patches.Circle((x_pendel1,y_pendel1),pendel_size,color='k')
    sphere2 = mpl.patches.Circle((x_pendel2,y_pendel2),pendel_size,color='0.3')
    
    # car
    car = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,fill=True,facecolor='0.75',linewidth=2.0)
    
    # joint
    joint = mpl.patches.Circle((x_car,0),0.005,color='k')
    
    # rods
    rod1 = mpl.lines.Line2D([x_car,x_pendel1],[y_car,y_pendel1],color='k',zorder=1,linewidth=2.0)
    rod2 = mpl.lines.Line2D([x_car,x_pendel2],[y_car,y_pendel2],color='0.3',zorder=1,linewidth=2.0)
    
    image.patches.append(sphere1)
    image.patches.append(sphere2)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod1)
    image.lines.append(rod2)
    
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex2_InvertedDualPendulumSwingUp.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(0,'x'),(2,'phi1'),(4,'phi2')], plotinputs=[(0,'u')])

    xmin = np.min(S.sim_data[1][:,0])
    xmax = np.max(S.sim_data[1][:,0])
    A.set_limits(xlim=(xmin - 1.0, xmax + 1.0), ylim=(-0.8,0.8))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex2_InvertedDualPendulumSwingUp.gif')
