'''
This example of the inverted pendulum demonstrates the basic usage of
PyTrajectory as well as its visualisation capabilities.
'''

# import all we need for solving the problem
from pytrajectory import ControlSystem
import numpy as np
from sympy import cos, sin
from numpy import pi

# the next imports are necessary for the visualisatoin of the system
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

from pytrajectory import log
log.console_handler.setLevel(10)


# first, we define the function that returns the vectorfield
def f(x, u, uref=None, t=None, pp=None):
    x1, x2, x3, x4 = x  # system variables
    u1, = u             # input variable

#x1 ... poloha ramene
#x2 ... rychlost ramene
#x3 ... poloha voziku
#x4 ... rychlost voziku
#u1 ... zrychleni voziku (vstup)


    l = 0.15    # length of the pendulum
    g = 9.81    # gravitational acceleration
    b=0.0225    # tlumeni, treci koeficient
    bcoff=0.07   #b~

    # this is the vectorfield
    ff = [          x2,
                    # g/l*sin(x1)-b/l*x2+1/l*u1*cos(x1),             #Case 1: pendulum mass concentrated in a mass point M
                    3./4*g/l*sin(x1)-bcoff*x2 + 3./4*1/l*u1*cos(x1),    #Case 2: pendulum mass is represented by a homogenous valve-shaped rod
                    x4,
                    u1]

    return ff

## I used this for debugging (opens an interactive shell on the command line)
## you can install the module with pip install ipydex
# from ipydex import IPS
# IPS()

# then we specify all boundary conditions
a = 0.0
xa = [pi, 0.0, 0.0, 0.0]

b = 2
xb = [0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

# next, this is the dictionary containing the constraints
con = { 2 : [-0.4, 0.35],3 : [-0.8, 0.5]}

# now we create our Trajectory object and alter some method parameters via the keyword arguments
#S = ControlSystem(f, a, b, xa, xb, ua, ub, constraints=con, kx=5, use_chains=False)
S = ControlSystem(f, a, b, xa, xb, ua, ub, kx=5, use_chains=False)



# now we create our Trajectory object and alter some method parameters via the keyword arguments
#S = ControlSystem(f, a, b, xa, xb, ua, ub, kx=5, use_chains=False)

# time to run the iteration
S.solve()


# now that we (hopefully) have found a solution,
# we can visualise our systems dynamic

# therefore we define a function that draws an image of the system
# according to the given simulation data
def draw(xt, image):
    # to draw the image we just need the translation `x` of the
    # cart and the deflection angle `phi` of the pendulum.
    x = xt[0]
    phi = xt[2]

    # next we set some parameters
    car_width = 0.05
    car_heigth = 0.02

    rod_length = 0.5
    pendulum_size = 0.015

    # then we determine the current state of the system
    # according to the given simulation data
    x_car = x
    y_car = 0

    x_pendulum = -rod_length * sin(phi) + x_car
    y_pendulum = rod_length * cos(phi)

    # now we can build the image

    # the pendulum will be represented by a black circle with
    # center: (x_pendulum, y_pendulum) and radius `pendulum_size
    pendulum = mpl.patches.Circle(xy=(x_pendulum, y_pendulum), radius=pendulum_size, color='black')

    # the cart will be represented by a grey rectangle with
    # lower left: (x_car - 0.5 * car_width, y_car - car_heigth)
    # width: car_width
    # height: car_height
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)

    # the joint will also be a black circle with
    # center: (x_car, 0)
    # radius: 0.005
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')

    # and the pendulum rod will just by a line connecting the cart and the pendulum
    rod = mpl.lines.Line2D([x_car,x_pendulum], [y_car,y_pendulum],
                            color='black', zorder=1, linewidth=2.0)

    # finally we add the patches and line to the image
    image.patches.append(pendulum)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod)

    # and return the image
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex0_InvertedPendulumSwingUp.pcl')

# now we can create an instance of the `Animation` class 
# with our draw function and the simulation results
#



#::start

# save the simulation data (solution of IVP) to csv

tt, xx, uu = S.sim_data
export_array = np.hstack((tt.reshape(-1, 1), xx, uu))


np.savetxt("resultZRYCHLENI.csv", export_array, delimiter=",")


# first column: time
# next n columns: state (here n = 4)
# last m columns: input (here m = 1)

# this can be used for interactively playing arround
# from IPython import embed as IPS
# IPS()


#::end

# to plot the curves of some trajectories along with the picture
# we also pass the appropriate lists as arguments (see documentation)
if 'plot' in sys.argv or 'animate' in sys.argv:
    A = Animation(drawfnc=draw, simdata=S.sim_data, 
                  plotsys=[(0,'x'), (2,'phi')], plotinputs=[(0,'u')])

    # as for now we have to explicitly set the limits of the figure
    # (may involves some trial and error)
    xmin = np.min(S.sim_data[1][:,0]); xmax = np.max(S.sim_data[1][:,0])
    A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-0.6,0.6))

if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    # if everything is set, we can start the animation
    # (might take some while)
    A.animate()

    # then we can save the animation as a `mp4` video file or as an animated `gif` file
    A.save('ex0_InvertedPendulum.gif')

