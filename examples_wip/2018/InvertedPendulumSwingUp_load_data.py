"""
Load trajectories from a pickle file and plot the result (system specific)

"""


import numpy as np
from numpy import sin, cos, pi

# the next imports are necessary for the visualisatoin of the system
import sys
import matplotlib as mpl
from pytrajectory import aux
from pytrajectory.visualisation import Animation
import pickle

from ipydex import IPS


pfname = "swingup_splines.pcl"
with open(pfname, "rb") as pfile:
    cont_dict = pickle.load(pfile)
    print("Trajectories loaded from {}".format(pfname))

traj_splines = aux.unpack_splines_from_containerdict(cont_dict)
xxf, uuf = aux.get_xx_uu_funcs_from_containerdict(cont_dict)

tt = np.linspace(xxf.a, xxf.b, 1000)

xx = xxf(tt)
uu = uuf(tt)




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

sol_data = tt, xx, uu

# to plot the curves of some trajectories along with the picture
# we also pass the appropriate lists as arguments (see documentation)

A = Animation(drawfnc=draw, simdata=sol_data,
              plotsys=[(0,'x'), (2,'phi')], plotinputs=[(0,'u')])

# as for now we have to explicitly set the limits of the figure
# (may involve some trial and error)
xmin = np.min(xx[:,0]); xmax = np.max(xx[:,0])
A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-0.6,0.6))


if 'animate' in sys.argv:
    # if everything is set, we can start the animation
    # (might take some while)
    A.animate()

    # then we can save the animation as a `mp4` video file or as an animated `gif` file
    A.save('InvertedPendulum.gif')

else:
    A.show(t=tt[-1])
