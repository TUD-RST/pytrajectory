# translation of the inverted pendulum

# import trajectory class and necessary dependencies
from pytrajectory import Trajectory
from sympy import sin, cos
import numpy as np

# define the function that returns the vectorfield
def f(x,u):
    x1, x2, x3, x4 = x       # system state variables
    u1, = u                  # input variable

    l = 0.5     # length of the pendulum rod
    g = 9.81    # gravitational acceleration
    M = 1.0     # mass of the cart
    m = 0.1     # mass of the pendulum

    s = sin(x3)
    c = cos(x3)

    ff = np.array([                     x2,
                   m*s*(-l*x4**2+g*c)/(M+m*s**2)+1/(M+m*s**2)*u1,
                                        x4,
            s*(-m*l*x4**2*c+g*(M+m))/(M*l+m*l*s**2)+c/(M*l+l*m*s**2)*u1
                ])
    return ff

# boundary values at the start (a = 0.0 [s])
xa = [  0.0,
        0.0,
        np.pi,
        0.0]

# boundary values at the end (b = 1.0 [s])
xb = [  0.0,
        0.0,
        0.0,
        0.0]

# create trajectory object
T = Trajectory(f, a=0.0, b=2.0, xa=xa, xb=xb)

# change method parameter to increase performance
T.setParam('use_chains', False)

# run iteration
T.startIteration()


# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
do_animation = False

if do_animation:
    import matplotlib as mpl
    from pytrajectory.utilities import Animation
    
    def draw(xti, image):
        x = xti[0]
        phi = xti[2]
        
        L = 0.5
    
        car_width = 0.05
        car_heigth = 0.02
        pendel_size = 0.015
    
        x_car = x
        y_car = 0
    
        x_pendel =-L*sin(phi)+x_car
        y_pendel = L*cos(phi)
    
        # rod
        rod = mpl.lines.Line2D([x_car,x_pendel],[y_car,y_pendel],color='k',zorder=0,linewidth=2.0)
        
        # pendulum
        sphere = mpl.patches.Circle((x_pendel,y_pendel),pendel_size,color='k')
        
        # cart
        cart = mpl.patches.Rectangle((x_car-0.5*car_width,y_car-car_heigth),car_width,car_heigth,
                                    fill=True,facecolor='0.75',linewidth=2.0)
        
        # joint
        joint = mpl.patches.Circle((x_car,0),0.005,color='k')
        
        image.lines.append(rod)
        image.patches.append(sphere)
        image.patches.append(cart)
        image.patches.append(joint)
        
        return image
    
    A = Animation(drawfnc=draw, simdata=T.sim)
    A.set_limits(xlim=(-0.3,0.8), ylim=(-0.1,0.6))
    
    A.animate()
    A.save('ex1_InvertedPendulumTranslation.mp4')
