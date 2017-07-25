# constrained double pendulum

# import all we need for solving the problem
from pytrajectory import TransitionProblem
import numpy as np
import sympy as sp
from sympy import cos, sin, Matrix
from numpy import pi


# to define a callable function that returns the vectorfield
# we first solve the motion equations of form Mx = B

def solve_motion_equations(M, B, state_vars=[], input_vars=[], parameters_values=dict()):
    '''
    Solves the motion equations given by the mass matrix and right hand side
    to define a callable function for the vector field of the respective
    control system.
    
    Parameters
    ----------
    
    M : sympy.Matrix
        A sympy.Matrix containing sympy expressions and symbols that represents
        the mass matrix of the control system.
    
    B : sympy.Matrix
        A sympy.Matrix containing sympy expressions and symbols that represents
        the right hand site of the motion equations.
    
    state_vars : list
        A list with sympy.Symbols's for each state variable.
    
    input_vars : list
        A list with sympy.Symbols's for each input variable.
    
    parameter_values : dict
        A dictionary with a key:value pair for each system parameter.
    
    Returns
    -------
    
    callable
    '''
    
    M_shape = M.shape
    B_shape = B.shape
    assert(M_shape[0] == B_shape[0])
    
    # at first we create a buffer for the string that we complete and execute 
    # to dynamically define a function and return it
    fnc_str_buffer ='''
def f(x, u):
    # System variables
    %s  # x_str
    %s  # u_str
    
    # Parameters
    %s  # par_str
    
    # Sympy Common Expressions
    %s # cse_str

    # Vectorfield
    %s  # ff_str
    
    return ff
'''

    #################################
    # handle system state variables #
    #################################
    # --> leads to x_str which shows how to unpack the state variables
    x_str = ''
    for var in state_vars:
        x_str += '%s, '%str(var)
        
    # as a last we remove the trailing '; ' to avoid syntax erros
    x_str = x_str + '= x'
    
    ##########################
    # handle input variables #
    ##########################
    # --> leads to u_str which will show how to unpack the inputs of the control system
    u_str = ''
    for var in input_vars:
        u_str += '%s, '%str(var)
    
    # after we remove the trailing '; ' to avoid syntax errors x_str will look like:
    #   'u1, u2, ... , um = u'
    u_str = u_str + '= u'
    
    ############################
    # handle system parameters #
    ############################
    # --> leads to par_str
    par_str = ''
    for k, v in parameters_values.items():
        # 'k' is the name of a system parameter such as mass or gravitational acceleration
        # 'v' is its value in SI units
        par_str += '%s = %s; '%(str(k), str(v))
    
    # as a last we remove the trailing '; ' from par_str to avoid syntax errors
    par_str = par_str[:-2]
    
    # now solve the motion equations w.r.t. the accelerations
    sol = M.solve(B)
    
    # use SymPy's Common Subexpression Elimination
    cse_list, cse_res = sp.cse(sol, symbols=sp.numbered_symbols('q'))
    
    ################################
    # handle common subexpressions #
    ################################
    # --> leads to cse_str
    cse_str = ''
    #cse_list = [(str(l), str(r)) for l, r in cse_list]
    for cse_pair in cse_list:
        cse_str += '%s = %s; '%(str(cse_pair[0]), str(cse_pair[1]))
    
    # add result of cse
    for i in xrange(M_shape[0]):
        cse_str += 'q%d_dd = %s; '%(i, str(cse_res[0][i]))
    
    cse_str = cse_str[:-2]
    
    ######################
    # create vectorfield #
    ######################
    # --> leads to ff_str
    ff_str = 'ff = ['
    
    for i in xrange(M_shape[0]):
        ff_str += '%s, '%str(state_vars[2*i+1])
        ff_str += 'q%s_dd, '%(i)
    
    # remove trailing ',' and add closing brackets
    ff_str = ff_str[:-2] + ']'
    
    ############################
    # Create callable function #
    ############################
    # now we can replace all placeholders in the function string buffer
    fnc_str = fnc_str_buffer%(x_str, u_str, par_str, cse_str, ff_str)
    # and finally execute it which will create a python function 'f'
    exec(fnc_str)
    
    # now we have defined a callable function that can be used within PyTrajectory
    return f




# system and input variables
state_vars = sp.symbols('x, dx, phi1, dphi1, phi2, dphi2')
input_vars = sp.symbols('F,')
x, dx, phi1, dphi1, phi2, dphi2 = state_vars
F, = input_vars

# parameters
l1 = 0.25                   # 1/2 * length of the pendulum 1
l2 = 0.25                   # 1/2 * length of the pendulum 
m1 = 0.1                    # mass of the pendulum 1
m2 = 0.1                    # mass of the pendulum 2
m = 1.0                     # mass of the car
g = 9.81                    # gravitational acceleration
I1 = 4.0/3.0 * m1 * l1**2   # inertia 1
I2 = 4.0/3.0 * m2 * l2**2   # inertia 2

param_values = {'l1':l1, 'l2':l2, 'm1':m1, 'm2':m2, 'm':m, 'g':g, 'I1':I1, 'I2':I2}

# mass matrix
M = Matrix([[      m+m1+m2,          (m1+2*m2)*l1*cos(phi1),   m2*l2*cos(phi2)],
            [(m1+2*m2)*l1*cos(phi1),   I1+(m1+4*m2)*l1**2,   2*m2*l1*l2*cos(phi2-phi1)],
            [  m2*l2*cos(phi2),     2*m2*l1*l2*cos(phi2-phi1),     I2+m2*l2**2]])

# and right hand site
B = Matrix([[ F + (m1+2*m2)*l1*sin(phi1)*dphi1**2 + m2*l2*sin(phi2)*dphi2**2 ],
            [ (m1+2*m2)*g*l1*sin(phi1) + 2*m2*l1*l2*sin(phi2-phi1)*dphi2**2 ],
            [ m2*g*l2*sin(phi2) + 2*m2*l1*l2*sin(phi1-phi2)*dphi1**2 ]])

f = solve_motion_equations(M, B, state_vars, input_vars)

# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0, pi, 0.0]

b = 4.0
xb = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

# here we specify the constraints for the velocity of the car
con = {'x1': [-1.0, 1.0],
       'x2': [-2.0, 2.0]}

# now we create our Trajectory object and alter some method parameters via the keyword arguments
S = TransitionProblem(f, a, b, xa, xb, ua, ub, constraints=con,
                      eps=2e-1, su=20, kx=2, use_chains=False,
                      use_std_approach=False)

# time to run the iteration
x, u = S.solve()

# the following code provides an animation of the system above
# for a more detailed explanation have a look at the 'Visualisation' section in the documentation
import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation

def draw(xt, image):
    x = xt[0]
    phi1 = xt[2]
    phi2 = xt[4]

    car_width = 0.05
    car_heigth = 0.02

    rod_length = 2.0 * 0.25
    pendulum_size = 0.015

    x_car = x
    y_car = 0

    x_pendulum1 = x_car + rod_length * sin(phi1)
    y_pendulum1 = rod_length * cos(phi1)

    x_pendulum2 = x_pendulum1 + rod_length * sin(phi2)
    y_pendulum2 = y_pendulum1 + rod_length * cos(phi2)

    # create image
    pendulum1 = mpl.patches.Circle(xy=(x_pendulum1, y_pendulum1), radius=pendulum_size, color='black')
    pendulum2 = mpl.patches.Circle(xy=(x_pendulum2, y_pendulum2), radius=pendulum_size, color='black')
    
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')
    
    rod1 = mpl.lines.Line2D([x_car,x_pendulum1], [y_car,y_pendulum1],
                            color='black', zorder=1, linewidth=2.0)
    rod2 = mpl.lines.Line2D([x_pendulum1,x_pendulum2], [y_pendulum1,y_pendulum2],
                            color='black', zorder=1, linewidth=2.0)

    # add the patches and lines to the image
    image.patches.append(pendulum1)
    image.patches.append(pendulum2)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod1)
    image.lines.append(rod2)

    # and return the image
    return image

if not 'no-pickle' in sys.argv:
    # here we save the simulation results so we don't have to run
    # the iteration again in case the following fails
    S.save(fname='ex8_ConstrainedDoublePendulum.pcl')

if 'plot' in sys.argv or 'animate' in sys.argv:
    # create Animation object
    A = Animation(drawfnc=draw, simdata=S.sim_data,
                  plotsys=[(0,'$x$'),(1,'$\\dot{x}$')], plotinputs=[(0,'$u$')])
    xmin = np.min(S.sim_data[1][:,0])
    xmax = np.max(S.sim_data[1][:,0])
    A.set_limits(xlim=(xmin - 1.0, xmax + 1.0), ylim=(-1.2,1.2))
    
if 'plot' in sys.argv:
    A.show(t=S.b)

if 'animate' in sys.argv:
    A.animate()
    A.save('ex8_ConstrainedDoublePendulum.gif')
