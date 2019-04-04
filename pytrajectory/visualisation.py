# IMPORTS
import numpy as np
import matplotlib as mpl
#mpl.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import os

from IPython import embed as IPS

class Animation():
    """
    Provides animation capabilities.

    Given a callable function that draws an image of the system state and smiulation data
    this class provides a method to created an animated representation of the system.


    Parameters
    ----------

    drawfnc : callable
        Function that returns an image of the current system state according to :attr:`simdata`

    simdata : numpy.ndarray
        Array that contains simulation data (time, system states, input states)

    plotsys : list
        List of tuples with indices and labels of system variables that will be plotted along the picture

    plotinputs : list
        List of tuples with indices and labels of input variables that will be plotted along the picture
    """

    def __init__(self, drawfnc, simdata, plotsys=[], plotinputs=[], rcParams=None):

        if rcParams:
            # apply provided matplotlib options
            mpl.rcParams.update(rcParams)
        self.fig = plt.figure()

        self.image = 0

        self.t = simdata[0]
        self.xt = simdata[1]
        self.ut = simdata[2]

        if np.ndim(self.ut) == 1:
            self.ut = self.ut.reshape(-1, 1)  # one column

        self.plotsys = plotsys
        self.plotinputs = plotinputs

        self.get_axes()

        self.axes['ax_img'].set_frame_on(True)
        self.axes['ax_img'].set_aspect('equal')
        # self.axes['ax_img'].set_axis_bgcolor('w')
        self.axes['ax_img'].set_facecolor('w')

        self.nframes = int(round(24*(self.t[-1] - self.t[0])))

        self.draw = drawfnc

        # set axis limits and labels of system curves
        xlim = (self.t[0], self.t[-1])
        for i, idxlabel in enumerate(self.plotsys):
            idx, label = idxlabel

            try:
                ylim = (min(self.xt[:, idx]), max(self.xt[:, idx]))
            except:
                ylim = (min(self.xt), max(self.xt))

            self.set_limits(ax='ax_x%d' % i, xlim=xlim, ylim=ylim)
            self.set_label(ax='ax_x%d' % i, label=label)

        # set axis limits and labels of input curves
        for i, idxlabel in enumerate(self.plotinputs):
            idx, label = idxlabel

            try:
                ylim = (min(self.ut[:, idx]), max(self.ut[:, idx]))
            except:
                ylim = (min(self.ut), max(self.ut))

            self.set_limits(ax='ax_u%d' % i, xlim=xlim, ylim=ylim)
            self.set_label(ax='ax_u%d' % i, label=label)

        # enable LaTeX text rendering --> slow
        plt.rc('text', usetex=True)

    class Image():
        """
        This is just a container for the drawn system.
        """
        def __init__(self):
            self.patches = []
            self.lines = []

        def reset(self):
            self.patches = []
            self.lines = []

    def get_axes(self):
        sys = self.plotsys
        inputs = self.plotinputs

        if not sys+inputs:
            gs = GridSpec(1,1)
        else:
            l = len(sys+inputs)

            gs = GridSpec(l, 3)

        axes = dict()
        syscurves = []
        inputcurves = []

        if not sys+inputs:
            axes['ax_img'] = self.fig.add_subplot(gs[:,:])
        else:
            axes['ax_img'] = self.fig.add_subplot(gs[:,1:])

        for i in range(len(sys)):
            axes['ax_x%d'%i] = self.fig.add_subplot(gs[i,0])

            curve = mpl.lines.Line2D([], [], color='black')
            syscurves.append(curve)

            axes['ax_x%d'%i].add_line(curve)

        lensys = len(sys)
        for i in range(len(inputs)):
            axes['ax_u%d'%i] = self.fig.add_subplot(gs[lensys+i,0])

            curve = mpl.lines.Line2D([], [], color='black')
            inputcurves.append(curve)

            axes['ax_u%d'%i].add_line(curve)

        self.axes = axes
        self.syscurves = syscurves
        self.inputcurves = inputcurves

    def set_limits(self, ax='ax_img', xlim=(0,1), ylim=(0,1)):
        self.axes[ax].set_xlim(*xlim)
        self.axes[ax].set_ylim(*ylim)

    def set_label(self, ax='ax_img', label=''):
        self.axes[ax].set_ylabel(label, rotation='horizontal', horizontalalignment='right')

    def show(self, t=0.0, xlim=None, ylim=None, axes_callback=None, save_fname=None, show=True):
        """
        Plots one frame of the system animation.

        Parameters
        ----------

        t : float
            The time for which to plot the system
        """

        # determine index of sim_data values corresponding to given time
        if t <= self.t[0]:
            i = 0
        elif t >= self.t[-1]:
            i = -1
        else:
            i = 0
            while self.t[i] < t:
                i += 1

        # draw picture
        image = self.image
        ax_img = self.axes['ax_img']

        if image == 0:
            # init
            image = self.Image()
        else:
            # update
            for p in image.patches:
                p.remove()
            for l in image.lines:
                l.remove()
            image.reset()

        # call the provided drawfnc
        image = self.draw(self.xt[i, :], image=image)

        for p in image.patches:
            ax_img.add_patch(p)

        for l in image.lines:
            ax_img.add_line(l)

        self.image = image
        self.axes['ax_img'] = ax_img

        if xlim is not None and ylim is not None:
            self.set_limits(ax='ax_img', xlim=xlim, ylim=ylim)

        # update system curves
        for k, curve in enumerate(self.syscurves):
            try:
                curve.set_data(self.t[:i], self.xt[:i, self.plotsys[k][0]])
            except:
                assert False  # TODO: this should not happen (unclear index handling)
                curve.set_data(self.t[:i], self.xt[:i])
            self.axes['ax_x%d'%k].add_line(curve)

        # update input curves
        for k, curve in enumerate(self.inputcurves):
            try:
                curve.set_data(self.t[:i], self.ut[:i, self.plotinputs[k][0]])
            except Exception as e:
                assert False  # TODO: this should not happen (unclear index handling)
                # TODO: better exception handling
                curve.set_data(self.t[:i], self.ut[:i])
            self.axes['ax_u%d' % k].add_line(curve)


        if axes_callback:
            # this is the possibility for each example to modify the axes (like xticks etc)
            axes_callback(self)

        plt.draw()

        if save_fname:
            plt.savefig(save_fname)

        if show:
            plt.show()

    def animate(self):
        """
        Starts the animation of the system.
        """
        t = self.t
        xt = self.xt
        ut = self.ut

        # NEW: try to repeat first and last frame
        pause_time = 1.0 #[s]

        # how many frames will be plotted per second of system time
        fps = self.nframes/(t[-1] - t[0])

        # add so many frames that they fill the `pause`
        add_frames = int(fps * pause_time)

        if len(ut.shape) == 1:
            ut = ut.reshape(-1, 1)

        for i in range(add_frames):
            t = np.hstack((t[0],t,t[-1]))
            xt = np.vstack((xt[0],xt,xt[-1]))
            ut = np.vstack((ut[0],ut,ut[-1]))

        # array of indices corresponding to the shown frames
        f_idcs = np.linspace(0,xt.shape[0]-1, self.nframes,endpoint=True)
        f_idcs = np.hstack(([f_idcs[0]]*add_frames, f_idcs, [f_idcs[-1]]*add_frames))
        # convert the floats to integer
        f_idcs = np.int32(f_idcs)

        self.T = t[-1] - t[0] + 2 * pause_time

        # raise number of frames
        self.nframes += 2 * add_frames

        def _animate(frame_nbr):
            idx = f_idcs[frame_nbr]
            out = "frame = {f}, t = {t}, x = {x}, u = {u}"
            print(out.format(f=frame_nbr, t=t[idx], x=xt[idx, :], u=ut[idx, :]))

            # draw picture
            image = self.image
            ax_img = self.axes['ax_img']

            if image == 0:
                # init
                image = self.Image()
            else:
                # update
                for p in image.patches:
                    p.remove()
                for l in image.lines:
                    l.remove()
                image.reset()

            image = self.draw(xt[idx,:], image=image)

            for p in image.patches:
                ax_img.add_patch(p)

            for l in image.lines:
                ax_img.add_line(l)

            self.image = image
            self.axes['ax_img'] = ax_img

            # update system curves
            for k, curve in enumerate(self.syscurves):
                try:
                    curve.set_data(t[:idx], xt[:idx, self.plotsys[k][0]])
                except:
                    curve.set_data(t[:idx], xt[:idx])
                self.axes['ax_x%d'%k].add_line(curve)

            # update input curves
            for k, curve in enumerate(self.inputcurves):
                try:
                    curve.set_data(t[:idx], ut[:idx,self.plotinputs[k][0]])
                except:
                    curve.set_data(t[:idx], ut[:idx])
                self.axes['ax_u%d'%k].add_line(curve)

            plt.draw()

        self.anim = animation.FuncAnimation(self.fig, _animate, frames=self.nframes,
                                            interval=1, blit=False)


    def save(self, fname, fps=None, dpi=200):
        """
        Saves the animation as a video file or animated gif.
        """

        if not fps:
            fps = self.nframes/(float(self.T))  # add pause_time here?

        if fname.endswith('gif'):
            self.anim.save(fname, writer='imagemagick', fps=fps)
        else:
            FFWriter = animation.FFMpegFileWriter()
            self.anim.save(fname, fps=fps, dpi=dpi, writer='mencoder')
        print(("File written: {}".format(fname)))
