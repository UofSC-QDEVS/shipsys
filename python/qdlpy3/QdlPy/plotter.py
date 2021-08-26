
import os
import pickle
import collections
import itertools

import numpy as np

from scipy.stats import gaussian_kde  # kernel-density estimate
from scipy.interpolate import interp1d

from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes.formatter', useoffset=False)


class MarkerUpdater:
    def __init__(self):

        self.figs = {}
        self.timer_dict = {}

    def add_ax(self, ax, features=[]):
        ax_dict = self.figs.setdefault(ax.figure,dict())
        ax_dict[ax] = {
            'xlim' : ax.get_xlim(),
            'ylim' : ax.get_ylim(),
            'figw' : ax.figure.get_figwidth(),
            'figh' : ax.figure.get_figheight(),
            'scale_s' : 1.0,
            'scale_a' : 1.0,
            'features' : [features] if isinstance(features,str) else features,
        }
        ax.figure.canvas.mpl_connect('draw_event', self.update_axes)

    def update_axes(self, event):

        for fig,axes in self.figs.items():
            if fig is event.canvas.figure:

                for ax, args in axes.items():

                    update = True

                    fw = fig.get_figwidth()
                    fh = fig.get_figheight()
                    fac1 = min(fw/args['figw'], fh/args['figh'])


                    xl = ax.get_xlim()
                    yl = ax.get_ylim()
                    fac2 = min(
                        abs(args['xlim'][1]-args['xlim'][0])/abs(xl[1]-xl[0]),
                        abs(args['ylim'][1]-args['ylim'][0])/abs(yl[1]-yl[0])
                    )

                    ##factor for marker size
                    facS = (fac1*fac2)/args['scale_s']

                    ##factor for alpha -- limited to values smaller 1.0
                    facA = min(1.0,fac1*fac2)/args['scale_a']

                    if facS != 1.0:
                        for line in ax.lines:
                            if 'size' in args['features']:
                                line.set_markersize(line.get_markersize()*facS)

                            if 'alpha' in args['features']:
                                alpha = line.get_alpha()
                                if alpha is not None:
                                    line.set_alpha(alpha*facA)


                        for path in ax.collections:
                            if 'size' in args['features']:
                                path.set_sizes([s*facS**2 for s in path.get_sizes()])

                            if 'alpha' in args['features']:
                                alpha = path.get_alpha()
                                if alpha is not None:
                                    path.set_alpha(alpha*facA)

                        args['scale_s'] *= facS
                        args['scale_a'] *= facA

                self._redraw_later(fig)


    def _redraw_later(self, fig):

        timer = fig.canvas.new_timer(interval=10)
        timer.single_shot = True
        timer.add_callback(lambda : fig.canvas.draw_idle())
        timer.start()

        if fig in self.timer_dict:
            self.timer_dict[fig].stop()

        self.timer_dict[fig] = timer


def plot_qss(dir_, atoms, ylabel=None, loc="best", method="LIQSS1",
             update_rate=True, subplot=False, ode=True):

    fig = plt.figure()

    n = len(atoms)

    if not subplot:

        plt.subplot(1, 1, 1)

        ax1 = None
        ax2 = None

        ax1 = plt.gca()

        ax2 = ax1.twinx()

        if update_rate:
            ax2.set_ylabel("QSS Update Rate (Hz)")
        else:
            ax2.set_ylabel("Cummulative QSS Updates")

    updater = MarkerUpdater()

    for i, (atom, units, color) in enumerate(atoms):

        if subplot:

            plt.subplot(n, 1, i+1)

            ax1 = None
            ax2 = None

            ax1 = plt.gca()

            ax2 = ax1.twinx()

            if update_rate:
                ax2.set_ylabel("QSS Update Rate (Hz)")
            else:
                ax2.set_ylabel("Cummulative QSS Updates")

        devicename, atomname = atom.split(".")

        tpth = os.path.join(dir_, devicename + "_" + atomname + "_tout.pickle")
        qpth = os.path.join(dir_, devicename + "_" + atomname + "_qout.pickle")

        topth = os.path.join(dir_, devicename + "_" + atomname + "_tode.pickle")
        xopth = os.path.join(dir_, devicename + "_" + atomname + "_xode.pickle")

        #upth = os.path.join(dir_, devicename + "_" + atomname + "_nupd.pickle")
        #tzpth = os.path.join(dir_, devicename + "_" + atomname + "_tzoh.pickle")
        #qzpth = os.path.join(dir_, devicename + "_" + atomname + "_qzoh.pickle")

        with open(tpth, "rb") as f: tout = pickle.load(f)
        with open(qpth, "rb") as f: qout = pickle.load(f)

        if ode:
            with open(topth, "rb") as f: tode = pickle.load(f)
            with open(xopth, "rb") as f: xode = pickle.load(f)

        #with open(upth, "rb") as f: upds = pickle.load(f)
        #with open(tzpth, "rb") as f: tzoh = pickle.load(f)
        #with open(qzpth, "rb") as f: qzoh = pickle.load(f)

        tzoh = []
        qzoh = []

        for i in range(len(tout)-1):
                
            tzoh.append(tout[i])
            tzoh.append(tout[i])
            qzoh.append(qout[i])
            qzoh.append(qout[i+1])

        upds = list(range(len(tout)))

        if subplot:

            lbl = ""
            lblo = f"ODE (Radau)"
            lblz = f"QSS (LIQSS1)"

            if update_rate:
                lblu = f"Update Rate (Hz)"
            else:
                lblu = f"Update"

        else:

            if units:
                lbl = f"{atom} ({units})"
                lblo = f"{atom} ({units}) (ODE-Radau)"
                lblz = f"{atom} ({units}) (QSS-LIQSS1)"

            else:
                lbl = f"{atom}"
                lblo = f"{atom} (ODE-Radau)"
                lblz = f"{atom} (QSS-LIQSS1)"

            if update_rate:
                lblu = f"{atom} Update Rate (Hz)"
            else:
                lblu = f"{atom} Updates"

        interp1d

        if ode:

            ax1.plot(tode, xode,
                     alpha=1.0,
                     linestyle='--',
                     color='grey',
                     linewidth=1.5,
                     label=lblo)

        ax1.plot(tzoh, qzoh,
                 alpha=1.0,
                 linestyle='-',
                 color=color,
                 linewidth=1.5,
                 label=lblz)

        if update_rate: 

            rate = np.gradient(upds, tout)

            tout2 = collections.deque(itertools.islice(tout, 2, len(tout)))
            rate2 = collections.deque(itertools.islice(rate, 2, len(tout)))

            ax2.plot(tout2, rate2,
                     alpha=1.0,
                     linestyle='dotted',
                     color=color,
                     linewidth=1.5,
                     label=lblu)
        else:
            ax2.plot(tout, upds,
                     alpha=1.0,
                     linestyle='dotted',
                     color=color,
                     linewidth=1.5,
                     label=lblu)

        """
        ax1.plot(tout, qout,
                 marker='.',
                 markersize=3,
                 markerfacecolor='k',
                 markeredgecolor='k',
                 markeredgewidth=0.5,
                 linestyle='none',
                 alpha=0.1,
                 label=lbl)

        updater.add_ax(ax1, ['size', 'alpha'])
        """

        if subplot:

            if ylabel:
                ax1.set_ylabel(ylabel) # , color=color)

            ax1.grid()

            lines1, labels1 = ax1.get_legend_handles_labels()

            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc=loc)
            else:
                ax1.legend(lines1, labels1, loc=loc)

            ax1.set_ylabel(f"{atom} ({units})")

    if not subplot:

        if ylabel:

            ax1.set_ylabel(ylabel) # , color=color)

        ax1.grid()

        lines1, labels1 = ax1.get_legend_handles_labels()

        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, loc=loc)
        else:
            ax1.legend(lines1, labels1, loc=loc)

    
    plt.xlabel("t (s)")

    plt.show()


def plot_updates(dir_, atoms, upd_bins=50):

    touts = []

    for atom in atoms:

        devicename, atomname = atom.split(".")

        pth = os.path.join(dir_, devicename + "_" + atomname + "_tout.pickle")

        with open(pth, "rb") as f: touts.append(pickle.load(f))

    fig = plt.figure()

    ax = plt.subplot(1, 1, 1)

    lbl = f"{atom}, updates"

    for i, atom in enumerate(atoms):

        if 1:

            nbins = 100

            tout = touts[i]

            tspan = tout[-1] - tout[0]

            #plt.hist(tout, bins=nbins, label=atom, histtype="step", stacked=True, log=True)

            hist, bins = np.histogram(tout, nbins)

            factor = nbins / tspan

            xzoh = []
            yzoh = []

            for i in range(nbins-1):
                
                xzoh.append(bins[i])
                xzoh.append(bins[i])
                yzoh.append(hist[i] * factor)
                yzoh.append(hist[i+1] * factor)

            plt.semilogy(xzoh, yzoh, label=atom)

            ax.set_ylim(1e0, 1e7)

            ax.set_xlim(1.0, tout[-1])

        if 0:

            upd_bins = 100

            istart = 2
            iend = len(touts[i])

            tout = collections.deque(itertools.islice(touts[i], istart, iend))

            dt = tout[-1] / upd_bins

            n = len(tout)
            bw = n**(-2/3)
            kde = gaussian_kde(tout, bw_method=bw)
            t = np.arange(tout[0], tout[-1], dt/10)
            pdensity = kde(t) * n

            plt.semilogy(t, pdensity, label=atom)
            
            ax.set_ylim(1e0, 1e7)

            ax.set_xlim(1.0, tout[-1])

    ax.set_xlabel("t (s)")

    ax.set_ylabel("Update Frequency (Hz)")

    plt.grid()

    plt.legend()

    plt.show()


"""
if __name__ == '__main__':

    my_updater = MarkerUpdater()

    ##setting up the figure
    fig, axes = plt.subplots(nrows = 2, ncols =2)#, figsize=(1,1))
    ax1,ax2,ax3,ax4 = axes.flatten()

    ## a line plot
    x1 = np.linspace(0,np.pi,30)
    y1 = np.sin(x1)
    ax1.plot(x1, y1, 'ro', markersize = 10, alpha = 0.8)
    ax3.plot(x1, y1, 'ro', markersize = 10, alpha = 1)

    ## a scatter plot
    x2 = np.random.normal(1,1,30)
    y2 = np.random.normal(1,1,30)
    ax2.scatter(x2,y2, c = 'b', s = 100, alpha = 0.6)

    ## scatter and line plot
    ax4.scatter(x2,y2, c = 'b', s = 100, alpha = 0.6)
    ax4.plot([0,0.5,1],[0,0.5,1],'ro', markersize = 10) ##note: no alpha value!

    ##setting up the updater
    my_updater.add_ax(ax1, ['size'])  ##line plot, only marker size
    my_updater.add_ax(ax2, ['size'])  ##scatter plot, only marker size
    my_updater.add_ax(ax3, ['alpha']) ##line plot, only alpha
    my_updater.add_ax(ax4, ['size', 'alpha']) ##scatter plot, marker size and alpha

    plt.show()

"""

dir_ = r"D:\School\qdl\LOAD_INCREASE\5s"

if 0:

    dir_ = r"D:\School\qdl\LOAD_INCREASE\5s"

    atoms = [
    ("sm.wr", "rad/s", "tab:red"),
    ("load.id", "A", "tab:blue"),
    ]

    plot_qss(dir_, atoms, loc="lower right", ylabel="(V)",
             update_rate=True, subplot=True, ode=True)

if 1:

    dir_ = r"D:\School\qdl\LOAD_INCREASE\30s"

    atoms = [
    ("sm.wr", "rad/s", "tab:red"),
    ("load.id", "A", "tab:blue"),
    ("im.wr", "rad/s", "tab:green"),
    ]

    plot_qss(dir_, atoms, loc="lower right", ylabel="(V)",
             update_rate=False, subplot=True, ode=False)

if 0:

    dir_ = r"D:\School\qdl\VREF_INCREASE\30s"

    atoms = [
    ("avr.x1", "V", "tab:green"),
    ("sm.wr", "rad/s", "tab:red"),
    ("load.id", "A", "tab:blue"),
    ]

    plot_qss(dir_, atoms, loc="lower right", ylabel="(V)",
             update_rate=False, subplot=True, ode=False)

if 0:

    dir_ = r"D:\School\qdl\LOAD_INCREASE\30s"

    plot_updates(dir_, ["sm.wr", "sm.th", "cable12.id", "bus1.vd", "load.id"])



