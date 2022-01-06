
import os
import pickle
import collections
import itertools

import numpy as np

from scipy.stats import gaussian_kde  # kernel-density estimate
from scipy.interpolate import interp1d

from scipy.fft import fft, fftfreq

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

    for i, (atom, label, color) in enumerate(atoms):

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

        upth = os.path.join(dir_, devicename + "_" + atomname + "_nupd.pickle")
        tzpth = os.path.join(dir_, devicename + "_" + atomname + "_tzoh.pickle")
        qzpth = os.path.join(dir_, devicename + "_" + atomname + "_qzoh.pickle")

        with open(tpth, "rb") as f: tout = pickle.load(f)
        with open(qpth, "rb") as f: qout = pickle.load(f)

        if ode:
            with open(topth, "rb") as f: tode = pickle.load(f)
            with open(xopth, "rb") as f: xode = pickle.load(f)

        #with open(upth, "rb") as f: upds = pickle.load(f)
        #with open(tzpth, "rb") as f: tzoh = pickle.load(f)
        #with open(qzpth, "rb") as f: qzoh = pickle.load(f)

        #tzoh = np.zeros((len(tout)*2,))
        #qzoh = np.zeros((len(tout)*2,))

        upds = list(range(len(tout)))

        if subplot:

            lbl = ""
            lblo = f"ODE (Radau)"
            lblz = f"QSS (LIQSS1)"

            if update_rate:
                lblu = f"Update Rate (Hz)"
            else:
                lblu = f"Updates"

        else:

            lbl = f"{label}"
            lblo = f"{label} (ODE-Radau)"
            lblz = f"{label} (QSS-LIQSS1)"

            if update_rate:
                lblu = f"{atom} Update Rate (Hz)"
            else:
                lblu = f"{atom} Updates"

        ax1.plot(tout, qout,
                 alpha=1.0,
                 linestyle='-',
                 color=color,
                 linewidth=0.5,
                 label=lblz)

        if ode:

            ax1.plot(tode, xode,
                     alpha=1.0,
                     linestyle='--',
                     color='black',
                     linewidth=1.0,
                     label=lblo)

        if update_rate:

            rate = np.gradient(upds, tout)

            tout2 = collections.deque(itertools.islice(tout, 2, len(tout)))
            rate2 = collections.deque(itertools.islice(rate, 2, len(tout)))

            ax2.plot(tout2, rate2,
                     alpha=1.0,
                     linestyle='dotted',
                     color=color,
                     linewidth=2.0,
                     label=lblu)
        else:
            ax2.plot(tout, upds,
                     alpha=1.0,
                     linestyle='dotted',
                     color=color,
                     linewidth=2.0,
                     label=lblu)

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

            ax1.set_ylabel(f"{label}")

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


def plot_updates(dir_, atoms, upd_bins=50, labels=None, cumm=False, title="", log=False, refdt=None):

    touts = []
    nupds = []

    for atom in atoms:

        devicename, atomname = atom.split(".")

        pth = os.path.join(dir_, devicename + "_" + atomname + "_tout.pickle")
        with open(pth, "rb") as f: touts.append(pickle.load(f))

        try:
            pth = os.path.join(dir_, devicename + "_" + atomname + "_nupd.pickle")
            with open(pth, "rb") as f: nupds.append(pickle.load(f))
        except:
            pass

    fig = plt.figure()

    ax = plt.subplot(1, 1, 1)

    lbl = f"{atom}, updates"

    if cumm and refdt:

        t0, tf = touts[0][0], touts[0][-1]

        t = np.arange(t0, tf, tf/1000.0)

        y = [p/refdt for p in t]

        if log:
            plt.semilogy(t, y, color='tab:gray', linestyle='--', label="ODE time steps")
        else:
            plt.plot(t, y, 'k--', label="ODE time steps")

    for i, atom in enumerate(atoms):

        if labels:
            label = labels[i]
        else:
            label = atom

        if cumm:

            if nupds:

                nupd = nupds[i]

            else:

                nupd = [0]

                for j, t in enumerate(touts[i]):
                    nupd.append(1 + nupd[j])

            if log:
                plt.semilogy(touts[i], nupd[:-1], label=label)
            else:
                plt.plot(touts[i], nupd[:-1], label=label)

            ax.set_ylabel("Cummulative Atom Updates")

        else:

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

            if log:
                plt.semilogy(xzoh, yzoh, label=label)
            else:
                plt.plot(xzoh, yzoh, label=label)

            ax.set_ylim(1e0, 1e7)

            ax.set_xlim(1.0, tout[-1])

            ax.set_ylabel("Update Frequency (Hz)")

    ax.set_xlabel("t (s)")

    if title:
        plt.title(title)

    plt.grid()

    plt.legend()

    plt.show()


def plot_fft(dir_, atoms, tspan, dt=1e-4):

    tstart, tstop = tspan

    for i, atom in enumerate(atoms):

        devicename, atomname = atom.split(".")

        tpth = os.path.join(dir_, devicename + "_" + atomname + "_tout.pickle")
        qpth = os.path.join(dir_, devicename + "_" + atomname + "_qout.pickle")

        with open(tpth, "rb") as f: tout = pickle.load(f)
        with open(qpth, "rb") as f: qout = pickle.load(f)

        n = int((tstop - tstart) / dt)

        print(n)
                
        f = interp1d(tout, qout)
        
        tnew = np.linspace(tstart, tstop, num=n)
        
        T = dt
        yf = fft(f(tnew))
        xf = fftfreq(n, T)[:n//2]

        plt.plot(xf[1:n], 2.0/n * np.abs(yf[1:n//2]))
        plt.grid()
        plt.show()


def freq_analysis():

    dir_ = r"D:\School\qdl\VREF_INCREASE\60s"

    plot_fft(dir_, ["sm.ids", "bus1.vd"], [20.0, 60.0])


def paper2_plots():

    dir_ = r"D:\School\qdl\VREF_INCREASE\60s"

    ode = True

    if 0:
        atoms = [
        ("avr.x1", "$x_1$ (V)", "tab:green"),
        ("avr.x2", "$x_2$ (V)", "tab:red"),
        ("avr.x3", "$x_3$ (V)", "tab:blue"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)
        
    if 0:  
        atoms = [
        ("im.wr", "$\omega_r$ (rad/s)", "tab:green"),
        ("im.ids", "$I_{ds}$ (A)", "tab:red"),
        ("bus2.vd", "$V_{ds}$ (V)", "tab:blue"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)
    if 0:
        atoms = [
        ("sm.wr", "$\omega_{r}$ (rad/s)", "tab:green"),
        ("sm.th", r"$\theta_{r}$ (rad)", "tab:red"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)
    if 0:
        atoms = [
        ("sm.iqs", "$I_{qs}$ (A)", "tab:green"),
        ("bus1.vd", "$V_{ds}$ (V)", "tab:red"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)
    if 0:
        atoms = [
        ("trload.id", "$I_{d}$ (A)", "tab:green"),
        ("trload.vdc", "$V_{dc}$ (V)", "tab:red"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)
    if 0:
        atoms = [
        ("load.id", "$I_{d}$ (A)", "tab:green"),
        ("load.iq", "$I_{q}$ (V)", "tab:red"),
        ]

        plot_qss(dir_, atoms, loc="lower right",
                 update_rate=False, subplot=True, ode=ode)

    if 1:
        plot_updates(dir_, ["im.wr", "im.ids", "bus2.vd"],
                 labels=[r"$\omega_r$", "$I_{ds}$", "$V_{ds}$"],
                 #title="Induction Machine Atom Updates",
                 cumm=True, log=True, refdt=1e-5)
    if 0:
        plot_updates(dir_, ["sm.wr", "sm.th", "sm.ids", "bus1.vd"],
                 labels=[r"$\omega_r$", r"$\theta$", "$I_{ds}$", "$V_{ds}$"],
                 #title="Synchronous Machine Atom Updates",
                 cumm=True, log=True, refdt=1e-5)
#paper2_plots()

freq_analysis()
