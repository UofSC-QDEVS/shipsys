import liqss
from math import pi, sin, cos, floor
from matplotlib import pyplot as plt
import numpy as np


def plot(*atoms, plot_updates=False, plot_ss=False):

    c, j = 2, 1
    r = floor(len(atoms)/2) + 1

    for atom in atoms:
    
        plt.subplot(r, c, j)
        plt.tight_layout()
        ax1 = plt.gca()

        ax1.plot(atom.tout, atom.qout, 'b-', label="qss")

        if plot_ss:
            ax1.plot(atom.tout2, atom.qout2, 'c--', label="euler")

        if plot_updates:
            ax2 = ax1.twinx()
            ax2.plot(atom.tout, atom.nupd, 'r-')
            ax2.set_ylabel('total updates', color='r')

        plt.xlabel("t (s)")
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')

        ax1.grid()
        ax1.legend()
    
        j += 1
    
    plt.show()



"""                      
                      |
                      +--[RL]
          |           |
   (SM)---+---[CBL]---+--(IM)
          |           |
                      +--[TR]
                      |


"""

# SM:

Psm  = 25.0e6
VLL  = 4160.0
wmb  = 60.0 * pi
P    = 4.0
pf   = 0.8
rs   = 3.0e-3
Lls  = 0.2e-3
Lmq  = 2.0e-3
Lmd  = 2.0e-3
rkq  = 5.0e-3
Llkq = 0.04e-3
rfd  = 20.0e-3
Likd = 0.04e-3
rfd  = 20.0e-3
Llfd = 0.15e-3
vfdB = 90.1

# Exciter:
Kpr  = 200.0
Kir  = 0.8
Kdr  = 0.001
Tdr  = 0.001
Ka   = 1.0
Ta   = 0.0001
Vrms = 5.0

