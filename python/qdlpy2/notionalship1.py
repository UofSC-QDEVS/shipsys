"""
1. Ship Speed (m/s) at steady state? Normal mission, fast?   (~40 kn, or ~20 m/s)
2. Mass of the ship (kg), order of magnitude?  (10,000 Tons ~10,000 metric tons)
3. Power of the Gensets (MW) (2 units x 36 MW each)
4. Nominal DC Bus Voltage (V) ? 12 kVdc
5. Drag? or at least efficency of ship? (power lost to drag (2 x 18 MW)
6. Hotel + other loads 2 x 10 MW

"""

import liqss
from math import floor
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

   Stage 3 (Linear 7th order)

                  Rgs Lgs        Rcb Lcb              Rp  Lp
  .---.---.   .---VVV-UUU-----.--VVV-UUU-.--------.---VVV-UUU----.
  |   |   |  +|    -igs-> +   |  -icb->  |        |              |
 ( ) === < > < > Vgs    vcb1 === Ccb/2  === Ccb/2 '---VVV-UUU----+
  |   |   |   |           -   |          |            Rh  Lh     |
  '---'---'  -+-             -+-        -+-                     -+-
              '               '          '                       '

     Rph = (Rp * Rh) / (Rp + Rh)
     Req = Rgs + Rcb + Rph

     igs = Vgs / Req
     icb = igs
     ih =  igs * Rph / Rh
     ip =  igs * Rph / Rp
     vcb1 = Vgs - igs * Rgs
     vcb2 = vcb1 - icb * Rcb
             
     Vbase = 12 kV
     Sbase = 10 MVA
     
     branches:
     
     Vgs = Rgs * igs + igs'*Lgs + vcb1
     vcb1 = icb * Rcb + icb' * Lcb + vcb2
     vcb2 = ih * Rh + ih' * Lh
     vcb2 = ip * Rp + ip' * Lp
     
     igs' = (1/Lgs) * (Vgs - Rgs * igs - vcb1) 
     icb' = (1/Lcb) * (vcb1 - icb * Rcb - vcb2) 
     ih'  = (1/Lh) * (vcb2 - ih * Rh)
     ip'  = (1/Lp) * (vcb2 - ip * Rp)
     
     nodes:
     
     igs = vcb1' * Ccb/2 + icb
     icb = vcb2' * Ccb/2 + ih + ip
     
     vcb1' = (2/Ccb) * (igs - icb)
     vcb2' = (2/Ccb) * (icb - ih - ip) 


     i     Z     B     E     R     L    Q      j
     o----< >---< >---( )---VVV---UUU---/ -----o
                      --->
                       ij



                    o node i
                    |
                     / Q
     .-------.------+------.------.
     |       |      |      |      |
    < > S   < > T  ( ) H  === C  [ ] B
     |       |      |      |      |
     '-------'------+------'------'
                    |
                   -+-
                    '

"""

Vgs = 1.0
Rgs = 0.01
Lgs = 0.01
Rcb = 0.01
Lcb = 0.01
Ccb = 0.1
Rh  = 0.1
Lh  = 0.01
Rp  = 1.0
Lp  = 1.0

# steady-state:

Rph   = (Rp * Rh) / (Rp + Rh)
Req   = Rgs + Rcb + Rph
igs0  = Vgs / Req
icb0  = igs0
ih0   = igs0 * Rph / Rh
ip0   = igs0 * Rph / Rp
vcb10 = Vgs - igs0 * Rgs
vcb20 = vcb10 - icb0 * Rcb

dq0 = 0.00001

digs  = lambda: (1/Lgs) * (Vgs - Rgs * igs.q - vcb1.q) 
dicb  = lambda: (1/Lcb) * (vcb1.q - icb.q * Rcb - vcb2.q) 
dih   = lambda: (1/Lh) * (vcb2.q - ih.q * Rh)
dip   = lambda: (1/Lp) * (vcb2.q - ip.q * Rp)
dvcb1 = lambda: (2/Ccb) * (igs.q - icb.q)
dvcb2 = lambda: (2/Ccb) * (icb.q - ih.q - ip.q)  

#print((1/Lgs) * (Vgs - Rgs * igs0 - vcb10)    )
#print((1/Lcb) * (vcb10 - icb0 * Rcb - vcb20)  )
#print((1/Lh) * (vcb20 - ih0 * Rh)             )
#print((1/Lp) * (vcb20 - ip0 * Rp)             )
#print((2/Ccb) * (igs0 - icb0)                 )
#print((2/Ccb) * (icb0 - ih0 - ip0)            )

sys = liqss.Module("notional_ship1", dqmin=dq0, dqmax=dq0, dqerr=dq0)

igs  = liqss.Atom("igs",  func=digs,  x0=igs0,  units="pu", dq=dq0)
icb  = liqss.Atom("icb",  func=dicb,  x0=icb0,  units="pu", dq=dq0)
ih   = liqss.Atom("ih",   func=dih,   x0=ih0,   units="pu", dq=dq0)
ip   = liqss.Atom("ip",   func=dip,   x0=ip0,   units="pu", dq=dq0)
vcb1 = liqss.Atom("vcb1", func=dvcb1, x0=vcb10, units="pu", dq=dq0)
vcb2 = liqss.Atom("vcb2", func=dvcb2, x0=vcb20, units="pu", dq=dq0)

igs.connects(vcb1) 
icb.connects(vcb1, vcb2) 
ih.connects(vcb2) 
ip.connects(vcb2) 
vcb1.connects(igs, icb)
vcb2.connects(icb, ih, ip)

sys.add_atoms(igs, icb, ih, ip, vcb1, vcb2)

tmax = 20.0

plot_ss = True

if plot_ss:
    sys.initialize()                                                                                
    sys.run_to(tmax*0.1, verbose=True, fixed_dt=1.0e-3)
    Rp = 1.2
    sys.run_to(tmax, verbose=True, fixed_dt=1.0e-3)
    sys.save_data()
    Rp = 1.0

# qss sim:
sys.initialize()
sys.run_to(tmax*0.1, verbose=True)
Rp = 1.2
sys.run_to(tmax*0.8, verbose=True)

#d0 = lambda: 0.0
#
#for key in sys.atoms:
#
#    sys.atoms[key].func = d0

sys.run_to(tmax, verbose=True)

plot(*sys.atoms.values(), plot_updates=True, plot_ss=plot_ss)


