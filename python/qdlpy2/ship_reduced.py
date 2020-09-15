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
            if plot_ss:
                ax2.plot(atom.tout2, atom.nupd2, 'm--')

        plt.xlabel("t (s)")
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')

        ax1.grid()
        ax1.legend()
    
        j += 1
    
    plt.show()



"""      

  Notional System
  ===============

                      |
                      +--[RL]
          |           |
   (SM)---+---[CBL]---+--(IM)
          |           |
                      +--[TR]
                      |

  Governor
  ========

            .---------.
     wrm -->| TurbGov |--> Tm
            '---------'
        
      wrm ----.    .-->[Kp]----.     
              |    |           |
            - v    |         + v  
             (Σ)---o          (Σ)---> Tm
            + ^    |         + ^
              |    |           |
  wrm_cmd ----'    '-->[Ki/s]--'

  e  = wrm_cmd - wrm
  z' = e
  Tm = Kp * e + Ki * z

  z' = wrm_cmd - wrm
  Tm = Kp * e + Ki * z


  Exciter (Simplified IEEEAC8B)
  =============================
  
                .-------.               
             .->|  Kpr  |-----.                Vrmax
             |  '-------'     |               ,----
             |  .-------.   + v      .------'-.           .--------.
        ,-.  |  |  Kir  | +  ,-.    |    Ka   | +  ,-.    |   1    |     
 vref->( Σ )-+->| ----- |-->( Σ )-->| ------- |-->( Σ )-->| ------ |------+-> ve
        `-'  |  |   s   |    `-'    | 1+s*Ta  |    `-'    |  s*Te  |      |
       - ^   |  '-------'   + ^     '--,------'   - ^     '--------'      |
         |   |  .-------.     |   ----'             |  ,-.  + .--------.  |
        vc   |  | s*Kdr |     |   Vrmin             '-( Σ )<--| Se(ve) |<-+
             '->|-------|-----'                        `-'    '--------'  |
                |1+s*Tdr|                            +  ^     .--------.  |
                '-------'                               '-----|   Ke   |<-'
                                                              '--------'
  verr = vref - vc                                           
  vp = Kpr * verr

  dvi = Kir * verr
  dvd = (Kdr * verr / Tdr - vd) / Tdr

  vpid = vp + vi + vd

  dvr = (Ka * vpid - vr) / Ta

  verr2 = vr - vfe
            
  dve = verr2 / Te

  vfe = Ke * ve


  Cable
  =====

"""

# 

# Synchronous Machine:

Psm  = 25.0e6   # Machine power base (VA)
VLL  = 4160.0   # Nominal Terminal voltage (V)
wmb  = 60.0*pi  # Machine base speed (rad/s)
P    = 4.00     # Poles
pf   = 0.80     # Power factor
rs   = 3.00e-3  # Stator resistance (Ω)
Lls  = 0.20e-3  # Stator self inductance (H)
Lmq  = 2.00e-3  # Q-axis mutual inductance (H)
Lmd  = 2.00e-3  # D-axis mutual  inductance (H)
rkq  = 5.00e-3  # Q-axis resistance (Ω)
Llkq = 0.04e-3  #
rkd  = 5.00e-3  # D-axis resistance (Ω)
Llkd = 0.04e-3  # 
rfd  = 20.0e-3  # Field winding resistance (Ω) 
Llfd = 0.15e-3  # Field winding self inductance (H)
vfdb = 90.1     # base field voltage (V LL-RMS)

# Exciter:

Kpr  = 200.0
Kir  = 0.8
Kdr  = 0.001
Tdr  = 0.001
Ka   = 1.0
Ta   = 0.0001
Vrmx = 5.0      # (not yet used)
Vrmn = 0.0      # (not yet used)
Te   = 1.0
Ke   = 1.0
SEa  = 1.0119   # (not yet used)
SEb  = 0.0875   # (not yet used)

# turbine/governor:

Kp = 10.0e4
Ki = 10.0e4
J  = 4221.7

# derived:

Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)
Ld = Lls + (Lmd * Llfd * Llkd) / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd)

# intial conditions:

fkq0  = 0.0
fkd0  = 0.0
ffd0  = 0.0
wrm0  = wmb
th0   = 0.0
vi0   = 0.0
vd0   = 0.0
vr0   = 0.0
ve0   = 0.0

dq0 = 0.01

sys = liqss.Module("reduced_order_ship1", dqmin=dq0, dqmax=dq0, dqerr=dq0)

# SM:

iqs = 0.0
ids = 0.0
vfd = vfdb

def fq():   return Lmq / (Lmq + Llkq) * fkq.q
def fd():   return Lmd * (fkd.q / Llkd + ffd.q / Llfd) / (1 + Lmd / Llfd + Lmd / Llkd)
def vqs():  return rs * iqs + wrm.q * Ld * ids + wrm.q * fd()
def vds():  return rs * ids - wrm.q * Lq * iqs - wrm.q * fq()
def fqs():  return Lq * iqs + fq()
def fds():  return Ld * ids + fd()
def Te():   return 3.0 * P / 4.0 * (fds() * iqs - fqs() * ids)
def Tm():   return Kp * (wmb - wrm.q) + Ki * th.q

def dfkq(): return -rkq / Llkq * (fkq.q - Lq * iqs - fq() + Lls * iqs)
def dfkd(): return -rkd / Llkd * (fkd.q - Ld * ids + fd() - Lls * ids)
def dffd(): return vfd - rfd / Llfd * (ffd.q - Ld * ids + fd() - Lls * ids)
def dwrm(): return (Te() - Tm()) / J
def dth():  return wmb - wrm.q

fkq = liqss.Atom("fkq", func=dfkq, x0=fkq0, units="Wb",    dq=dq0)
fkd = liqss.Atom("fkd", func=dfkd, x0=fkd0, units="Wb",    dq=dq0)
ffd = liqss.Atom("ffd", func=dffd, x0=ffd0, units="Wb",    dq=dq0)
wrm = liqss.Atom("wrm", func=dwrm, x0=wrm0, units="rad/s", dq=dq0)
th  = liqss.Atom("th",  func=dth,  x0=th0,  units="rad",   dq=dq0)    

fkq.connects()
fkd.connects(ffd)
ffd.connects(fkd)
wrm.connects(fkq, fkd, ffd, th)
th.connects(wrm) 

sys.add_atoms(fkq, fkd, ffd, wrm, th)

#Vd = liqss.Atom("Vd", source_type=liqss.SourceType.FUNCTION, srcfunc=vds, srcdt=1e-3, units="V")  
#sys.add_atoms(Vd)

# qss sim:
sys.initialize()
sys.run_to(1.0, verbose=True, fixed_dt=1.0e-4)
iqs = 1.0
ids = 0.0
sys.run_to(2.0, verbose=True, fixed_dt=1.0e-4)

plot(*sys.atoms.values())


# Exciter
# =======

# dummy for unit tests:

#dq0 = 0.01
#
#vref = 1.0
#vc = 1.0
#
#def vp():    return Kpr * verr()
#def verr():  return vref - vc
#def vpid():  return vp() + vi.q + vd.q
#def vfe():   return Ke * ve.q
#def verr2(): return vr.q - vfe()
#
#def dvi():   return Kir * verr()
#def dvd():   return (Kdr * verr() / Tdr - vd.q) / Tdr
#def dvr():   return (Ka * vpid() - vr.q) / Ta
#def dve():   return verr2() / Te
#
#vi = liqss.Atom("vi", func=dvi, x0=vi0, units="V", dq=dq0)
#vd = liqss.Atom("vd", func=dvd, x0=vd0, units="V", dq=dq0)
#vr = liqss.Atom("vr", func=dvr, x0=vr0, units="V", dq=dq0)
#ve = liqss.Atom("ve", func=dve, x0=ve0, units="V", dq=dq0)
#
#vi.connects(ve)
#vd.connects(ve)
#vr.connects(vi, vd)
#ve.connects(vr)
#
#sys.add_atoms(vi, vd, vr, ve)
#
## euler sim:
#sys.initialize()
#vref = 1.0
#sys.run_to(0.1, verbose=True, fixed_dt=1.0e-4)
#vref = 1.05
#sys.run_to(2.0, verbose=True, fixed_dt=1.0e-4)
#vref = 1.0
#sys.run_to(4.0, verbose=True, fixed_dt=1.0e-4)
#sys.save_data()
#
## qss sim:
#sys.initialize()
#vref = 1.0
#sys.run_to(0.1, verbose=True)#, fixed_dt=1.0e-4)
#vref = 1.05
#sys.run_to(2.0, verbose=True)#, fixed_dt=1.0e-4)
#vref = 1.0
#sys.run_to(4.0, verbose=True)#, fixed_dt=1.0e-4)
#
#plot(*sys.atoms.values(), plot_updates=True, plot_ss=True)







