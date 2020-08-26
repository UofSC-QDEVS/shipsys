"""

Getting number for Cruiser level

1. Ship Speed (m/s) at steady state? Normal mission, fast?   (~40 kn, or ~20 m/s)
2. Mass of the ship (kg), order of magnitude?  (10,000 Tons ~10,000 metric tons)
3. Power of the Gensets (MW) (2 units x 36 MW each)
4. Nominal DC Bus Voltage (V) ? 12 kVdc
5. Drag? or at least efficency of ship? (power lost to drag (2 x 18 MW)
6. Hotel + other loads 2 x 10 MW

"""


import os
import pickle
from math import pi, sin, cos, atan2, sqrt, floor

import numpy as np
from matplotlib import pyplot as plt

import liqss



def plot(*atoms, plot_updates=False):

    c, j = 2, 1
    r = floor(len(atoms)/2) + 1

    for atom in atoms:
    
        plt.subplot(r, c, j)
        ax1 = plt.gca()
        ax1.plot(atom.tzoh, atom.qzoh, 'b-')
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')

        if plot_updates:
            ax2 = ax1.twinx()
            ax2.plot(atom.tout, atom.nupd, 'r-')
            ax2.set_ylabel('total updates', color='r')

        plt.xlabel("t (s)")
        ax1.grid()
    
        j += 1
    
    plt.show()


def xfmr2winding():

    """
                             N=0.1 =>  10:1 (V)  1:10 (A)

               R1   L1          1:N                   R2   L2  
          .----VVV--UUU----.            .--------.----VVV--UUU---.
          |     ---->      |            |        |      ---->    |
       + ,-.      i1      /+\ v1/N     /^\       |  +    i2      |
    E1  (   )            (   )        ( | )  C1 === v1           |
         `-'              \-/     i1/N \ /       |  -            |
          |                |            |        |               |
          '----------------'            '--------'---------------' 


    E1 = i1*R1 + di1*L1 + v1/N
    i1/N = dv1*C1 + i2
    v1 = i2*R2 + di2*L2

    di1 = 1/L1 * (E1 - i1*R1 - v1/N)
    dv1 = 1/C1 * (i1/N - i2)
    di2 = 1/L2 * (v1 - i2*R2)

    """

    E1 = 500.0
    R1 = 0.05
    L1 = 0.01
    C1 = 0.01
    R2 = 10.0
    L2 = 0.01
    N = 0.1

    dq0 = 0.01

    di1 = lambda: 1/L1 * (E1 - i1.q*R1 - v1.q/N)
    dv1 = lambda: 1/C1 * (i1.q/N - i2.q)
    di2 = lambda: 1/L2 * (v1.q - i2.q * R2)

    sys = liqss.Module("shipsys", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    i1 = liqss.Atom("i1", func=di1, units="A", dq=dq0)
    v1 = liqss.Atom("v1", func=dv1, units="V", dq=dq0)
    i2 = liqss.Atom("i2", func=di2, units="A", dq=dq0)

    i1.connects(v1)
    v1.connects(i1, i2)        
    i2.connects(v1)

    sys.add_atoms(i1, v1, i2)

    sys.initialize()
    sys.run_to(10.0, verbose=True)#, fixed_dt=1.0e-4)

    plot(*sys.atoms.values(), plot_updates=True)


def propulsion():

    """
                             
             R1   L1             
        .----VVV--UUU----.                    .--------.------------.
        |     ---->      |                    |        |            |           +
     + ,-.      i1      /+\ v2*K1            /^\      [ ]          _|_          
 Vdc  (   )            (   )          i1*K2 ( | )     [ ] D2       ___ M2      v2 (velocity)
       `-'              \-/        (thrust)  \ /      [ ] (drag)    |  (mass) 
        |                |                    |        |            |           -
        '----------------'                    +--------+------------'
                                             _|_
                                              -
    Vdc = i1*R1 + di1*L1 + v2*K1
    i1*K2 = v2*D + dv2*M

    di1 = 1/L1 * (Vdc - i1*R1 - v2*K1)
    dv2 = 1/M * (i1*K2 - v2*D)

    """

    Vdc = 500.0
    R1 = 0.01
    L1 = 0.01

    M2 = 1.0e3
    D2 = 0.1

    K1 = 1.0
    K2 = 1.0

    dq0 = 1.0

    di1 = lambda: 1/L1 * (Vdc - i1.q*R1 - v2.q*K1)
    dv2 = lambda: 1/M2 * (i1.q*K2 - v2.q*D2)

    propulsion = liqss.Module("propulsion", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    i1 = liqss.Atom("i1", func=di1, units="A",   dq=dq0)
    v2 = liqss.Atom("v2", func=dv2, units="m/s", dq=dq0)

    i1.connects(v2)
    v2.connects(i1)        

    propulsion.add_atoms(i1, v2)

    # sys = liqss.System(dqmin=dq0, dqmax=dq0, dqerr=dq0)
    # sys.Add_Modules(machine, dq2dc, propulsion)

    propulsion.initialize()
    propulsion.run_to(1.0, verbose=True)

    plot(*propulsion.atoms.values())


def xfmr3winding():

    """
             
               R1  L1       1:N1                                   
         .----VVV--UUU-----.      .--------+---------.
         |    --->         |      |        |         |      
      + ,-.    i1           ) || (         |         |      
    E1 (   )                ) || (         |         |
        `-'                 ) || (         |         |       
         |                 |      |        |         |        + 
         +-----------------'      '-.     _|_        <          
                                    |     ___ C1     <  G1   v1 
              R2   L2       1:N2    |      |         <          
         .----VVV--UUU-----.      .-'      |         |        - 
         |    --->         |      |        |         |
      + ,-.    12           ) || (         |         |
    E2 (   )                ) || (         |         |
        `-'                 ) || (         |         |
         |                 |      |        |         |
         +-----------------'      '--------+---------'
                     
               R1   L1          
          .----VVV--UUU----.      
          |     ---->      |      
       + ,-.      i1      /+\ v1/N1    (1:N1)
    E1  (   )  10 V      (   )    
         `-'              \-/                             
          |                |                 .--------.--------.
          '----------------'                 |        |        |
                                  i1/N1 +   /^\       |  +     |
                                  i2/N2    ( | )  C1 === v1   [ ] G1
               R2   L2                      \ /       |  -     |
          .----VVV--UUU----.                 |        |        |
          |     ---->      |                 '--------'--------' 
       + ,-.      i2      /+\ v1/N2 
    E2  (   ) 10 V       (   )        (1:N2)
         `-'              \-/      
          |                |       
          '----------------'       
  

    E1 = i1*R1 + di1*L1 + v1/N1
    E2 = i2*R2 + di2*L2 + v1/N2
    i1/N1 + i2/N2 = dv1*C1 + v1*G1

    di1 = 1/L1 * (E1 - i1*R1 - v1/N1)
    di2 = 1/L2 * (E2 - i2*R2 - v1/N2)
    dv1 = 1/C1 * (i1/N1 + i2/N2 - v1*G1)

    """

    E1 = 1.0
    R1 = 1.0
    L1 = 1.0e-2

    E2 = 1.0
    R2 = 1.0
    L2 = 1.0e-2

    C3 = 1.0e-2
    G3 = 1.0

    N1 = 0.1
    N2 = 0.1

    dq0 = 0.001

    di1 = lambda: 1/L1 * (E1 - i1.q*R1 - v3.q/n0.q)
    di2 = lambda: 1/L2 * (E2 - i2.q*R2 - v3.q/n0.q)
    dv3 = lambda: 1/C3 * (i1.q/n0.q + i2.q/n0.q - v3.q*G3)

    sys = liqss.Module("shipsys", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    n0 = liqss.Atom("n0", source_type=liqss.SourceType.RAMP, x1=1.0, x2=0.01, t1=0.0, t2=1.0, dq=dq0)
    i1 = liqss.Atom("i1", func=di1, units="A", dq=dq0)
    i2 = liqss.Atom("i2", func=di2, units="A", dq=dq0)
    v3 = liqss.Atom("v3", func=dv3, units="V", dq=dq0)

    i1.connects(v3, n0)
    i2.connects(v3, n0)
    v3.connects(i1, i2, n0)        

    sys.add_atoms(i1, i2, v3, n0)

    sys.initialize()
    sys.run_to(1.0, verbose=True)

    plot(*sys.atoms.values())




def system():

    # parametemrs:

    # machine:
    f = 50.0
    wb = 2*pi*f
    Ld = 7.0e-3
    Ll = 2.5067e-3
    Lm = 6.6659e-3
    LD = 8.7419e-3
    LF = 7.3835e-3
    Lq = 5.61e-3
    MQ = 4.7704e-3
    Ra = 0.001
    Rs = 1.6e-3
    RF = 9.845e-4
    RD = 0.11558
    RQ = 0.0204
    n = 1.0
    J = 2.812e4

    #Ll = 0.35*2.25/(2*pi*50)

    Xd = wb * Ld
    Ra = Rs

    # converter:
    Rf = 0.001
    Lf = 0.01
    Cf = 0.01
    Clim = 0.001
    Rlim = 100.0

    # propulsion system:
    Rp = 100.0
    Lp = 1000.0

    # avr control:
    Ka = 10.0/120.0e3
    Ta = 10.0

    Tm_max = -2.65e5
    vb = 20.0e3
    efd0 = 10.3
    efd_cmd = 9.406

    # simulation parametmrs:

    # good but slow: 1e-7 to 1e-4 err: 0.005
    dqmin = 1e-7
    dqmax = 1e-4
    dqerr = 0.005
    sr = 80
    tstop = 5.0

    # derived:

    det1 = Lm*(Lm**2-LF*Lm)+Lm*(Lm**2-LD*Lm)+(Ll+Ld)*(LD*LF-Lm**2)
    det2 = (Lq+Ll)*MQ-MQ**2
    a11 = (LD*LF-Lm**2) / det1
    a12 = (Lm**2-LD*Lm) / det1
    a13 = (Lm**2-LF*Lm) / det1
    a21 = (Lm**2-LD*Lm) / det1
    a22 = (LD*(Ll+Ld)-Lm**2) / det1
    a23 = (Lm**2-(Ll+Ld)*Lm) / det1
    a31 = (Lm**2-LF*Lm) / det1
    a32 = (Lm**2-(Ll+Ld)*Lm) / det1
    a33 = (LF*(Ll+Ld)-Lm**2) / det1
    b11 = MQ / det2
    b12 = -MQ / det2
    b21 = -MQ / det2
    b22 = (Lq+Ll) / det2

    # initial states:

    tm0 = 0.0
    fdr0 = 63.661977153494156
    fqr0 = -9.330288611488223e-11
    fF0 = 72.98578663103797
    fD0 = 65.47813807847828
    fQ0 = -5.483354402732449e-11
    wr0 = 314.1592653589793
    theta0 = -5.1145054174461215e-05
    vdc0 = -1.4784789895393902
    vt0 = 1405.8929299980925
    idc0 = -0.010300081646320524
    vf0 = -1.4784641022274498
    ip0 = -0.010300632920201253

    S = sqrt(3/2) * 2 * sqrt(3) / pi

    der_dt = 1e-3

    # algebraic functions:

    def Sd():
        return S * cos(theta.q)

    def Sq():
        return -S * sin(theta.q) 

    def efd():
        return efd0

    def id():
        return a11 * fdr.q + a12 * fF.q + a13 * fD.q 

    def iq():
        return b11 * fqr.q + b12 * fQ.q

    def iF():
        return a21 * fdr.q + a22 * fF.q + a23 * fD.q

    def iD():
        return a31 * fdr.q + a32 * fF.q + a33 * fD.q

    def iQ():
        return b21 * fqr.q + b22 * fQ.q

    def vt():
        return sqrt(Xd**2 + Ra**2) * sqrt((iq() - Sq() * idc.q)**2 + (id() - Sd() * idc.q)**2)

    def ed():
        return vb * sin(theta.q)

    def eq():
        return vb * cos(theta.q)

    def vdc():
        return (Sd() * (Ra * (id() - Sd() * idc.q) - Xd * (iq() - Sq() * idc.q))
             + Sq() * (Ra * (iq() - Sq() * idc.q) + Xd * (id() - Sd() * idc.q)))

    # ed = did * Ll - wm * Ll * iq - vd
    # eq = diq * Ll + wm * Ll * id - vq

    def vd():
        return ed() - (id() - Id.get_previous_state()) / der_dt * Ll + wr.q * Ll * iq()

    def vq():
        return eq() - (iq() - Iq.get_previous_state()) / der_dt * Ll - wr.q * Ll * id()

    def vpark():
        return sqrt(vd()*vd() + vq()*vq())

    def p():
        return iq() * vq() + id() * vd()

    def q():
        return id() * vq() - iq() * vd()

    # derivative functions:

    def didc():
        return 1/Lf * (Vdc.q - idc.q * Rf - vf.q)

    def dvf():
        return 1/Cf * (idc.q - ip.q)

    def dip():
        return 1/Lp * (vf.q - ip.q * Rp)

    def dfdr():
       return ed() - Rs * id() + wr.q * fqr.q

    def dfqr():
        return eq() - Rs * iq() - wr.q * fdr.q

    def dfF():
        return efd() - iF() * RF

    def dfD():
        return -iD() * RD

    def dfQ():
        return -iQ() * RQ

    def dwr():
        return (n/J) * (iq() * fdr.q - id() * fqr.q - tm.q)

    def dtheta():
        return wr.q - wb

    def davr():
        #return (1/Ta) * (Ka * sqrt(vd.q**2 + vq.q**2) - avr.q)
        return (1/Ta) * (Ka * vdc.q - avr.q)   #  v = i'*L + i*R    i' = (R/L)*(v/R - i)

    plot_only_mode = False
    speed_only_dq_sweep = False

    dq0 = 1e-4

    tmax = 6.0

    euler_dt = 1.0e-3

    plot_files = []

    exp0 = -5
    exp1 = -3
    npts = 3

    dq_points = np.logspace(exp0, exp1, num=npts)

    for i in range(npts):
         plot_files.append("saved_data_dq_{}_b.pickle".format(i))

    dq_points = [dq0]
    plot_files = ["test.pickle"]

    if not plot_only_mode:

        for i, dq in enumerate(dq_points):

            if speed_only_dq_sweep:
                dqmin = dq0
                dqmax = dq0
            else:
                dqmin = dq
                dqmax = dq

            ship = liqss.Module("genset", print_time=True, dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

            # machine:
            tm    = liqss.Atom("tm", source_type=liqss.SourceType.RAMP, x1=0.0, x2=Tm_max, t1=5.0, t2=20.0, dq=1e-1, units="N.m")

            fdr   = liqss.Atom("fdr",   x0=fdr0,   func=dfdr,   units="Wb",    dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            fqr   = liqss.Atom("fqr",   x0=fqr0,   func=dfqr,   units="Wb",    dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            fF    = liqss.Atom("fF",    x0=fF0,    func=dfF,    units="Wb",    dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            fD    = liqss.Atom("fD",    x0=fD0,    func=dfD,    units="Wb",    dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            fQ    = liqss.Atom("fQ",    x0=fQ0,    func=dfQ,    units="Wb",    dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            
            if not speed_only_dq_sweep:
                wr = liqss.Atom("wr",    x0=wr0,    func=dwr,    units="rad/s", dqmin=dqmin*0.1, dqmax=dqmax*0.1, dqerr=dqerr)
            else:
                wr = liqss.Atom("wr",    x0=wr0,    func=dwr,    units="rad/s", dqmin=dq, dqmax=dq, dqerr=dqerr)

            theta = liqss.Atom("theta", x0=theta0, func=dtheta, units="rad",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

            fdr.connects(fqr, fF, fD, wr, theta)
            fqr.connects(fdr, fQ, wr, theta)
            fF.connects(fdr, fD)
            fD.connects(fdr, fF)  # iD
            fQ.connects(fqr)  # iQ
            wr.connects(fqr, fdr, fF, fD, fQ, tm)
            theta.connects(wr)

            ship.add_atoms(wr, tm, fdr, fqr, fF, fD, fQ, theta)

            # algebraic atoms:

            id0 = id()
            iq0 = iq()
            Id = liqss.Atom("id", source_type=liqss.SourceType.FUNCTION, srcfunc=id, srcdt=1e-3, x0=id0, units="A", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            Iq = liqss.Atom("iq", source_type=liqss.SourceType.FUNCTION, srcfunc=iq, srcdt=1e-3, x0=iq0, units="A", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

            ship.add_atoms(Id, Iq)

            vd0 = vd()
            vq0 = vq()
            vpark0 = vpark()
            Vd = liqss.Atom("vd", source_type=liqss.SourceType.FUNCTION, srcfunc=vd, srcdt=der_dt, x0=vd0, units="V", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            Vq = liqss.Atom("vq", source_type=liqss.SourceType.FUNCTION, srcfunc=vq, srcdt=der_dt, x0=vq0, units="V", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            Vpark = liqss.Atom("vpark", source_type=liqss.SourceType.FUNCTION, srcfunc=vpark, srcdt=der_dt, x0=vpark0, units="V", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

            ship.add_atoms(Vd, Vq, Vpark)

            p0 = p()
            q0 = q()
            P = liqss.Atom("p", source_type=liqss.SourceType.FUNCTION, srcfunc=p, srcdt=der_dt, x0=p0, units="W", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
            Q = liqss.Atom("q", source_type=liqss.SourceType.FUNCTION, srcfunc=q, srcdt=der_dt, x0=q0, units="VAr", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

            ship.add_atoms(P, Q)


            # ====================== CONVERTER =================================

            """                       
               R1   L1          
          o----VVV--UUU----.      
     +          ---->      |      
                  i1      /+\ vdc/Sd    (1:N1)
    vd                   (   )               id/Sd + iq/Sq = Cdc * dvdc + vdc / Rdc
                          \-/                dvdc = 1/Cdc * (id/Sd + iq/Sq - vdc / Rdc)            
     -                     |                 .--------.--------.
          o----------------'                 |        |        |
                                  id/Sd +   /^\       |  +     |
                                  iq/Sq    ( | ) Cdc === vdc  [ ] Rdc
               R2   L2                      \ /       |  -     |
          o----VVV--UUU----.                 |        |        |
     +          ---->      |                 '--------'--------' 
                  i2      /+\ vdc/Sq 
    vq                   (   )        (1:N2)
                          \-/      
     -                     |       
          o----------------'   
          
          S = sqrt(3/2) * 2 * sqrt(3) / pi
          Sd = S * cos(theta.q)
          Sq = -S * sin(theta.q) 
          Vd = vdc/Sd
          Vq = vdc/Sq

          dvdc = 1/Cdc * (id/Sd + iq/Sq - vdc/Rdc)

          vdc = (1/s) * (1/Cdc * (id/Sd + iq/Sq - vdc/Rdc))

          vdc = (1/s) * (1/Cdc * (id/Sd + iq/Sq - vdc / Rdc)) 
  
            """

            R1 = 0.01
            L1 = 1.0e-2

            R2 = 0.01
            L2 = 1.0e-2

            C3 = 1.0e-2
            G3 = 1.0

            N1 = 1.0
            N2 = 1.0

            dq0 = 0.001

            di1 = lambda: 1/L1 * (vd() - i1.q*R1 - v3.q/N1)
            di2 = lambda: 1/L2 * (vq() - i2.q*R2 - v3.q/N2)
            dv3 = lambda: 1/C3 * (i1.q/N1 + i2.q/N2 - v3.q*G3)

            #n0 = liqss.Atom("n0", source_type=liqss.SourceType.RAMP, x1=1.0, x2=0.01, t1=0.0, t2=1.0, dq=dq0)
            
            i1 = liqss.Atom("i1", func=di1, units="A", dq=dq0)
            i2 = liqss.Atom("i2", func=di2, units="A", dq=dq0)
            v3 = liqss.Atom("v3", func=dv3, units="V", dq=dq0)

            i1.connects(v3, wr, fdr, fqr, fF, fQ, fD)
            i2.connects(v3, wr, fdr, fqr, fF, fQ, fD)

            v3.connects(i1, i2)        

            ship.add_atoms(i1, i2, v3)


            #  ===================== SIMULATION ================================

            # state space simulation:

            ship.initialize()
            ship.run_to(tmax, fixed_dt=euler_dt)
            ship.save_data()

            # qdl simulation:

            ship.initialize()
            ship.run_to(tmax, verbose=True)

            saved_data = {}

            for atom in ship.atoms.values():

                saved_data[atom.name] = {}
                saved_data[atom.name]["name"] = atom.name
                saved_data[atom.name]["units"] = atom.units
                saved_data[atom.name]["nupd"] = atom.nupd
                saved_data[atom.name]["tzoh"] = atom.tzoh
                saved_data[atom.name]["qzoh"] = atom.qzoh
                saved_data[atom.name]["tout"] = atom.tout
                saved_data[atom.name]["qout"] = atom.qout
                saved_data[atom.name]["tout2"] = atom.tout2
                saved_data[atom.name]["qout2"] = atom.qout2
                saved_data[atom.name]["error"] = atom.get_error("rpd")

            f = open(plot_files[i], "wb")
            pickle.dump(saved_data, f)
            f.close()

    def nrmsd(atom):

        """get normalized relative root mean squared error (%)
        """

        qout_interp = np.interp(saved_data[atom]["tout2"], saved_data[atom]["tout"], saved_data[atom]["qout"])

        dy_sqrd_sum = 0.0
        y_sqrd_sum = 0.0

        for q, y in zip(qout_interp, saved_data[atom]["qout2"]):
            dy_sqrd_sum += (y - q)**2
            y_sqrd_sum += y**2

        rng = max(saved_data[atom]["qout2"]) - min(saved_data[atom]["qout2"])

        if rng:
            rslt = sqrt(dy_sqrd_sum / len(qout_interp)) / rng
        else:
            rslt = 0.0

        return rslt

    time_plots = True
    time_dq_sens_plots = False
    accuracy_time_plots = False
    accuracy_agg_plots = False
    accuracy_agg_plots_per_atom = False
    speed_only_err_sens = False

    if time_plots:

        plot_file = plot_files[0]

        f = open(plot_file, "rb")
        saved_data = pickle.load(f)
        f.close()

        def plot_paper(atom, label, show_upd=True, xlim=None, ylim1=None, ylim2=None, scl=1.0, save2file=False, filename=None, order=[0, 1], multilabel="", holdstart=False, holdend=False, lstyle=None):

            if not holdend: plt.figure()

            yax1 = plt.gca()

            width = 1.5
    
            linestyles = ["c--", "b-"]
            if lstyle: linestyles = lstyle
            touts, qouts, labels = ["tout2", "tout"], ["qout2", "qout"], ["euler", "qss"]

            for idx in order:
                x = saved_data[atom][touts[idx]]
                y = [scl*v for v in saved_data[atom][qouts[idx]]]

                if holdstart or holdend:
                    yax1.plot(x, y, linestyles[idx], label="{} ({})".format(label, labels[idx]), linewidth=width)
                else:
                    yax1.plot(x, y, linestyles[idx], label=labels[idx], linewidth=width)

            if ylim1: yax1.set_ylim(*ylim1)

            if holdstart or holdend:
                yax1.set_ylabel(multilabel)
            else:
                yax1.set_ylabel(label)

            if show_upd:

                yax1.spines['left'].set_color('blue')
                yax1.tick_params(axis='y', colors='blue')
                yax1.yaxis.label.set_color('blue')
                x = saved_data[atom]["tout"]
                y = saved_data[atom]["nupd"]

                yax2 = yax1.twinx()
                yax2.plot(x, [yy*1e-4 for yy in y], 'r:', linewidth=width)

                if ylim2: yax2.set_ylim(*ylim2)

                yax2.set_ylabel("qss updates x $10^4$ (cummulative)")
                yax2.spines['right'].set_color('red')
                yax2.tick_params(axis='y', colors='red')
                yax2.yaxis.label.set_color('red')

            if xlim: plt.xlim(*xlim)
            yax1.set_xlabel("t (s)")

            if not holdstart: yax1.grid()
        
            if len(order) > 1 and not holdstart: yax1.legend()
    
            if save2file and not holdstart:
                if filename:
                    plt.savefig(filename)
                else:
                    plt.savefig("{}.pdf".format(atom))
            
            if not holdstart:
                plt.show()

        xlim = [0, 6]

        # states:

        show_upd = False
        save2file = False

       #plot_paper("fdr",   r"$\Psi_{dr} (Wb)$",     show_upd=show_upd, save2file=save2file, filename=r"plots\fdr_full_dq_1e-5.pdf",     order=[1, 0], xlim=xlim)
       #plot_paper("fqr",   r"$\Psi_{qr} (Wb)$",     show_upd=show_upd, save2file=save2file, filename=r"plots\fqr_full_dq_1e-5.pdf",     order=[1, 0], xlim=xlim)
       #plot_paper("fF",    r"$\Psi_{F} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fF_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim)
       #plot_paper("fD",    r"$\Psi_{D} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fD_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim)
       #plot_paper("fQ",    r"$\Psi_{Q} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fQ_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim)
        plot_paper("wr",    r"$\omega_{r} (rad/s)$", show_upd=show_upd, save2file=save2file, filename=r"plots\wr_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim)
       #plot_paper("theta", r"$\theta (rad)$",       show_upd=show_upd, save2file=save2file, filename=r"plots\theta _full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim)

        # derived plots without updates:

        plot_paper("vd", r"$\nu_d\:$", show_upd=False, order=[1, 0], xlim=xlim, holdstart=True, multilabel="v (V)")
        plot_paper("vq", r"$\nu_q\:$", show_upd=False, save2file=True, filename=r"plots\volts_full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim, holdend=True, multilabel="v (V)", lstyle=["y--", "r-"])
        plot_paper("id", r"$i_d\:$",   show_upd=False, order=[1, 0], xlim=xlim, holdstart=True, multilabel="i (A)")
        plot_paper("iq", r"$i_q\:$",   show_upd=False, save2file=True, filename=r"plots\currents_full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim, holdend=True, multilabel="i (A)", lstyle=["y--", "r-"])

        # converter:

        plot_paper("v3", r"$v_{conv}\:$",   show_upd=False, save2file=True, filename=r"plots\v_conv_full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim, holdend=True, multilabel="v (V)", lstyle=["y--", "r-"])


if __name__ == "__main__":

    #xfmr2winding()
    #xfmr3winding()
    #propulsion()
    system()
    