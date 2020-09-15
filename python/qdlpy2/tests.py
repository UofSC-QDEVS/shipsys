
import os
from math import pi, sin, cos, atan2, sqrt
from cmath import rect
import liqss
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pickle


RAD_PER_SEC_2_RPM = 9.5492965964254

save_data = None

def simple():

    sys = liqss.Module("simple")

    node1 = liqss.Atom("node1", 1.0, 1.0, 1.0, dq=1e-2)
    branch1 = liqss.Atom("branch1", 1.0, 1.0, 1.0, dq=1e-2)

    node1.connect(branch1, -1.0)
    branch1.connect(node1, 1.0)

    sys.add_atoms(node1, branch1)

    sys.initialize()
    sys.run_to(10.0)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(node1.tzoh, node1.qzoh, 'b-')
    plt.plot(node1.tout, node1.qout, 'k.')
    plt.subplot(2, 1, 2)
    plt.plot(branch1.tzoh, branch1.qzoh, 'b-')
    plt.plot(branch1.tout, branch1.qout, 'k.')
    plt.show()


def stiffline():

    dq = 0.01

    sys = liqss.Module("stiffline", dqmin=dq, dqmax=dq)

    node1 = liqss.Atom("node1", 1.0, 1.0, 1.0)
    node2 = liqss.Atom("node2", 1.0e3, 1.0, 1.0)
    node3 = liqss.Atom("node3", 1.0e6, 1.0, 1.0)
    branch1 = liqss.Atom("branch1", 1.0, 1.0, 1.0)
    branch2 = liqss.Atom("branch2", 1.0, 1.0, 1.0)

    node1.connect(branch1, -1.0)
    node2.connect(branch2, 1.0)
    node3.connect(branch2, -1.0)

    branch1.connect(node1, 1.0)
    branch2.connect(node2, -1.0)
    branch2.connect(node3, 1.0)

    sys.add_atoms(node1, node2, node3, branch1, branch2)

    sys.initialize()
    sys.run_to(1.0e5)

    plt.figure()

    plt.subplot(3, 2, 1)
    plt.plot(node1.tzoh, node1.qzoh, 'b-')
    plt.plot(node1.tout, node1.qout, 'k.')

    plt.subplot(3, 2, 2)
    plt.plot(branch1.tzoh, branch1.qzoh, 'b-')
    plt.plot(branch1.tout, branch1.qout, 'k.')

    plt.subplot(3, 2, 3)
    plt.plot(node2.tzoh, node2.qzoh, 'b-')
    plt.plot(node2.tout, node2.qout, 'k.')
    
    plt.subplot(3, 2, 4)
    plt.plot(branch2.tzoh, branch2.qzoh, 'b-')
    plt.plot(branch2.tout, branch2.qout, 'k.')
    
    plt.subplot(3, 2, 5)
    plt.plot(node3.tzoh, node3.qzoh, 'b-')
    plt.plot(node3.tout, node3.qout, 'k.')
    plt.show()


def delegates():

    dqmin = 0.01
    dqmax = 0.01
    dqerr = 0.01

    sys = liqss.Module("delegates", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    def f1():
        return 1.0 - sys.node1.q + sys.branch1.q

    def f2():
        return 1.0 - sys.branch1.q - sys.node1.q

    node1 = liqss.Atom("node1", func=f1)
    branch1 = liqss.Atom("branch1", func=f2)

    node1.connect(branch1)
    branch1.connect(node1)

    sys.add_atoms(node1, branch1)

    sys.initialize()
    sys.run_to(10.0)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(node1.tzoh, node1.qzoh, 'b-')
    plt.plot(node1.tout, node1.qout, 'k.')
    plt.subplot(2, 1, 2)
    plt.plot(branch1.tzoh, branch1.qzoh, 'b-')
    plt.plot(branch1.tout, branch1.qout, 'k.')
    plt.show()


def genset():

    # parametmrs:

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

    Tm_max = -265000.0  # 25% rated
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

    #def didc():
    #    return 1/Lf * (Vdc.q - idc.q * Rf - vf.q)

    #def dvf():
    #    return 1/Cf * (idc.q - ip.q)

    #def dip():
    #    return 1/Lp * (vf.q - ip.q * Rp)

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

    #def davr():
    #    #return (1/Ta) * (Ka * sqrt(vd.q**2 + vq.q**2) - avr.q)
    #    return (1/Ta) * (Ka * vdc.q - avr.q)   #  v = i'*L + i*R    i' = (R/L)*(v/R - i)

    plot_only_mode = True
    speed_only_dq_sweep = False

    dq0 = 1.0e-4

    tmax = 40.0

    euler_dt = 1.0e-4

    plot_files = []

    exp0 = -5
    exp1 = -3
    npts = 3

    #dq_points = np.logspace(exp0, exp1, num=npts)

    #for i in range(npts):
    #     plot_files.append("saved_data_dq_{}_b.pickle".format(i))
   
    # for fixed dq:
    dq_points = [1.0e-4]
    plot_files.append("saved_data_dq_{}.pickle".format(dq_points[0]))

    # for zoom plots:
    #plot_files = ["test.pickle"]

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
            tm    = liqss.Atom("tm", source_type=liqss.SourceType.RAMP, x1=0.0, x2=Tm_max, t1=15.0, t2=20.0, dq=1e-1, units="N.m")

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

            font_size = 14

            plt.rc('font', size=font_size)          # controls default text sizes
            #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

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
        
            #if len(order) > 1 and not holdstart: yax1.legend()
    
            plt.tight_layout()

            if save2file and not holdstart:
                if filename:
                    plt.savefig(filename, bbox_inches='tight')
                else:
                    plt.savefig("{}.pdf".format(atom), bbox_inches='tight')

            #if not holdstart:
            #    plt.show()

            if not holdstart:
                plt.close()


        xlim = [0, 40]

        # states:

        show_upd = True
        save2file = True

        ylim2 = [-50, 250]
        plot_paper("fdr",   r"$\Psi_{dr} (Wb)$",     show_upd=show_upd, save2file=save2file, filename=r"plots\fdr_full_dq_1e-5.pdf",     order=[1, 0], xlim=xlim, ylim2=ylim2)
        plot_paper("fqr",   r"$\Psi_{qr} (Wb)$",     show_upd=show_upd, save2file=save2file, filename=r"plots\fqr_full_dq_1e-5.pdf",     order=[1, 0], xlim=xlim, ylim2=ylim2)
        plot_paper("fF",    r"$\Psi_{F} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fF_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim, ylim2=ylim2)
        plot_paper("fD",    r"$\Psi_{D} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fD_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim, ylim2=ylim2)
        plot_paper("fQ",    r"$\Psi_{Q} (Wb)$",      show_upd=show_upd, save2file=save2file, filename=r"plots\fQ_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim, ylim2=ylim2)
        plot_paper("wr",    r"$\omega_{r} (rad/s)$", show_upd=show_upd, save2file=save2file, filename=r"plots\wr_full_dq_1e-5.pdf",      order=[1, 0], xlim=xlim)
        plot_paper("theta", r"$\theta (rad)$",       show_upd=show_upd, save2file=save2file, filename=r"plots\theta _full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim)

        # derived plots without updates:

        plot_paper("vd", r"$\nu_d\:$", show_upd=False, order=[1, 0], xlim=xlim, holdstart=True, multilabel="v (V)")
        plot_paper("vq", r"$\nu_q\:$", show_upd=False, save2file=True, filename=r"plots\volts_full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim, holdend=True, multilabel="v (V)", lstyle=["y--", "r-"])
        plot_paper("id", r"$i_d\:$",   show_upd=False, order=[1, 0], xlim=xlim, holdstart=True, multilabel="i (A)")
        plot_paper("iq", r"$i_q\:$",   show_upd=False, save2file=True, filename=r"plots\currents_full_dq_1e-5.pdf",  order=[1, 0], xlim=xlim, holdend=True, multilabel="i (A)", lstyle=["y--", "r-"])

        print("Done creating plots.")

    if time_dq_sens_plots:

        plt.figure()

        colors = ["red", "blue", "green"]

        f = open(plot_files[0], "rb")
        saved_data = pickle.load(f)
        f.close()

        x = [t+14 for t in saved_data["wr"]["tout2"]]
        y = saved_data["wr"]["qout2"]

        plt.plot(x, y, 'k--', label="euler")

        for i, plot_file in enumerate(plot_files):

            if i > 0:
                f = open(plot_file, "rb")
                saved_data = pickle.load(f)
                f.close()

            x = [t+14 for t in saved_data["wr"]["tzoh"]]
            y = saved_data["wr"]["qzoh"]

            label = "qss ($\Delta Q\:=$ {})".format(dq_points[i])

            plt.plot(x, y, color=colors[i], label=label)

        plt.xlabel("t (s)")
        #plt.xlim([14.0, 16.0])
        plt.ylabel("$\omega_{r}$ (rad/s)")
        plt.legend()
        plt.grid()
        plt.show()

    if speed_only_err_sens:

        global save_data
        
        plt.figure()

        yax1 = plt.gca()
        yax2 = yax1.twinx()

        x = dq_points

        yerrors = []
        yupdates = []

        for plot_file in plot_files:   # dq dimension

            f = open(plot_file, "rb")
            saved_data = pickle.load(f)
            f.close()

            yerrors.append(nrmsd("wr"))
            yupdates.append(saved_data["wr"]["nupd"][-1])

        yax1.loglog(x, yerrors, "b.-", label="$\omega_{r}\:$ atom relative error")
        yax2.loglog(x, yupdates, "r.--", label="$\omega_{r}\:$ atom total updates")

        yax1.set_ylabel("Error (%)")
        yax1.spines['left'].set_color('blue')
        yax1.tick_params(axis='y', colors='blue')
        yax1.yaxis.label.set_color('blue')

        yax2.set_ylabel("Updates")
        yax2.spines['right'].set_color('red')
        yax2.tick_params(axis='y', colors='red')
        yax2.yaxis.label.set_color('red')

        lines, labels = yax1.get_legend_handles_labels()
        lines2, labels2 = yax2.get_legend_handles_labels()
        yax2.legend(lines + lines2, labels + labels2, loc=0)

        yax1.set_xlabel(r"$\Delta Q$")

        plt.xlim([1e-7, 1e-2])

        plt.show()

    if accuracy_time_plots:

        plt.figure()

        j = 0

        styles = ["solid", "dashed", "dotted"]
        widths = [1, 1, 3]
        colors = ["blue", "red", "green"]

        for i in [19, 14, 0]:    # -6, ~-3, -2

            dq_point = dq_points[i]

            plot_file = plot_files[i]

            f = open(plot_file, "rb")
            saved_data = pickle.load(f)
            f.close()

            x = saved_data["fdr"]["tout2"]
            y = saved_data["fdr"]["error"]

            lbl = r"$\phi_{dr}\:\Delta Q\:=" + " {:4.2e}".format(dq_point) + "$"

            plt.plot(x, y, label=lbl, linestyle=styles[j], linewidth=widths[j], color=colors[j])
            j += 1

        plt.grid()
        plt.ylabel("Error (%)")
        plt.xlabel("t (s)")
        plt.xlim([1.0, 1.5])
        plt.ylim([-0.1, 1.1])
        plt.legend(loc="upper left")
        plt.show()

    if accuracy_agg_plots:

        plt.figure()

        yax1 = plt.gca()
        yax2 = yax1.twinx()

        x = dq_points[:9]

        atoms = ["fdr", "fqr", "fF", "fD", "fQ", "wr", "theta"]

        yerror = []

        yupdates = []

        for plot_file in plot_files[:9]:   # dq dimension

            f = open(plot_file, "rb")
            saved_data = pickle.load(f)
            f.close()

            atom_error_norms = []
            atom_updates = []

            for atom in atoms:  # atom dimension

                atom_error_norms.append(nrmsd(atom))
                atom_updates.append(saved_data[atom]["nupd"][-1])

            yerror.append(max(atom_error_norms))
            yupdates.append(sum(atom_updates))

        yax1.loglog(x, yerror, "b.-", label="Maximum Error")
        yax2.loglog(x, yupdates, "r.--", label="Total updates")

        yax1.set_ylabel("Error (%)")
        yax1.spines['left'].set_color('blue')
        yax1.tick_params(axis='y', colors='blue')
        yax1.yaxis.label.set_color('blue')

        yax2.set_ylabel("Updates")
        yax2.spines['right'].set_color('red')
        yax2.tick_params(axis='y', colors='red')
        yax2.yaxis.label.set_color('red')

        lines, labels = yax1.get_legend_handles_labels()
        lines2, labels2 = yax2.get_legend_handles_labels()
        yax2.legend(lines + lines2, labels + labels2, loc=0)

        yax1.set_xlabel(r"$\Delta Q$")
        plt.show()

    if accuracy_agg_plots_per_atom:

        plt.figure()

        yax1 = plt.gca()
        yax2 = yax1.twinx()

        x = dq_points

        atoms = ["fdr", "fqr", "fF", "fD", "fQ", "wr", "theta"]

        yerror = []

        yupdates = []

        atom_error_norms = {}
        atom_updates = {}

        for plot_file in plot_files[:11]:   # dq dimension

            f = open(plot_file, "rb")
            saved_data = pickle.load(f)
            f.close()

            for atom in atoms:  # atom dimension

                if not atom in atom_error_norms: atom_error_norms[atom] = []
                if not atom in atom_updates: atom_updates[atom] = []

                atom_error_norms[atom].append(nrmsd(atom))
                atom_updates[atom].append(saved_data[atom]["nupd"][-1])

        for atom in atoms:
            yax1.loglog(x, atom_error_norms[atom], linestyle="solid", label="{} error".format(atom))
            yax2.loglog(x, atom_updates[atom], linestyle="dashed", label="{} updates".format(atom))

        yax1.set_ylabel("Error (%)")
        yax1.spines['left'].set_color('blue')
        yax1.tick_params(axis='y', colors='blue')
        yax1.yaxis.label.set_color('blue')

        yax2.set_ylabel("Updates")
        yax2.spines['right'].set_color('red')
        yax2.tick_params(axis='y', colors='red')
        yax2.yaxis.label.set_color('red')

        lines, labels = yax1.get_legend_handles_labels()
        lines2, labels2 = yax2.get_legend_handles_labels()
        yax2.legend(lines + lines2, labels + labels2, loc=0)


        plt.show()
            

def shipsys3():
        
    # machine parameters:

    f = 60.0
    Ld = 7.0e-3
    Lq = 5.61e-3
    Rs = 1.6e-3
    P = 2.0
    J = 2.812e4
    Tm0 = -2.65e5
    Efd = 20.0e3
    Cf = 0.001
    Clim = 0.001
    Lf = 0.001
    Rf = 0.001
    Lp = 10.0
    Rp = 1.0

    # simulation parameters:

    dqmin = 1e-7
    dqmax = 1e-2
    dqerr = 0.005
    sr = 80
    tstop = 0.1

    # initial states:

    vd0 = 0.0  
    vq0 = 0.0  
    idc0 = 0.0 
    vdc0 = 0.0  
    ip0 = 0.0 
    id0 = 0.0 
    iq0 = 0.0 
    wm0 = 377.0  
    theta0 = 0.0     

    S = sqrt(3/2) * 2 * sqrt(3) / pi
    wb = 2*pi*f

    def Sd():
        return S * cos(theta.q)

    def Sq():
        return S * sin(theta.q)

    # derivative functions:
 
    def dvd():
        return 1/Clim * (id.q - idc.q * Sd()) 

    def dvq():
        return 1/Clim * (iq.q - idc.q * Sq())
 
    def didc():
        return 1/Lf * (vd.q * Sd() + vq.q * Sq() - idc.q * Rf - vdc.q)
 
    def dvdc():
        return 1/Cf * (idc.q - ip.q) 
 
    def dip():
        return 1/Lp * (vdc.q - ip.q * Rp)
 
    def did():
        return 1/Ld * (Efd * sin(theta.q) - id.q * Rs - wm.q * iq.q * Lq - vd.q)
 
    def diq():
        return 1/Lq * (Efd * cos(theta.q) - iq.q * Rs + wm.q * id.q * Ld - vq.q)
 
    def dwm():
        return P/(2*J) * (3/2*P/2 * (id.q * Ld * iq.q - iq.q * Lq * id.q) - Tm0)
 
    def dtheta():
        return wm.q - wb

    ship = liqss.Module("ship", print_time=True)

    #tm = liqss.Atom("tm", source_type=liqss.SourceType.RAMP, x1=0.0, x2=Tm0, t1=0.1, t2=5.1, dq=1e4, units="N.m")
    #tm = liqss.Atom("tm", source_type=liqss.SourceType.CONSTANT, x1=0.0, units="N.m", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    
    vd    = liqss.Atom("vd",    x0=vd0   , func=dvd   , units="V"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)           
    vq    = liqss.Atom("vq",    x0=vq0   , func=dvq   , units="V"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    idc   = liqss.Atom("idc",   x0=id0   , func=didc  , units="A"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    vdc   = liqss.Atom("vdc",   x0=vd0   , func=dvdc  , units="V"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    ip    = liqss.Atom("ip",    x0=ip0   , func=dip   , units="A"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    id    = liqss.Atom("id",    x0=id0   , func=did   , units="A"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    iq    = liqss.Atom("iq",    x0=iq0   , func=diq   , units="A"    , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    wm    = liqss.Atom("wm",    x0=wm0   , func=dwm   , units="rad/s", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    theta = liqss.Atom("theta", x0=theta0, func=dtheta, units="rad"  , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    vd.connects(id, iq)   
    vq.connects(id, iq)    
    idc.connects(vd, vq, id, iq)   
    vdc.connects(id, iq, ip)    
    ip.connects(vdc, ip)    
    id.connects(vd, vq, id, iq)    
    iq.connects(vd, vq, id, iq)    
    wm.connects(id, iq)    
    theta.connects(wm) 

    ship.add_atoms(vd, vq, idc, vdc, ip, id, iq, wm, theta)

    # simulation:

    # no-load startup:

    ship.initialize()
    #ship.run_to(tstop)
    ship.run_to(tstop, fixed_dt=1e-6)

    r = 5
    c = 2
    i = 0

    plt.figure()

    for i, (name, atom) in enumerate(ship.atoms.items()):

        plt.subplot(r, c, i+1)
        plt.plot(atom.tzoh, atom.qzoh, 'b-')
        #plt.plot(atom.tout, atom.qout, 'k.')
        plt.xlabel("t (s)")
        plt.ylabel(name + " (" + atom.units + ")")

    plt.show()


def shipsys2():
        
    # machine parameters:

    Prate = 555.0e6 # V.A
    Vrate = 24.0e3  # V_LLRMS
    freq = 60.0     # Hz
    P = 2 # number of poles

    wb = 2*pi*freq  # base speed

    vb = 4160.0 # base voltage RMS LL

    Lad = 1.66
    Laq = 1.61
    Lo = 0.15
    Ll = 0.15
    Ra = 0.003
    Lfd = 0.165
    Rfd = 0.0006
    L1d = 0.1713
    R1d = 0.0284
    L1q = 0.7252
    R1q = 0.00619
    L2q = 0.125
    R2q = 0.2368

    J = 2.525 # 27548.0

    # derived:
    Lffd = Lad + Lfd
    Lf1d = Lffd - Lfd
    L11d = Lf1d + L1d
    L11q = Laq + L1q
    L22q = Laq + L2q

    # simulation parameters:
    dqmin = 1e-7
    dqmax = 1e-2
    dqerr = 0.005
    sr = 80
    tstop = 0.1

    # initial states:

    efd0 = 92.95
    fd0 = 0.0
    fq0 = 0.0
    fo0 = 0.0
    ffd0 = 0.0
    f1d0 = 0.0
    f1q0 = 0.0
    f2q0 = 0.0
    wr0 = wb
    theta0 = 0.0

    # algebraic equations:

    def id():
        return -(((Lad*Lf1d-L11d*Lad) * ffd.q + (L11d*Lffd-Lf1d**2) * fd.q + (Lad*Lf1d-Lad*Lffd) * f1d.q)
                 / ((L11d*Lffd-Lf1d**2)*Ll+(L11d*Lad-Lad**2)*Lffd-Lad*Lf1d**2+2*Lad**2*Lf1d-L11d*Lad**2))

    def iq():
        return -(((Laq**2-L11q*L22q)*fq.q + (L11q*Laq-Laq**2)*f2q.q + (L22q*Laq-Laq**2) * f1q.q)
                 / ((Laq**2-L11q*L22q)*Ll-Laq**3+(L22q+L11q)*Laq**2-L11q*L22q*Laq))

    def io():
        return -fo.q / Lo

    def ifd():
        return (((L11d*Ll-Lad**2+L11d*Lad) * ffd.q + (Lad*Lf1d-L11d*Lad) * fd.q + (-Lf1d*Ll-Lad*Lf1d+Lad**2) * f1d.q)
                /((L11d*Lffd-Lf1d**2)*Ll+(L11d*Lad-Lad**2)*Lffd-Lad*Lf1d**2+2*Lad**2*Lf1d-L11d*Lad**2))

    def i1d():
        return -(((Lf1d*Ll+Lad*Lf1d-Lad**2) * ffd.q + (Lad*Lffd-Lad*Lf1d) * fd.q + (-Lffd*Ll-Lad*Lffd+Lad**2) * f1d.q)
                 / ((L11d*Lffd-Lf1d**2)*Ll+(L11d*Lad-Lad**2)*Lffd-Lad*Lf1d**2+2*Lad**2*Lf1d-L11d*Lad**2))

    def i1q():
        return -(((Laq**2-L22q*Laq) * fq.q - (Laq*Ll) * f2q.q + (L22q*Ll-Laq**2+L22q*Laq) * f1q.q)
                 / ((Laq**2-L11q*L22q)*Ll-Laq**3+(L22q+L11q)*Laq**2-L11q*L22q*Laq))

    def i2q():
        return -(((Laq**2-L11q*Laq) * fq.q + (L11q*Ll-Laq**2+L11q*Laq) * f2q.q - (Laq*Ll) * f1q.q)
                 /((Laq**2-L11q*L22q)*Ll-Laq**3+(L22q+L11q)*Laq**2-L11q*L22q*Laq)) 

    def Te():
        return fd.q * iq() + fq.q * id()

    def ed():
        return vb * cos(theta.q)

    def eq():
        return -vb * sin(theta.q)

    def eo():
        return 0

    def efd():
        return efd0

    def tm():
        return 0

    # derivative functions:
 
    def dfd():
        # ed = 1/wb * dfd - fq*wr - Ra * id
        return wb * (ed() + fq.q * wr.q + Ra * id())

    def dfq():
        # eq = 1/wb * dfq + fd*wr - Ra * iq
        return wb * (eq() + fd.q * wr.q + Ra * iq())

    def dfo():
        # eo = 1/wb * dfo - Ra * io
        return wb * (eo() + Ra * io())

    def dffd():
        # efd = 1/wb * dffd - Rfd * ifd
        return wb * (efd() + Rfd * ifd())

    def df1d():
        # 0 = 1/wb * df1d - R1d * i1d
        return wb * R1d * i1d()

    def df1q():
        # 0 = 1/wb * df1q - R1q * i1q
        return wb * R1q * i1q()

    def df2q():
        # 0 = 1/wb * df2q - R2q * i2q
        return wb * R2q * i2q()

    def dwr():
        return P/(2*J) * (Te() - tm())
 
    def dtheta():
        return wr.q - wb 

    ship = liqss.Module("ship", print_time=True)

    #tm = liqss.Atom("tm", source_type=liqss.SourceType.RAMP, x1=0.0, x2=Tm0, t1=0.1, t2=5.1, dq=1e4, units="N.m")

    #tm = liqss.Atom("tm", source_type=liqss.SourceType.CONSTANT, x1=0.0, units="N.m", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
              
    fd     = liqss.Atom("fd"   , x0=fd0   , func=dfd   , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    fq     = liqss.Atom("fq"   , x0=fq0   , func=dfq   , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    fo     = liqss.Atom("fo"   , x0=fo0   , func=dfo   , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    ffd    = liqss.Atom("ffd"  , x0=ffd0  , func=dffd  , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    f1d    = liqss.Atom("f1d"  , x0=f1d0  , func=df1d  , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    f1q    = liqss.Atom("f1q"  , x0=f1q0  , func=df1q  , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    f2q    = liqss.Atom("f2q"  , x0=f2q0  , func=df2q  , units="Wb"   , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    wr     = liqss.Atom("wr"   , x0=wr0   , func=dwr   , units="rad/s", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    theta  = liqss.Atom("theta", x0=theta0, func=dtheta, units="rad"  , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    fd.connects(fd, f1d, ffd, fq, wr)
    fq.connects(fq, f1q, f2q, fd, wr)
    fo.connects(fo)
    ffd.connects(fd, f1d, ffd)
    f1d.connects(fd, f1d, ffd)
    f1q.connects(fq, f1q, f2q)
    f2q.connects(fq, f1q, f2q)
    wr.connects(fd, fq, f1d, ffd, f1q, f2q, wr)
    theta.connects(wr)

    ship.add_atoms(
      fd,    
      fq,    
      fo,    
      ffd,   
      f1d,   
      f1q,   
      f2q,   
      wr,    
      theta)

    # simulation:

    ship.initialize()
    #ship.run_to(tstop)
    ship.run_to(0.0003, fixed_dt=1e-8)

    r = 5
    c = 2
    i = 0

    plt.figure()

    for i, (name, atom) in enumerate(ship.atoms.items()):

        plt.subplot(r, c, i+1)
        plt.plot(atom.tzoh, atom.qzoh, 'b-')
        #plt.plot(atom.tout, atom.qout, 'k.')
        plt.xlabel("t (s)")
        plt.ylabel(name + " (" + atom.units + ")")

    plt.show()


def gencls():

    """ Simplified Sync Machine Model simulation

    
    """

    # parameters:

    H = 3.0
    Kd = 1.0
    fs = 60.0
    ws = 2*pi*fs


    # intial conditions:

    Tm0 = 5.5


    # odes:

    def dtheta():
        return (wr.q - ws) / ws

    # model and state stoms:

    gencls = liqss.Module("gencls", print_time=True)

    theta  = liqss.Atom("theta", x0=theta0, func=dtheta, units="rad"  , dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    theta.connects()

    ship.add_atoms(theta)

    ship.initialize()

    ship.run_to(0.0003, fixed_dt=1e-8)

    r = 1
    c = 2
    i = 0

    plt.figure()

    for i, (name, atom) in enumerate(ship.atoms.items()):

        plt.subplot(r, c, i+1)
        plt.plot(atom.tzoh, atom.qzoh, 'b-')
        plt.plot(atom.tout, atom.qout, 'k.')
        plt.xlabel("t (s)")
        plt.ylabel(name + " (" + atom.units + ")")

    plt.show()


def modulate(times, values, freq, npoints):

    mags = [abs(x) for x in values]
    phs = [atan2(x.imag, x.real) for x in values]

    fmag = interp1d(times, mags, kind='zero')
    fph = interp1d(times, phs, kind='zero')

    times2 = np.linspace(times[0], times[-1], npoints)
    mags2 = fmag(times2)
    phs2 = fph(times2)

    values2 = [mag * sin(2.0*pi*freq*t + ph) for (t, mag, ph) in zip(times2, mags2, phs2)]

    return times2, values2


def plot_cqss(*catoms, npoints=1000):

    nplots = len(catoms)

    plt.figure()

    for iplot, catom in enumerate(catoms):

        ax = plt.subplot(nplots, 1, iplot+1)

        l1 = plt.plot(catom.tzoh, [abs(x) for x in catom.qzoh], 'c-', color='lightblue', linewidth=1)
        l2 = plt.plot(catom.tout, [abs(x) for x in catom.qout], 'b.', color="darkblue", markersize=3)
        tmod, qmod = modulate(catom.tout, catom.qout, catom.freq1, npoints)
        l3 = plt.plot(tmod, qmod, 'b-', color='grey', linewidth=0.5, label="x(t) (modulated)")
        plt.xlabel("t(s)")

        plt.ylabel(catom.name + " (" + catom.units + ")", color='blue')

        if 1:
            ax2 = ax.twinx()
            l4 = ax2.plot(catom.tout, catom.nupd, 'c-', color='red', linewidth=1, label="cummulative updates")
            ax2.set_ylabel("updates", color='red')
            labels = ["|X(t)| qss (zoh)", "|X(t)| qss", "x(t) (modulated)", "cummulative updates"]
            plt.legend(handles=[l1[0], l2[0], l3[0], l4[0]], labels=labels, loc='lower right')

        labels = ["|X(t)| qss (zoh)", "|X(t)| qss", "x(t) (modulated)"]
        plt.legend(handles=[l1[0], l2[0], l3[0]], labels=labels, loc='lower right')

        ax.spines['left'].set_color('blue')
        ax.tick_params(axis='y', colors='blue', which='both')

        if 1:
            ax2.spines['right'].set_color('red')
            ax2.tick_params(axis='y', colors='red', which='both')

    plt.show()


def dynphasor():

    """                     
             I1                            I2
            --->     n1                   --->         n1           
     .---VVV---UUU---o-------.-------.-----VVV---UUU---o-------.-------.     
    +|   R1    L1    |       |       |  +  R2    L2    |       |       |     +
 E1 (~)          C1 ===  G1 [ ]  H1 (v) V1         C2 ===  G2 [ ]  H2 (v)    V2
    -|               |       |       |  -              |       |       |     -
     '---------------'-------'-------+-----------------'-------'-------'
                                    _|_
                                     -

    E1 = I1'*L1 + I1*(R1 + jwL1) + V1
    I1' = (1/L1) * (E1 - I1*(R1 + jwL1) - V1)

    I1 = V1'*C1 + V1*(G1 + 1/jwC1) + H1 + I2


    E1 = I2'*L2 + I2*(R2 + jwL2) + V2 - V1


    I2 = V2'*C2 + V2*(G2 + 1/jwC2) + H2

    """

    dqmin = 0.03
    dqmax = 0.03
    dqerr = 0.01

    f = 60.0
    omega = 2.0*pi*f

    E1 = rect(10000.0, 0.0)
    H1 = rect(0.0, 0.0)
    R1 = 0.001
    L1 = 0.0001
    C1 = 0.001
    G1 = 0.0

    E2 = rect(0.0, 0.0)
    H2 = rect(100.0, 0.0)
    R2 = 0.001
    L2 = 0.0001
    C2 = 0.001
    G2 = 0.01

    jwL1 = complex(0.0, omega*L1)
    jwC1 = complex(0.0, omega*C1)
    jwL2 = complex(0.0, omega*L2)
    jwC2 = complex(0.0, omega*C2)

    sys = liqss.Module("dynphasor", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)


    #I1' = (1/L1) * (E1 - I1*(R1 + jwL1)) - V1 
    #V1' = (1/C1) * (I1 - V2*(G1 + 1/jwC1)) - H1 - I2
    #I2' = (1/L2) * I*(R2 + jwL2) - V2 
    #V2' = (1/C2) * (I2 - V2*(G2 + 1/jwC2)) - H2
    
    def dI1():
        return (1/L1) * (E1 - branch1.q*(R1 + jwL1) - node1.q) 

    def dV1():
        return (1/C1) * (branch1.q - node1.q*(G1 + 1/jwC1) - H1 - branch2.q)

    def dI2():
        return (1/L2) * (E2 - branch2.q*(R2 + jwL2) - node2.q + node1.q) 

    def dV2():
        return (1/C2) * (branch2.q - node2.q*(G2 + 1/jwC2) - H2)

    branch1 = liqss.ComplexAtom("branch1", units="A", func=dI1, dq=10.0, freq1=f)
    node1 =   liqss.ComplexAtom("node1",   units="V", func=dV1, dq=1.0,  freq1=f)
    branch2 = liqss.ComplexAtom("branch2", units="A", func=dI2, dq=10.0, freq1=f)
    node2 =   liqss.ComplexAtom("node2",   units="V", func=dV2, dq=1.0,  freq1=f)
    
    branch1.connect(node1)
    node1.connect(branch1, branch2)
    branch2.connect(node1, node2)
    node2.connect(branch2)

    sys.add_atoms(node1, branch1, node2, branch2)

    sys.initialize()

    tmax = 0.01
    
    sys.run_to(tmax*0.5, verbose=True)

    E1 = 2.0 * E1 

    sys.run_to(tmax, verbose=True)

    plot_cqss(node1, branch1, node2, branch2, npoints=int(20*f*tmax))


def genphasor():

    # simulation parameters:

    dqmin = 0.03
    dqmax = 0.03
    dqerr = 0.01

    # per unit bases:

    sbase = 100.0
    fbase = 60.0

    # parameters (per unit):

    Ra    = 0.001
    Tdop  = 5.000
    Tdopp = 0.060	
    Tqop  = 0.200	
    Tqopp = 0.060	
    H     = 3.000	
    D     = 0.000	
    Xd    = 1.600	
    Xq    = 1.550	
    Xdp   = 0.700	
    Xqp   = 0.850	
    Xdpp  = 0.350	
    Xqpp  = 0.350
    Xl    = 0.200	
    
    Pmech = 100.0 / sbase

    # derived: 
          
    omega0 = 2.0 * pi * fbase
    G = Ra / (Xdpp**2 + Ra**2)
    B = -Xdpp / (Xdpp**2 + Ra**2)

    # initial conditions:

    delta0 = 0.0
    eqpp0  = 0.0
    edpp0  = 0.0
    eqp0   = 0.0
    edp0   = 0.0

    # algebraic functions:

    def Telec():
       return edpp.q*iq() - eqpp.q*id()

    def id():
        return ((Ra * edpp.q + Xdpp * eqpp.q) * omega.q) / ((Xdpp**2 + Ra**2) * omega0)

    def iq():
        return ((Ra * eqpp.q - Xdpp * edpp.q) * omega.q) / ((Xdpp**2 + Ra**2) * omega0)

    def theta():
        return delta.q

    def Isource():
        return complex(cos(delta.q) * id - sin(delta.q) * iq, 
                       cos(delta.q) * iq + sin(delta.q) * id)

    def efd():
        return 1.0

    # derivative functions:

    def ddelta():
        return omega.q * omega0

    def domega():
        return 1/(2*H) * ((Pmech - D*omega.q)/(1 + omega.q) - Telec())

    def deqpp():
        return (edp.q - edpp.q + (Xqp - Xl) * iq()) / Tqopp

    def dedpp():
        return (eqp.q - eqpp.q - (Xdp - Xl) * id()) / Tdopp

    def deqp():
        return (efd() - (Xd-Xdpp)/(Xdp-Xdpp) * eqp.q + (Xd-Xdp)/(Xdp-Xdpp) * eqpp.q) / Tdop

    def dedp():
        return (-(Xq-Xqpp)/(Xqp-Xqpp) * edp.q + (Xq-Xqp)/(Xqp-Xqpp) * edpp.q) / Tqop

    # system object:

    sys = liqss.Module("genphasor", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    # atoms:

    delta = liqss.Atom("delta", x0=delta0, func=ddelta, units="rad",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    omega = liqss.Atom("omega", x0=omega0, func=domega, units="rad/s", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    eqpp  = liqss.Atom("eqpp",  x0=eqpp0,  func=deqpp,  units="vpu",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    edpp  = liqss.Atom("edpp",  x0=edpp0,  func=dedpp,  units="vpu",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    eqp   = liqss.Atom("eqp",   x0=eqp0,   func=deqp,   units="vpu",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)
    edp   = liqss.Atom("edp",   x0=edp0,   func=dedp,   units="vpu",   dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    sys.add_atoms(delta, omega, eqpp, edpp, eqp, edp)

    sys.initialize()
    
    sys.run_to(0.1, verbose=True, fixed_dt=1.0e-6)

    #sys.run_to(2.0, verbose=True, fixed_dt=1.0e-4)

    # plot all results:

    r = 3
    c = 2
    i = 0

    plt.figure()

    f = open(r"c:\temp\initcond.txt", "w")

    j = 0

    for i, (name, atom) in enumerate(sys.atoms.items()):

        plt.subplot(r, c, j+1)
        plt.plot(atom.tzoh, atom.qzoh, 'b-')
        #plt.plot(atom.tout, atom.qout, 'k.')
        plt.xlabel("t (s)")
        plt.ylabel(name + " (" + atom.units + ")")
        j += 1

        f.write("\t{}0 = {}\n".format(name, atom.q))

    f.close()
    plt.show()


def shipsys3():

    """
    Machine model:
                   t_
                   /
    dw = 1/(2*H) * | [(Tm - Te) - K*dw] dt
                  _/
                   0
    w = w0 + dw

            .----------+----------+-------.
            |          |          |       |
           ,-.    1   _|_    1    >      / \
       Tm ( ^ )  ---  ___   ---   >     ( v ) Te
           `-'   2*H   |    Kd    >      \ /    
            |          |          |       |
            '----------+----------+-------'


                  Ra  Xd"          1:Sd      Rf   Lf
            .----VVV--UUU---o----.      .----VVV--UUU---+-----o
            |    --->            |      |      --->     |
         + ,-.    Id        +     > || <       idc      |
       Ed (   )            Vd     > || <                |
           `-'              -     > || <                |
            |                    |      |               |      +
            +---------------o ---'      '-.             |
                                          |        Cf  ===    vdc
                 Ra   Xq"          1:Sq   |             |
            .----VVV--UUU---o----.      .-'             |      -
            |    --->            |      |               |
         + ,-.    Iq        +     > || <                |
       Eq (   )            Vq     > || <                |
           `-'              -     > || <                |
            |                    |      |               |
            +---------------o ---'      '---------------+-----o

    Te = Ed * Id - Eq * Iq
    Tm = Pm / w
    S  = sqrt(3/2) * 2 * sqrt(3) / pi
    Sd = S * cos(th)
    Sq = S * sin(th)

    dw = 1/(2*H) * (Pm - D*w - Te)

    dth = w0 - w

    """

    # SI Params:

    vnom  = 315.0e3     # V RMS line to line
    sbase = 1000.0e6    # VA
    fbase = 60.0        # Hz
    npole = 2           # NA
    wbase = 2*pi*fbase  # rad.s^-1
    Jm = 168870         # kg.m^2
    Kd = 64.3           # ?
    R  = 99.225*0.02     # ohm
    L  = 99.225*0.2/(wbase)  # H

    # pu params:

    vbase = vnom * sqrt(2) / sqrt(3)
    zbase = vbase**2/sbase
    ibase = sbase / vnom * sqrt(2) / sqrt(3)
    Ra = R / zbase
    Xdpp = L * wbase / zbase
    Xqpp = L * wbase / zbase
    H = Jm * wbase**2 / (2*sbase*npole)

    id0 = -50.43858979029197
    iq0 = -29.184373989238278
    th0 = -8.894597172185733
    wm0 = -1.7132557189700638e-12

    Pm0 = 0.5
    efd0 = 1.0149 * vbase

    th_plot = []
    wm_plot = []
    id_plot = []
    iq_plot = []
    Pe_plot = []

    th = th0
    wm = wm0
    id = id0
    iq = iq0

    t = 0.0
    time = []
    dt = 0.001
    npts = 1e4

    for i in range(1, int(npts)):

        if i > npts / 2: Pm0 = 1.0

        # algebra:

        ed = efd0 / vbase * sin(th)
        eq = efd0 / vbase * cos(th)
        vd = 1.0
        vq = 0.0
        Pe = 3/2 * (ed*id - eq*iq)

        # plot arrays:

        th_plot.append(th)
        wm_plot.append(wm)
        id_plot.append(id)
        iq_plot.append(iq)
        Pe_plot.append(Pe)

        # diff equ:

        id += (1/Xdpp * (ed - id * Ra - vd)) * dt
        iq += (1/Xqpp * (eq - iq * Ra - vq)) * dt
        wm += (1/(2*H) * (Pm0 - Kd*wm - Pe)) * dt
        th += wm * wbase * dt

        time.append(t)
        t += dt

    print("id0 = {}".format(id_plot[-1]))
    print("iq0 = {}".format(iq_plot[-1]))
    print("th0 = {}".format(th_plot[-1]))
    print("wm0 = {}".format(wm_plot[-1]))

    plt.subplot(2, 1, 1)
    plt.plot(time, [x*sbase/1e6 for x in Pe_plot], label="Pe (MW)")
    plt.plot(time, [(1.0-x)*wbase*30/pi/npole for x in wm_plot], label="Speed (rad/s)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, id_plot, label="id")
    plt.plot(time, iq_plot, label="iq")
    plt.legend()
    plt.show()


def shipsys4():

    """
    Machine model:
                   t_
                   /
    dw = 1/(2*H) * | [(Tm - Te) - K*dw] dt
                  _/
                   0
    w = w0 + dw

         .----------+----------+-------.             
         |          |          |       |       
        ,-.    1   _|_    1    >      / \      
    Tm ( ^ )  ---  ___   ---   >     ( v ) Te  
        `-'   2*H   |    Kd    >      \ /      
         |          |          |       |       
         '----------+----------+-------'       


               Ra  Xd"          1:Sd         Rf   Lf                             
         .----VVV--UUU----+---.      .-------VVV--UUU---+-----------------.
         |    --->        |   |      |         --->     |      --->       |
      + ,-.    Id     +  _|_   ) || (          idc      |       ip        |
    Ed (   )         Vd  ___   ) || (                   |                 |
        `-'           -   |    ) || (                   |                < 
         |                |   |      |     +            |      +         <  Rp
         +----------------+---'      '-.               _|_               < 
                                       |  vdc      Cf  ___    vf          |
              Ra   Xq"          1:Sq   |                |                (
         .----VVV--UUU----+---.      .-'   -            |      -         (  Lp
         |    --->        |   |      |                  |                (
      + ,-.    Iq     +  _|_   ) || (                   |                 |
    Eq (   )         Vq  ___   ) || (                   |                 |
        `-'           -   |    ) || (                   |                 |
         |                |   |      |                  |                 |
         +----------------+---'      '------------------+-----------------'


                        
              Ra  Xd"                   1:S                                 Rf   Lf             Rp   Ms
         .----VVV--UUU------------.                 .----------+------------VVV--UUU----+-------VVV--UUU----|-
         |              --->      |                 |          |  ic ->            +    |        ip ->
      + ,-.              id      /+\  wm*iq*Ld +    |          |                  vdc  === Cf/2
    Ed (   )                    (   ) vc/Sd         |          |                   -    |
        `-'                      \-/                |          |                       _|_
         |                        |                 |          |     +                  -
         +------------------------'                /^\         |   
                                         id/Sd +  ( | )  Cf/2 ===   vc         id*sd + iq*sq = dvc * Cf/2 + ic
              Ra   Xq"                   iq/Sq     \ /         |               vc = ic * Rf + ic'*Lf + vdc
         .----VVV--UUU------------.                 |          |     -         ic = vdc' * Cf/2 + ip
         |              --->      |                 |          |               vdc = ip * Rp + ip' * Ms
      + ,-.              iq      /+\ -wm*id*Lq +    |          |            
    Eq (   )                    (   ) vc/Sq         |          |            
        `-'                      \-/                |          |            
         |                        |                 |          |            
         +------------------------'                 '----+-----'  
                                                        _|_  
                                                         -
    
    sq = S * cos(th)
    sd = -S * sin(th) 
    ed = efd * sin(th)
    eq = efd * cos(th)
    Pe = 3/2 * (ed*id - eq*iq)
    
    dwm_dt = 1/M * (Pm - Kd*wm + Pe)
    dth_dt = -wm.q * wbase
    did_dt = 1/Ld * (ed - vc/sd - id * Ra - wm * iq * Ld - 1.0)
    diq_dt = 1/Lq * (eq - vc/sq - iq * Ra + wm * id * Lq)
    dvc_dt = 2/Cf * (id*sd + iq*sq - ic)
    dic_dt = 1/Lf * (vc - ic * Rf)
    dvf_dt = 2/Cf * (ic - idc)
    dip_dt = 1/Ms * (vf - ip*Rp)

    """

    # SI Params:

    vnom  = 4.16e3         
    sbase = 10.0e6        
    fbase = 60.0            
    npole = 2               
    wbase = 2*pi*fbase      
    Jm = 168870             
    Kd = 200.0              
    R  = 99.225*0.02        
    L  = 99.225*0.2/(wbase)

    # Electric to Mechanical:
    # 
    # Pe_pu = vdc_pu * idc_pu (V * A = W)
    # Pm_pu = Vs_pu * Ft_pu  (ship velocity and thrust, m/s * N = W)
    #
    # Pe_pu = Pm_pu
    #
    # vdc_pu * idc_pu =  Ft_pu * Vs_pu
    #
    # voltage_scale = thrust_scale = vbase
    # current_scale = velocity_scale = ibase
    #
    #
    # Plots:
    #
    # Vdc/Idc (in SI units)
    # Machine speed
    # Pm from the machine
    # Ship Velocity (m/s) and Propulsion sys Thurst (N)
    # Drag in SI units (N) Fp = Rp * ip
    # (Cummumlative updates/time for all, and state space value @ dt for all, error quanity/time)

    # per unit:

    vbase = vnom            # * sqrt(2) / sqrt(3)
    zbase = vbase**2/sbase  # 
    ibase = sbase / vnom    # * sqrt(2) / sqrt(3)
    Ra = R / zbase
    Xdpp = L * wbase / zbase
    Xqpp = L * wbase / zbase
    H = Jm * wbase**2 / (2*sbase*npole)
    
    Ld = 0.019
    Lq = 0.019
    Rf = 0.01   
    Lf = 0.001              
    Cf = 0.01               
    Clim = 0.01 
    M = 10.0        # machine inertia constant
    Ms = 100.0     # effective ship mass as seen from electrical bus
    Rp = 10.0
    Ra = 0.02

    RpDC  = 10.0
    RpAmp = 2.0
    fp = 1/30.0 # wave frequency

    S = sqrt(3/2) * 2 * sqrt(3) / pi

    # initial conditions:

    # transformer method #1:

    Pm0 = 1.0
    id0 = 0.0
    iq0 = 0.0
    wm0 = 0.0
    th0 = 0.0
    ic0 = 0.0
    vd0 = 0.0
    vq0 = 0.0
    vc0 = 0.0
    ip0 = 0.0
    rp0 = 1.0

    # transformer method #2:

    #Pm0 = 0.5
    #wm0 = -0.00021218086019892082
    #th0 = 3.601840275214806
    #id0 = -91.97803493716289
    #iq0 = -45.99448699465649
    #vc0 = 0.004343266878119145
    #ic0 = 0.4821762950119456
    #vp0 = -236.20899863372753
    #ip0 = 1.6431195806593977
    #rp0 = 10.0

    # inputs:

    efd0 = 1.0149

    # algebraic functions:

    sq = lambda: sqrt(3/2) * 2 * sqrt(3) / pi * cos(th.q)
    sd = lambda: -sqrt(3/2) * 2 * sqrt(3) / pi * sin(th.q) 
    ed = lambda: efd0 * sin(th.q)
    eq = lambda: efd0 * cos(th.q)

    # transformer method #2:

    #def vd():
    #    if sd() != 0.0:
    #        v = vc.q / sd()
    #    else:
    #        v = 0.0
    #    return v
    #
    #def vq():
    #    if sq() != 0.0:
    #        v = vc.q / sq()
    #    else:
    #        v = 0.0
    #    return v

    Pe  = lambda: 3/2 * (ed()*id.q - eq()*iq.q)
    Fd  = lambda: rp.q * ip.q  
    Pdc = lambda: vdc.q * idc.q
    Ps  = lambda: ip.q * rp.q * rp.q   # ship propulsion power (W)

    # diff eq:

    dwm = lambda: 1/M * (Pm.q - Kd*wm.q + Pe())
    dth = lambda: -wm.q * wbase

    # transformer method #1:
    did = lambda: 1/Ld * (ed() - id.q * Ra - wm.q * iq.q * Ld - 1.0)
    diq = lambda: 1/Lq * (eq() - iq.q * Ra + wm.q * id.q * Lq) 
    dvd = lambda: 1/Clim * (id.q - ic.q * sd() - vd.q / Ra) 
    dvq = lambda: 1/Clim * (iq.q - ic.q * sq() - vq.q / Ra)
    dvc = lambda: 1/Cf * (ic.q - ip.q) 
    dic = lambda: 1/Lf * (vd.q * sd() + vq.q * sq() - ic.q * Rf - vc.q)
    dip = lambda: 1/Ms * (vc.q - ip.q * rp.q)

    # transformer method #2:
    #did = lambda: 1/Ld * (ed() - vd() - id.q * Ra - wm.q * iq.q * Ld - 1.0)
    #diq = lambda: 1/Lq * (eq() - vq() - iq.q * Ra + wm.q * id.q * Lq)
    #dvc = lambda: 2/Cf * (id.q / sd() + iq.q / sq() - ic.q)
    #dic = lambda: 1/Lf * (vc.q - ic.q * Rf)
    #dvp = lambda: 2/Cf * (ic.q - ip.q)
    #dip = lambda: 1/Ms * (vp.q - ip.q * rp.q)

    dq0 = 0.001

    sys = liqss.Module("shipsys", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    Pm = liqss.Atom("Pm", source_type=liqss.SourceType.RAMP, x1=1.0, x2=1.0, t1=1.0, t2=20.0, units="MW", output_scale=sbase*1e-6) 
    wm = liqss.Atom("wm", x0=wm0, func=dwm, units="RPM",   dq=dq0) # , output_scale=wbase*9.5493)
    th = liqss.Atom("th", x0=th0, func=dth, units="deg",   dq=dq0) # , output_scale=180.0/pi)

    # transformer method #1:

    id = liqss.Atom("id", x0=id0, func=did, units="A",       dq=dq0    ) # ,    output_scale=ibase)
    iq = liqss.Atom("iq", x0=iq0, func=diq, units="A",       dq=dq0    ) # ,    output_scale=ibase)
    vd = liqss.Atom("vd", x0=vd0, func=dvd, units="kV",      dq=dq0    ) # ,    output_scale=vbase*1e-3)
    vq = liqss.Atom("vq", x0=vq0, func=dvq, units="kV",      dq=dq0    ) # ,    output_scale=vbase*1e-3)
    vc = liqss.Atom("vc", x0=vc0, func=dvc, units="kV",      dq=dq0    ) # ,    output_scale=vbase*1e-3)
    ic = liqss.Atom("ic", x0=ic0, func=dic, units="A",       dq=dq0    ) # ,    output_scale=ibase)
    ip = liqss.Atom("ip", x0=ip0, func=dip, units="knots",   dq=0.0001 ) # ,    output_scale=ibase*1.94384)

    # transformer method #2:
    
    # id = liqss.Atom("id", x0=id0, func=did, units="A",     dq=dq0) # , output_scale=ibase)
    # iq = liqss.Atom("iq", x0=iq0, func=diq, units="A",     dq=dq0) # , output_scale=ibase)
    # vc = liqss.Atom("vc", x0=vc0, func=dvc, units="kV",    dq=dq0) # , output_scale=vbase*1e-3)
    # ic = liqss.Atom("ic", x0=ic0, func=dic, units="A",     dq=dq0) # , output_scale=ibase)
    # vp = liqss.Atom("vp", x0=vp0, func=dvp, units="kV",    dq=dq0) # , output_scale=vbase*1e-3)
    # ip = liqss.Atom("ip", x0=ip0, func=dip, units="knots", dq=dq0) # , output_scale=ibase*1.94384)

     # 1. low speed:
    rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="Ns/m", x0=RpDC, output_scale=zbase)

    # 2. high speed:
    #RpDC = 10.0
    #rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="Ns/m", x0=RpDC, output_scale=zbase)

    # 3. sine drag:
    #rp = liqss.Atom("rp", source_type=liqss.SourceType.SINE, units="", x0=RpDC, xa=RpAmp, freq=fp, t1=30.0, dq=0.0001, output_scale=zbase)
    
    wm.connects(Pm, id, iq, th)
    th.connects(wm)

    # transformer method #1:

    id.connects(th)
    iq.connects(th)
    vd.connects(id, ic, th)
    vq.connects(iq, ic, th)
    vc.connects(ic, ip)
    ic.connects(vd, vq, vc)
    ip.connects(vc, rp)

    # transformer method #2:
    
    # id.connects(th, vc, wm)
    # iq.connects(th, vc, wm)
    # vc.connects(id, th, ic)
    # ic.connects(vc)
    # vp.connects(ic, ip)
    # ip.connects(vp, rp)

    sys.add_atoms(Pm, wm, th, id, iq)

    sys.add_atoms(ic, vd, vq, vc, ic, ip, rp)  # transformer method #1

    #sys.add_atoms(vc, ic, vp, ip, rp)           # transformer method #2

    # scenerio 0 (physics test)

    if 1:

        sys.initialize()
        sys.run_to(10.0, verbose=True) #, fixed_dt=1.0e-4)

    # scenerio 1 (low drag)

    if 0:

        RpDC = 10.0
        rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="", x0=RpDC, output_scale=zbase)

        tmax = 1.0

        # ss sim:
        sys.initialize()                                                                                
        sys.run_to(tmax, verbose=True, fixed_dt=1.0e-4)
        sys.save_data()

        dq_sclfactors = [1.0, 0.1, 0.1]

        errors = {}

        for dq_sclfactor in dq_sclfactors:

            # qss sim:
            for name, atom in sys.atoms.items():
                dq = atom.dq * dq_sclfactor
                atom.dq = dq
                atom.dqmax = dq
                atom.dqmin = dq

            sys.initialize()
            sys.run_to(tmax, verbose=True)

            # get relative error:
            for name, atom in sys.atoms.items():

                e = atom.get_error(typ="l2")

                if not name in errors:
                    errors[name] = {}
                    errors[name]["dq"] = []
                    errors[name]["e"] = []

                errors[name]["dq"].append(atom.dq)
                errors[name]["e"].append(e)

                #print("{} L^2 Relative Error = {:6.3f} %".format(name, e*100.0))

        # todo for accuracy plots:
        # 1) pick 3 atoms (fast, medium, slow)
        # 2) log scale(s) at least for x axes
        # 3) cummulative updates on sec. y axis (performace)

        if 0:
            plt.figure()
            for name, data in errors.items():
                 plt.plot(data["dq"], data["e"], label=name)
            plt.legend()
            plt.show()
    
    # scenerio 2 (high drag)

    if 0:
        RpDC = 15.0
        rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="", x0=Rp, output_scale=zbase)
        sys.initialize()
        sys.run_to(300.0, verbose=True) #, fixed_dt=1.0e-2)

    # scenerio 3 (choppy)

    if 0:
        sys.initialize()
        sys.run_to(10.0, verbose=True) #, fixed_dt=1.0e-3)

    # Accuracy Test: plot time/quantum for the first few seconds

    # for scaled quantum values of (1, 1/10, 1/100):
    # 1. run ss simulation (scenerio 3) (10 sec? 100 sec? see how long it takes)
    # 2. run qss simulation with same parameters
    # 3. compare results at ss time step assuming ZOH
    # 4. sqrt(sum(ss-qss).^2)./ss * 100.0 = percent error (pick three signals? fast-to-slow?, motor speed, terminal voltage, ship speed)

    f = open(r"c:\temp\initcond.txt", "w")

    r, c, j = 6, 2, 1

    subplot = True

    if subplot:
        plt.figure()

    for i, (name, atom) in enumerate(sys.atoms.items()):

        #dat = open(r"c:\temp\output_{}_{}.dat".format(name, atom.units), "w")

        #for t, x in zip(atom.tzoh, [x * atom.output_scale for x in atom.qzoh]):
        #    dat.write("{}\t{}\n".format(t, x))

        try:  # get steady state from last state space point
            f.write("    {}0 = {}\n".format(name, atom.qout2[-1]))
        except:  # or from last qss point:
            f.write("    {}0 = {}\n".format(name, atom.qout[-1]))

        #if name in ("id", "iq"): continue

        if subplot:
            plt.subplot(r, c, j)
        else:
            plt.figure()

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(atom.tzoh, [x * atom.output_scale for x in atom.qzoh], 'b-')

        try:
            ax1.plot(atom.tout2, [x * atom.output_scale for x in atom.qout2], 'c--')
        except:
            pass

        ax2.plot(atom.tout, atom.nupd, 'r-')
        
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')
        ax2.set_ylabel('total updates', color='r')

        plt.xlabel("t (s)")
        #plt.legend()
        ax1.grid()

        dir = r"c\temp"  # <--- change this to latex image folder dir !
        filepath = name + ".pdf" # os.path.join(dir, name + ".pdf")
        #plt.savefig(filepath, bbox_inches='tight')

        j += 1

        #dat.close()

    f.close()
    plt.show()


def shipsys5():

    """
    Machine model:
                   t_
                   /
    dw = 1/(2*H) * | [(Tm - Te) - K*dw] dt
                  _/
                   0
    w = w0 + dw

         .----------+----------+-------.             
         |          |          |       |       
        ,-.    1   _|_    1    >      / \      
    Tm ( ^ )  ---  ___   ---   >     ( v ) Te  
        `-'   2*H   |    Kd    >      \ /      
         |          |          |       |       
         '----------+----------+-------'  
         
               Ra  Xd"     + ,-.           1:Sd         Rf   Lf                             
         .----VVV--UUU------(   )----o---.      .----o--VVV--UUU---+-----------------.
         |    --->           `-'         |      |         --->     |      --->       |      id = dvd * Clim + vd / Rd + ic * sd
      + ,-.    Id          wm*Lm*id       ) || (          idc      |       ip        |      dvd = 1/Clim * (id - vd / Rd - ic * sd)
    Ed (   )                              ) || (                   |                 |
        `-'                               ) || (                   |                <       
         |                               |      |     +            |      +         <  Rp
         +---------------------------o---'      '-.               _|_               < 
                                                  |  vdc      Cf  ___    vf          |
              Ra   Xq"       ,-. +         1:Sq   |                |                (
         .----VVV--UUU------(   )----o---.      .-'   -            |      -         (  Lp
         |    --->           `-'         |      |                  |                (
      + ,-.    Iq          wm*Lm*id       ) || (                   |                 |
    Eq (   )                              ) || (                   |                 |
        `-'                               ) || (                   |                 |
         |                               |      |                  |                 |
         +---------------------------o---'      '----o-------------+-----------------'


         [ DQ Machine ] --- [ Resitive DQ Load ]

         [ Vd/Vq dc voltage sources ] --  [ DQ to DC Transformer ] -- [ Resitive Load ]    "three-winding qss tranformer"

    Module 
    N Atoms
    M Ports (atom states alias with getters and setters)

    Te = Ed * Id - Eq * Iq            
    Tm = Pm / w
    S  = sqrt(3/2) * 2 * sqrt(3) / pi
    Sd = S * cos(th)
    Sq = -S * sin(th)
    dw = 1/(2*H) * (Pm - D*w - Te)
    dth = w0 - w
    """

    # SI Params:

    vnom  = 4.16e3         
    sbase = 10.0e6        
    fbase = 60.0            
    npole = 2               
    wbase = 2*pi*fbase      
    Jm = 168870             
    Kd = 200.0              
    R  = 99.225*0.02        
    L  = 99.225*0.2/(wbase)

    # Electric to Mechanical:
    # 
    # Pe_pu = vdc_pu * idc_pu (V * A = W)
    # Pm_pu = Vs_pu * Ft_pu  (ship velocity and thrust, m/s * N = W)
    #
    # Pe_pu = Pm_pu
    #
    # vdc_pu * idc_pu =  Ft_pu * Vs_pu
    #
    # voltage_scale = thrust_scale = vbase
    # current_scale = velocity_scale = ibase
    #
    #
    # Plots:
    #
    # Vdc/Idc (in SI units)
    # Machine speed
    # Pm from the machine
    # Ship Velocity (m/s) and Propulsion sys Thurst (N)
    # Drag in SI units (N) Fp = Rp * ip
    # (Cummumlative updates/time for all, and state space value @ dt for all, error quanity/time)

    # per unit:

    vbase = vnom            # * sqrt(2) / sqrt(3)
    zbase = vbase**2/sbase  # 
    ibase = sbase / vnom    # * sqrt(2) / sqrt(3)
    Ra = R / zbase
    Xdpp = L * wbase / zbase
    Xqpp = L * wbase / zbase
    H = Jm * wbase**2 / (2*sbase*npole)
    
    Ld = 0.02
    Lq = 0.02
    Rf = 0.01   
    Lf = 0.01              
    Cf = 0.01               
    #Clim = 0.001 
    #Rlim = 0.1
    M = 10.0     # machine inertia constant
    Ms = 100.0  # effective ship mass as seen from electrical bus
    Ra = 0.02

    RpDC  = 100.0
    RpAmp = 2.0
    fp = 1/30.0 # wave frequency

    S = sqrt(3/2) * 2 * sqrt(3) / pi

    # initial conditions:

    Pm0 = 1.0
    wm0 = 2.4917180670524957e-07
    th0 = 7.068555136133191
    id0 = 39061870.4649704
    iq0 = 39059657.728514306
    vc0 = -2994041.181181343
    ic0 = -2641369.0284578763
    vp0 = -2604104.186739992
    ip0 = 44747.32850412856
    rp0 = 1.0

    Pm0 = 0.0
    wm0 = 0.0
    th0 = 0.01
    id0 = 0.0
    iq0 = 0.0
    vc0 = 0.0
    ic0 = 0.0
    vp0 = 0.0
    ip0 = 0.0
    rp0 = 0.0

    # inputs:

    efd0 = 1.0149

    # algebraic functions:

    sq = lambda: S * cos(th.q)
    sd = lambda: -S * sin(th.q) 
    ed = lambda: efd0 * sin(th.q)
    eq = lambda: efd0 * cos(th.q)

    sq_inv = lambda:  pi / (3 * sqrt(2) * cos(th.q))
    sd_inv = lambda: -pi / (3 * sqrt(2) * sin(th.q)) 

    Pe  = lambda: 3/2 * (ed()*id.q - eq()*iq.q)
    Fd  = lambda: rp.q * ip.q  
    Pdc = lambda: vdc.q * idc.q
    Ps  = lambda: ip.q * rp.q * rp.q   # ship propulsion power (W)

    # diff eq:

    #did = 1/Ld * (ed - id * Ra - wm * iq * Ld - vd)

    dwm = lambda: 1/M * (Pm.q - Kd*wm.q + Pe())
    dth = lambda: -wm.q * wbase

    # transformer method #1:

    #did = lambda: 1/Ld * (ed() - id.q * Ra - wm.q * iq.q * Ld - vd.q)
    #diq = lambda: 1/Lq * (eq() - iq.q * Ra + wm.q * id.q * Lq - vq.q)
    #dvd = lambda: 1/Clim * (id.q - ic.q * sd() - vd.q / Rlim) 
    #dvq = lambda: 1/Clim * (iq.q - ic.q * sq() - vq.q / Rlim)
    #dvc = lambda: 1/Cf * (ic.q - ip.q) 
    #dic = lambda: 1/Lf * (vd.q * sd() + vq.q * sq() - ic.q * Rf - vc.q)
    #dip = lambda: 1/Ms * (vc.q - ip.q * rp.q)

    # transformer method #2:

    did = lambda: 1/Ld * (ed() - id.q * Ra - wm.q * iq.q * Ld - vc.q * sd_inv())
    diq = lambda: 1/Lq * (eq() - iq.q * Ra + wm.q * id.q * Lq - vc.q * sd_inv())
    dvc = lambda: 2/Cf * (id.q * sd_inv() + iq.q * sq_inv() - ic.q)
    dic = lambda: 1/Lf * (vc.q - ic.q * Rf - vp.q)
    dvp = lambda: 2/Cf * (ic.q - ip.q)
    dip = lambda: 1/Ms * (vp.q - ip.q * rp.q)

    dqmin = 0.00001
    dqmax = 0.001
    dqerr = 0.001

    sys = liqss.Module("shipsys", dqmin=dqmin, dqmax=dqmax, dqerr=dqerr)

    Pm = liqss.Atom("Pm", source_type=liqss.SourceType.RAMP, x1=1.0, x2=1.0, t1=1.0, t2=20.0, units="MW", output_scale=sbase*1e-6)
    
    # 1. low speed:
    RpDC = 1.0
    rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="Ns/m", x0=RpDC, output_scale=zbase)

    # 2. high speed:
    #RpDC = 10.0
    #rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="Ns/m", x0=RpDC, output_scale=zbase)

    # 3. sine drag:
    #rp = liqss.Atom("rp", source_type=liqss.SourceType.SINE, units="", x0=RpDC, xa=RpAmp, freq=fp, t1=30.0, dq=0.0001, output_scale=zbase)
    
    dq0 = 0.001

    wm = liqss.Atom("wm", x0=wm0, func=dwm, units="RPM",     dq=dq0) # output_scale=wbase*9.5493)
    th = liqss.Atom("th", x0=th0, func=dth, units="deg",     dq=dq0) # output_scale=180.0/pi)
                                                                
    id = liqss.Atom("id", x0=id0, func=did, units="A",       dq=dq0) # output_scale=ibase)
    iq = liqss.Atom("iq", x0=iq0, func=diq, units="A",       dq=dq0) # output_scale=ibase)
    vc = liqss.Atom("vc", x0=vc0, func=dvc, units="kV",      dq=dq0) # output_scale=vbase*1e-3)
    ic = liqss.Atom("ic", x0=ic0, func=dic, units="A",       dq=dq0) # output_scale=ibase)
    ip = liqss.Atom("ip", x0=ip0, func=dip, units="knots",   dq=dq0) # output_scale=ibase*1.94384)

    # tranformer method # 1:

    #vd = liqss.Atom("vd", x0=vd0, func=dvd, units="kV",      dq=0.001) # output_scale=vbase*1e-3)
    #vq = liqss.Atom("vq", x0=vq0, func=dvq, units="kV",      dq=0.001) # output_scale=vbase*1e-3)
    
    # tranformer method # 2:

    vp = liqss.Atom("vp", x0=vp0, func=dvp, units="kV",    dq=dq0) # , output_scale=vbase*1e-3)

    wm.connects(Pm, id, iq, th)
    th.connects(wm)

    # transformer method #1:

    #id.connects(th, wm, iq, vd)
    #iq.connects(th, wm, id, vq)
    #vd.connects(id, ic, th)
    #vq.connects(iq, ic, th)
    #vc.connects(ic, ip)
    #ic.connects(vd, vq, vc)
    #ip.connects(vc, rp)

    # transformer method #2:

    id.connects(th, vc, wm, iq)
    iq.connects(th, vc, wm, id)
    vc.connects(id, th, ic)
    ic.connects(vc, vp)
    vp.connects(ic, ip)
    ip.connects(vp, rp)

    sys.add_atoms(Pm, wm, th)

    #sys.add_atoms(id, iq, ic, vd, vq, vc, ic, ip, rp)    # transformer method #1

    sys.add_atoms(id, iq, vc, ic, vp, ip, rp)           # transformer method #2

    # scenerio 0 (quick test)

    if 1:

        tmax = 1.0

        #sys.initialize()                                                                                
        #sys.run_to(tmax, verbose=True, fixed_dt=1.0e-5)
        #sys.save_data()

        sys.initialize()
        sys.run_to(tmax, verbose=True)

    # scenerio 1 (low drag)

    if 0:

        RpDC = 10.0
        rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="", x0=RpDC, output_scale=zbase)

        tmax = 1.0

        # ss sim:
        sys.initialize()                                                                                
        sys.run_to(tmax, verbose=True, fixed_dt=1.0e-4)
        sys.save_data()

        dq_sclfactors = [1.0, 0.1, 0.1]

        errors = {}

        for dq_sclfactor in dq_sclfactors:

            # qss sim:
            for name, atom in sys.atoms.items():
                dq = atom.dq * dq_sclfactor
                atom.dq = dq
                atom.dqmax = dq
                atom.dqmin = dq

            sys.initialize()
            sys.run_to(tmax, verbose=True)

            # get relative error:
            for name, atom in sys.atoms.items():

                e = atom.get_error(typ="l2")

                if not name in errors:
                    errors[name] = {}
                    errors[name]["dq"] = []
                    errors[name]["e"] = []

                errors[name]["dq"].append(atom.dq)
                errors[name]["e"].append(e)

                #print("{} L^2 Relative Error = {:6.3f} %".format(name, e*100.0))

        # todo for accuracy plots:
        # 1) pick 3 atoms (fast, medium, slow)
        # 2) log scale(s) at least for x axes
        # 3) cummulative updates on sec. y axis (performace)

        if 0:
            plt.figure()
            for name, data in errors.items():
                 plt.plot(data["dq"], data["e"], label=name)
            plt.legend()
            plt.show()
    
    # scenerio 2 (high drag)

    if 0:
        RpDC = 15.0
        rp = liqss.Atom("rp", source_type=liqss.SourceType.CONSTANT, units="", x0=Rp, output_scale=zbase)
        sys.initialize()
        sys.run_to(300.0, verbose=True) #, fixed_dt=1.0e-2)

    # scenerio 3 (choppy)

    if 0:
        sys.initialize()
        sys.run_to(10.0, verbose=True) #, fixed_dt=1.0e-3)

    # Accuracy Test: plot time/quantum for the first few seconds

    # for scaled quantum values of (1, 1/10, 1/100):
    # 1. run ss simulation (scenerio 3) (10 sec? 100 sec? see how long it takes)
    # 2. run qss simulation with same parameters
    # 3. compare results at ss time step assuming ZOH
    # 4. sqrt(sum(ss-qss).^2)./ss * 100.0 = percent error (pick three signals? fast-to-slow?, motor speed, terminal voltage, ship speed)

    f = open(r"c:\temp\initcond.txt", "w")

    r, c, j = 6, 2, 1

    subplot = True

    if subplot:
        plt.figure()

    for i, (name, atom) in enumerate(sys.atoms.items()):

        #dat = open(r"c:\temp\output_{}_{}.dat".format(name, atom.units), "w")

        #for t, x in zip(atom.tzoh, [x * atom.output_scale for x in atom.qzoh]):
        #    dat.write("{}\t{}\n".format(t, x))

        try:  # get steady state from last state space point
            f.write("    {}0 = {}\n".format(name, atom.qout2[-1]))
        except:  # or from last qss point:
            f.write("    {}0 = {}\n".format(name, atom.qout[-1]))

        #if name in ("id", "iq"): continue

        if subplot:
            plt.subplot(r, c, j)
        else:
            plt.figure()

        ax1 = plt.gca()
        ax2 = ax1.twinx()
        
        ax1.plot(atom.tzoh, [x * atom.output_scale for x in atom.qzoh], 'b-')

        try:
            ax1.plot(atom.tout2, [x * atom.output_scale for x in atom.qout2], 'c--')
        except:
            pass

        ax2.plot(atom.tout, atom.nupd, 'r-')
        
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')
        ax2.set_ylabel('total updates', color='r')

        plt.xlabel("t (s)")
        plt.legend()
        ax1.grid()

        dir = r"c\temp"  # <--- change this to latex image folder dir !
        filepath = name + ".pdf" # os.path.join(dir, name + ".pdf")
        #plt.savefig(filepath, bbox_inches='tight')

        j += 1

        #dat.close()

    f.close()
    plt.show()

    # presentation plots:

    plt.figure()
     

if __name__ == "__main__":

    #simple()
    #stiffline()
    #delegates()
    genset()
    #shipsys()
    #shipsys2()
    #dynphasor()
    #gencls()
    #genphasor()
    #shipsys3()
    #shipsys4()
    #shipsys5()