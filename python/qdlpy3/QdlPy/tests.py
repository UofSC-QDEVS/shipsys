
import os
import time

from qdl import *
from models import *


def test1():

    sys = System(qss_method=QssMethod.LIQSS1)

    dq = 1e-1

    ground = GroundNode("ground")
    node1 = LimNode("node1", 1.0, 0.1, 1.0, dq=dq)
    branch1 = LimBranch("branch1", 0.1, 0.1, 1.0, dq=dq)

    sys.add_devices(ground, node1, branch1)

    branch1.connect(ground, node1)
    node1.connect(branch1, terminal='j')

    tstop = 5.0
    dt = 1.0e-2
    dc = 1
    optimize_dq = 1
    ode = 1
    qss = 1

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        node1.g = 2.0

    sys.schedule(event, 2.0)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, print_interval=1e-1)

    plotargs = {"plot_ode":ode, "plot_qss":False, "plot_zoh":qss,
                "plot_upd":True, "upd_bins":1000, "errorband":True,
                "legloc":"lower right"}

    sys.plot(branch1.current, node1.voltage, **plotargs)


def test2():

    w = 2*PI*60
    sbase = 100e6
    vbase = 115e3

    zbase = vbase**2/sbase
    ybase = 1/zbase
    ibase = sbase/vbase

    rpu = 0.01
    xpu = 0.1
    bpu = 0.1

    r = rpu * zbase
    l = xpu * zbase / w
    c = bpu * zbase / w
    g = 0.0

    pload = 2.0e6
    pf = 0.9
    qload = pload * tan(acos(pf))

    vgen = vbase

    rload = vbase**2 / pload
    lload  = vbase**2 / (qload / w)

    dq_v = vbase * 0.01
    dq_i = pload / dq_v

    sys = System(dq=dq_v)

    ground = GroundNode("ground")

    node1 = LimNode("node1", c=c, g=g, dq=dq_v)
    node2 = LimNode("node2", c=c, g=g, dq=dq_v)
    node3 = LimNode("node3", c=c, g=g, dq=dq_v)
    node4 = LimNode("node4", c=c, g=g, dq=dq_v)
    node5 = LimNode("node5", c=c, g=g, dq=dq_v)

    branch1 = LimBranch("branch1", l=l, r=r, e0=vgen, dq=dq_v)
    branch2 = LimBranch("branch2", l=l, r=r, dq=dq_v)
    branch3 = LimBranch("branch3", l=l, r=r, dq=dq_v)
    branch4 = LimBranch("branch4", l=l, r=r, dq=dq_v)
    branch5 = LimBranch("branch5", l=l, r=r, dq=dq_v)
    branch6 = LimBranch("branch6", l=l*10, r=rload, dq=dq_v)

    sys.add_devices(ground, node1, node2, node3, node4, node5,
                    branch1, branch2, branch3, branch4, branch5, branch6)

    # inode, jnode
    branch1.connect(ground, node1)
    branch2.connect(node1, node2)
    branch3.connect(node2, node3)
    branch4.connect(node3, node4)
    branch5.connect(node4, node5)
    branch6.connect(node5, ground)

    node1.connect(branch1, terminal="j")
    node1.connect(branch2, terminal="i")

    node2.connect(branch2, terminal="j")
    node2.connect(branch3, terminal="i")

    node3.connect(branch3, terminal="j")
    node3.connect(branch4, terminal="i")

    node4.connect(branch4, terminal="j")
    node4.connect(branch5, terminal="i")

    node5.connect(branch5, terminal="j")
    node5.connect(branch6, terminal="i")

    tstop = 20.0
    dt = 1.0e-2
    dc = 1
    optimize_dq = 1
    ode = 1
    qss = 1

    sys.initialize(dt=dt, dc=dc)

    def fault(sys):
        node2.g = 100.0

    def clear(sys):
        node2.g =  0.1

    sys.schedule(fault, tstop*0.1)

    sys.schedule(clear, tstop*0.1+0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq)

    plotargs = {"plot_ode":ode, "plot_qss":False, "plot_zoh":qss,
                "plot_upd":True, "upd_bins":1000, "errorband":True,
                "legloc":"lower right"}

    sys.plot(node1.v, node2.v, node3.v, **plotargs)
    sys.plot(node4.v, node5.v, **plotargs)
    sys.plot(branch1.i, branch2.i, branch3.i, **plotargs)
    sys.plot(branch4.i, branch5.i, branch6.i, **plotargs)


def test3():

    SCENARIO = "LOAD_INCREASE"

    ws = 2*60*PI
    vfdb = 90.1
    VLL = 4160.0
    vref = 1.0

    dq_i = 1e-1
    dq_v = 1e-1
    dq_wr = 1e-3
    dq_th = 1e-4
    dq_f = 1e-3
    dq_x1 = 1e-8
    dq_x2 = 1e-9
    dq_x3 = 1e-5
    dq_vfd = 1e-2

    sys = System(qss_method=QssMethod.LIQSS1)

    sm = SyncMachineDQ("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v,
                        dq_th=dq_th, dq_wr=dq_wr, dq_f=dq_f)

    avr = AC8B("avr", vref=vref, dq_x1=dq_x1, dq_x2=dq_x2, dq_x3=dq_x3,
               dq_vfd=dq_vfd)

    bus1 = LimNodeDQ("bus1", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus2 = LimNodeDQ("bus2", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus3 = LimNodeDQ("bus3", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    cable12 = LimBranchDQ("cable12", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    cable13 = LimBranchDQ("cable13", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    cable23 = LimBranchDQ("cable23", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=dq_i)
    trload = TRLoadDQ2("trload", w=ws, vdc0=VLL, dq_i=dq_i/10, dq_v=dq_v/10)
    im = InductionMachineDQ("im", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i/10, dq_wr=dq_wr, Tb=0.0)
    gnd = GroundNodeDQ("gnd")

    sys.add_devices(sm, avr, bus1, bus2, bus3, cable12, cable13, cable23, load, trload, im, gnd)

    sm.connect(bus1, avr)
    avr.connect(bus1, sm)
    bus1.connect(sm, terminal="j")

    bus1.connect(cable12, terminal="i")
    cable12.connect(bus1, bus2)
    bus2.connect(cable12, terminal="j")

    bus1.connect(cable13, terminal="i")
    cable13.connect(bus1, bus3)
    bus3.connect(cable13, terminal="j")

    bus2.connect(cable23, terminal="i")
    cable23.connect(bus2, bus3)
    bus3.connect(cable23, terminal="j")

    if SCENARIO == "IM_START":
        im.connect(gnd)
        im.wr0 = 0.0
    else:
        im.connect(bus2)
        bus2.connect(im, terminal="i")

    load.connect(bus3, gnd)
    bus3.connect(load, terminal="i")

    trload.connect(bus3)
    bus3.connect(trload, terminal="i")

    tstop = 30
    dt = 1.0e-1
    dc = 1
    optimize_dq = 0
    ode = 0
    qss = 1

    outdir = r"C:\Temp\qdl"

    chk_ss_delay = 1.0
    chk_ss_delay = None

    def increase_load(sys):
        load.r *= 0.8

    def apply_fault(sys):
        bus2.g = 1e3
        bus2.c = 1e2
        #bus2.vd.set_state(0.0)
        #bus2.vq.set_state(0.0)

    def clear_fault(sys):
        bus2.g = 1e-4

    def trip_line(sys):
        cable12.l = 1e6

    def vref_increase(sys):
        avr.vref = 1.02

    def im_start(sys):
        im.connect(bus2)
        bus2.connect(im, terminal="i")

    if SCENARIO == "LOAD_INCREASE":  # load increase:
        sys.schedule(increase_load, 1.0)

    if SCENARIO == "FAULT_BUS2":  # fault simulation:
        sys.schedule(apply_fault, 1.0)
        sys.schedule(clear_fault, 1.01)
        #sys.schedule(trip_line, 1.0 + tfault)

    if SCENARIO == "VREF_INCREASE":  # SM vset change simulation:
        sys.schedule(vref_increase, 1.0)

    if SCENARIO == "IM_START":
        sys.schedule(im_start, 1.0)

    sys.Km = 2.0

    #sys.storage_type = StorageType.LIST
    #sys.storage_type = StorageType.ARRAY
    sys.storage_type = StorageType.DEQUE

    sys.initialize(dt=dt, dc=dc, savedt=1e-9, tocsv=False, outdir=outdir)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="Radau",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay, print_interval=1e-3)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":qss,
                "plot_upd":False,
                "errorband":False}

    for atom in sys.state_atoms:

        print(f"writing {atom.device.name}.{atom.name} data to file...")

        pth = f"C:\\Temp\\qdl\\{int(tstop)}s\\{atom.device.name}_{atom.name}"

        pth_tout = pth + "_tout.pickle"
        pth_qout = pth + "_qout.pickle"
        pth_tzoh = pth + "_tzoh.pickle"
        pth_qzoh = pth + "_qzoh.pickle"
        pth_nupd = pth + "_nupd.pickle"

        with open(pth_tout, 'wb') as f: pickle.dump(atom.tout, f)
        with open(pth_qout, 'wb') as f: pickle.dump(atom.qout, f)
        with open(pth_tzoh, 'wb') as f: pickle.dump(atom.tzoh, f)
        with open(pth_qzoh, 'wb') as f: pickle.dump(atom.qzoh, f)
        with open(pth_nupd, 'wb') as f: pickle.dump(atom.nupd, f)

    print("\nDONE.")


def test4():

    ws = 2*60*PI  # system radian frequency
    vfdb = 90.1   # sm rated field voltage
    VLL = 4160.0  # bus nominal line-to-line rms voltage
    vref = 1.0    # pu voltage setpoint

    dq_i = 1e-1
    dq_v = 1e-1
    dq_wr = 1e-4

    sys = System(dq=1e-3)

    source = LimBranchDQ("source", l=1e-4, r=0.0, vd0=VLL, vq0=0.0,  w=ws, dq=dq_i)
    bus1 = LimNodeDQ("bus1", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=0.0, dq=dq_v)
    bus2 = LimNodeDQ("bus2", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=0.0, dq=dq_v)
    cable = LimBranchDQ("cable", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    im = InductionMachineDQ("im", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i, dq_wr=dq_wr, Tb=0.0)
    gnd = GroundNodeDQ("gnd")

    sys.add_devices(source, bus1, cable, bus2, im, gnd)

    source.connect(bus1, gnd)
    bus1.connect(source, terminal="j")

    bus1.connect(cable, terminal="i")
    cable.connect(bus1, bus2)
    bus2.connect(cable, terminal="j")

    im.connect(bus2)
    bus2.connect(im, terminal="i")

    tstop = 0.1

    dt = 1.0e-3
    dc = True

    ode = False
    qss = True
    upds = False
    upd_bins = 1000

    chk_ss = True
    chk_ndq = 10
    chk_dmax = 10

    plotargs = {
    "plot_ode":ode,
    "plot_qss":qss,
    "plot_upd":upds,
    "upd_bins":upd_bins
    }

    def event(sys):
        im.Tb = 26.53e3 * 0.5

    sys.schedule(event, tstop*0.2)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA")

    sys.plot(im.iqs, im.ids, im.iqr, im.idr, **plotargs)
    sys.plot(bus1.vd, bus1.vq, bus2.vd, bus2.vq, **plotargs)
    sys.plot(cable.id, cable.iq,  **plotargs)


def test5():

    sys = System(dq=1e-3)

    pendulum = Pendulum("pendulum", r=0.4, l=8.0, theta0=PI/4, omega0=0.0,
                        dq_omega=2e-2, dq_theta=2e-2)

    sys.add_devices(pendulum)

    tstop = 60.0
    dt = 1.0e-3
    dc = 0
    optimize_dq = 1
    ode = 1
    qss = 1

    chk_ss_delay = 10.0
    #chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        pendulum.omega.set_state(0.0)
        pendulum.theta.set_state(-PI/4, quantize=True)

    #sys.schedule(event, tstop*0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_upd":qss, "upd_bins":1000, "errorband":True}

    sys.plot(pendulum.omega, pendulum.theta, **plotargs)

    sys.plotxy(pendulum.theta, pendulum.omega, arrows=False, ss_region=True)


def test6():

    sys = System(qss_method=QssMethod.LIQSS1)

    k = 1.0
    r1 = 1.0
    r2 = 1.0
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    th10 = PI/4
    w10 = 0.0
    th20 = -PI/4
    w20 = 0.0
    dq_w = 1e-2
    dq_th = 1e-2

    pendulums = CoupledPendulums("pendulums", k, r1, r2, l1, l2, m1, m2,
                                 th10, w10, th20, w20, dq_w, dq_th)

    sys.add_devices(pendulums)

    tstop = 40.0
    dt = 1.0e-2
    dc = 0
    optimize_dq = 0
    ode = 1
    qss = 1

    chk_ss_delay = 5.0
    #chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        pendulums.th1.set_state(PI/4)

    #sys.schedule(event, tstop*0.1)

    runargs = {"ode":ode, "qss":qss, "verbose":2, "ode_method":"LSODA",
               "optimize_dq":optimize_dq}

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_upd":False, "errorband":True}

    sys.run(tstop, **runargs)

    sys.plotxy(pendulums.w1, pendulums.th1, arrows=True, ss_region=False)
    sys.plotxy(pendulums.w2, pendulums.th2, arrows=False, ss_region=True)

    #sys.plotxyt(pendulums.w1, pendulums.th1, ss_region=True)
    #sys.plotxyt(pendulums.w2, pendulums.th2, ss_region=True)

    sys.plot(pendulums.w1, pendulums.th1, **plotargs)
    #sys.plot(pendulums.th1, pendulums.th2, **plotargs)


def test7():

    sys = System()

    ground = GroundNode("ground")
    node1 = LimNode("node1", 1.0, 1.0, 1.0, dq=1e-2)
    branch1 = LimBranch("branch1", 1.0, 1.0, 1.0, dq=1e-2)

    branch1.connect(ground, node1)

    sys.add_devices(ground, node1, branch1)

    tstop = 20.0
    dt = 1.0e-2
    dc = 0
    optimize_dq = 0
    ode = 1
    qss = 1

    #chk_ss_delay = 5.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        node1.g = 2.0

    sys.schedule(event, tstop*0.5)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_upd":qss, "errorband":True}

    sys.plot(node1.voltage, branch1.current, **plotargs)


def test8():

    sys = System()

    Vs = 10;
    Gs = 1e2;
    Clim = 1e-6;
    Ra = 0.1;
    La = 0.001;
    Ke = 0.1;
    Kt = 0.1;
    Jm = 0.01;
    Bm = 0.001;
    Jp = 0.01;
    Fp = 1;
    JL = 0.5
    TL = 0;
    BL = 0.1;

    vsource = LimNode2("vsource", Clim, Gs, Vs*Gs, v0=10.0)
    ground = Ground("ground")
    motor = DCMotor("motor")

    sys.connect(motor.positive, vsource.positive)
    sys.connect(motor.negative, ground.positive)

    sys.add_devices(vsource, ground, motor)

    tstop = 20.0
    dt = 1.0e-2
    dc = 0
    optimize_dq = 0
    ode = 1
    qss = 1

    #chk_ss_delay = 5.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    #def event(sys):
    #    node1.g = 2.0

    #sys.schedule(event, tstop*0.5)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_upd":qss, "errorband":True}

    sys.plot(motor.ia, motor.wr, **plotargs)


def test9():

    dq_i = 1e-2
    dq_w = 1e-2
    freq = 100.0
    duty = 0.5

    sys = System()

    ground = GroundNode("ground")

    source = PwmSourceNode("source", vlo=0.0, vhi=1.0, freq=freq, duty=duty)

    motor = DCMotor("motor", ra=0.1, La=0.01, Jm=0.1, Bm=0.001, Kt=0.1, Ke=0.1,
                    ia0=0.0, wr0=0.0, dq_ia=dq_i, dq_wr=dq_w)

    motor.connect(source, ground)

    sys.add_devices(ground, source, motor)

    tstop = 10.0
    dt = 1e-4
    dc = 0
    optimize_dq = 0
    ode = 0
    qss = 1

    chk_ss_delay = None

    def event(sys):
        source.voltage.duty = 0.8

    sys.schedule(event, tstop*0.5)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":1,
                "plot_upd":0,
                "errorband":0,
                }

    sys.plot(source.voltage, motor.ia, motor.wr, **plotargs)


def test10():

    dq_v = 1e-1
    dq_i = 1e-1

    dq_ia = 1e-1
    dq_w = 1e-1

    freq = 100.0
    duty = 1.0

    sys = System()

    ground = GroundNode("ground")

    source = PwmSourceNode("source", vlo=0.0, vhi=10.0, freq=freq, duty=duty)

    cable = LimBranch("cable", 0.001, 0.01, dq=dq_i)

    bus = LimNode("bus", 0.01, dq=dq_v)

    motor = DCMotor("motor", ra=0.1, La=0.01, Jm=0.1, Bm=0.001, Kt=0.1, Ke=0.1,
                    ia0=0.0, wr0=0.0, dq_ia=dq_ia, dq_wr=dq_w)

    sys.add_devices(ground, source, cable, bus, motor)

    cable.connect(source, bus)

    bus.connect(cable, terminal="j")

    motor.connect(bus, ground)

    tstop = 10.0
    dt = 1e-2
    dc = 1
    optimize_dq = 0
    ode = 1
    qss = 1

    chk_ss_delay = None

    def event(sys):
        source.voltage.duty = 0.8

    sys.schedule(event, tstop*0.5)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":1,
                "plot_upd":0,
                "errorband":0,
                }

    sys.plot(source.voltage, cable.current, bus.voltage, **plotargs)
    sys.plot(motor.ia, motor.wr, **plotargs)


def test11():

    dq_i = 1e-2
    dq_v = 1e-2

    freq = 100.0
    duty = 0.5

    i0 = 0.0
    v0 = 0.0

    Vs = 10.0

    sys = System()

    ground = GroundNode("ground")

    source = LimBranch("source", l=0.01, r=0.01, e0=Vs, i0=i0, dq=dq_i)

    converter = Converter("converter", r=0.01, l=0.01, c=0.01, g=0.01, freq=freq, duty=duty,
                          vi0=v0, io0=i0, dq_i=dq_i, dq_v=dq_v)

    load = LimNode("load", c=0.01, g=1.0, h0=0.0, v0=v0, dq=dq_v)

    sys.add_devices(ground, source, converter, load)

    source.connect(ground, converter)
    converter.connect(source, ground, load, terminal="j")
    load.connect(converter, terminal="j")

    tstop = 10.0
    dt = 1e-3
    dc = 0
    optimize_dq = 0
    ode = 0
    qss = 1

    chk_ss_delay = None

    def event(sys):
        source.voltage.duty = 0.8

    #sys.schedule(event, tstop*0.5)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":1,
                "plot_upd":0,
                "errorband":0,
                }

    sys.plot(source.current, converter.e, converter.h, load.voltage, **plotargs)


def test12():

    dq_i = 1e-2
    dq_v = 1
    dq_ia = 1e-1
    dq_wr = 1e-1

    freq = 100.0
    duty = 0.5

    i0 = 0.0
    v0 = 0.0

    Vs = 10.0

    sys = System()

    ground = GroundNode("ground")

    source = LimBranch("source", l=0.01, r=0.01, e0=Vs, i0=i0, dq=dq_i)

    converter = Converter("converter", r=0.01, l=0.01, c=0.01, g=0.01, freq=freq, duty=duty,
                          vi0=v0, io0=i0, dq_i=dq_i, dq_v=dq_v)

    bus = LimNode("bus", c=0.01, g=10.0, dq=dq_v)

    motor = DCMotor("motor", ra=0.1, La=0.01, Jm=0.1, Bm=0.001, Kt=0.1, Ke=0.1,
                    ia0=0.0, wr0=0.0, dq_ia=dq_ia, dq_wr=dq_wr)

    sys.add_devices(ground, source, converter, bus, motor)

    source.connect(ground, converter)
    converter.connect(source, ground, bus, terminal="j")
    bus.connect(converter, terminal="j")

    bus.connect(motor, terminal="i")
    motor.connect(bus, ground)

    tstop = 15.0
    dt = 1e-3
    dc = 0
    optimize_dq = 0
    ode = 0
    qss = 1

    chk_ss_delay = None

    def event1(sys): motor.ra = 0.1
    sys.schedule(event1, 5.0)

    def event2(sys): motor.Bm = 0.01
    sys.schedule(event2, 10.0)

    motor.ra = 1000.0

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":1,
                "plot_upd":0,
                "errorband":0,
                }

    #sys.plot(source.current, converter.e, converter.h, bus.voltage, **plotargs)
    sys.plot(source.current, bus.voltage, motor.ia, motor.wr, **plotargs)

def test13():

    sys = System(qss_method=QssMethod.QSS2)

    dq = 1

    liqss = LiqssTest("liqss", x10=0.0, x20=20.0, dq1=dq, dq2=dq)

    sys.add_device(liqss)

    sys.initialize(dt=1e-2, dc=False)

    sys.run(500, ode=True, qss=True, verbose=2, ode_method="LSODA",
            print_interval=1)

    plotargs = {"plot_ode":True,
                "plot_qss":True,
                "plot_zoh":True,
                "plot_upd":False,
                "errorband":True,
                }

    if 1:
        sys.plot(liqss.x1, liqss.x2, **plotargs)

    if 0:
        plt.plot(liqss.x1.tzoh, liqss.x1.qzoh, "k-", label="$q_1(t)$", linewidth=1)
        plt.plot(liqss.x2.tzoh, liqss.x2.qzoh, "k-", label="$q_2(t)$", linewidth=1)
        plt.grid(linestyle="dotted", which="both")
        plt.xlabel("Time [sec]")
        plt.ylabel("$q_1(t)$, $q_2(t)$")
        plt.xlim((0, 500))
        plt.ylim((-5, 25))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
        plt.text(350, 22, "$q_1(t)$")
        plt.text(350, 2, "$q_2(t)$")
        plt.show()


if __name__ == "__main__":

    #test1()  # simple two atom RLCG system
    #test2()
    #test3()   # ship sys reduced # 1
    #test4()
    #test5()
    test6()  # coupled pendulums
    #test7()
    #test8()
    #test9()   # dc motor sys
    #test10()  # dc motor with cable
    #test11()  # converter test
    #test12()  # dc motor with converter
    #test13()  # Liqss 2nd order stiff system Test
