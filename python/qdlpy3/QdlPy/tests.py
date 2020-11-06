
from qdl import *
from models import *     


def test1():

    sys = System(dq=1e-3)

    ground = GroundNode("ground")
    node1 = LimNode("node1", 1.0, 1.0, 1.0)
    branch1 = LimBranch("branch1", 1.0, 1.0, 1.0)

    sys.add_devices(ground, node1, branch1)

    branch1.connect(ground, node1)
    node1.connect(branch1, terminal='j')

    sys.init_qss()
    sys.run_qss(10.0)

    sys.plot()


def test2():

    w = 2*pi*60
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
                "plot_qss_updates":True, "upd_bins":1000, "errorband":True,
                "legloc":"lower right"}

    sys.plot(node1.v, node2.v, node3.v, **plotargs)
    sys.plot(node4.v, node5.v, **plotargs)
    sys.plot(branch1.i, branch2.i, branch3.i, **plotargs)
    sys.plot(branch4.i, branch5.i, branch6.i, **plotargs)


def test3():

    ws = 2*60*pi 
    vfdb = 90.1  
    VLL = 4160.0 
    vref = 1.0   
    dq_i = 1e-1
    dq_v = 1e-1
    dq_wr = 1e-4
    dq_th = 1e-4
    dq_f = 1e-3
    dq_x1 = 1e-9
    dq_x2 = 1e-8
    dq_x3 = 1e-5
    dq_vfd = 1e-2

    sys = System(dq=1e-3)

    sm = SyncMachineDQ("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v,
                        dq_th=dq_th, dq_wr=dq_wr, dq_f=dq_f)

    avr = AC8B("avr", vref=vref, dq_x1=dq_x1, dq_x2=dq_x2, dq_x3=dq_x3,
               dq_vfd=dq_vfd)

    bus1 = LimNodeDQ("bus1", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus2 = LimNodeDQ("bus2", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    cable = LimBranchDQ("cable", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=dq_i)
    trload = TRLoadDQ2("trload", w=ws, vdc0=VLL, dq_i=dq_i/10, dq_v=dq_v/10)
    im = InductionMachineDQ("im", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i/10, dq_wr=dq_wr, Tb=0.0)
    gnd = GroundNodeDQ("gnd")

    sys.add_devices(sm, avr, bus1, cable, bus2, load, trload, im, gnd)

    sm.connect(bus1, avr)
    avr.connect(bus1, sm)
    bus1.connect(sm, terminal="j")

    bus1.connect(cable, terminal="i")
    cable.connect(bus1, bus2)
    bus2.connect(cable, terminal="j")

    load.connect(bus2, gnd)
    bus2.connect(load, terminal="i")

    trload.connect(bus2)
    bus2.connect(trload, terminal="i")

    im.connect(bus2)
    bus2.connect(im, terminal="i")

    tstop = 0.1
    dt = 1.0e-4
    dc = 1
    optimize_dq = 0
    ode = 1 
    qss = 1

    chk_ss_delay = 1.0
    chk_ss_delay = None

    def event(sys):
        load.r *= 0.8

    sys.schedule(event, tstop*0.2)

    sys.Km = 2.0

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":True, "errorband":True}

    sys.plot(sm.wr, im.wr, sm.th, **plotargs,     pth=r".\plots\sm.pdf")
    sys.plot(avr.x1, avr.x2, avr.x3, **plotargs,  pth=r".\plots\avr.pdf")
    sys.plot(cable.id, cable.iq,  **plotargs,     pth=r".\plots\cable.pdf")
    sys.plot(bus1.vd, bus1.vq, **plotargs,        pth=r".\plots\bus1.pdf")
    sys.plot(bus2.vd, bus2.vq, **plotargs,        pth=r".\plots\bus2.pdf")
    sys.plot(trload.id, trload.iq, **plotargs,    pth=r".\plots\tr.pdf")
    sys.plot(load.id, load.iq, **plotargs,        pth=r".\plots\load.pdf")

    #sys.plotxy(sm.wr, sm.th, arrows=False, ss_region=True, auto_limits=True)
    

def test4():

    ws = 2*60*pi  # system radian frequency
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

    tstop = 1.0

    dt = 1.0e-3
    dc = True

    ode = True 
    qss = False
    upds = False
    upd_bins = 1000

    chk_ss = True
    chk_ndq = 10
    chk_dmax = 10

    plotargs = {
    "plot_ode":ode,
    "plot_qss":qss,
    "plot_qss_updates":upds,
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

    pendulum = Pendulum("pendulum", r=0.4, l=8.0, theta0=pi/4, omega0=0.0,
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
        pendulum.theta.set_state(-pi/4, quantize=True)

    #sys.schedule(event, tstop*0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":qss, "upd_bins":1000, "errorband":True}

    sys.plot(pendulum.omega, pendulum.theta, **plotargs)

    sys.plotxy(pendulum.theta, pendulum.omega, arrows=False, ss_region=True)


def test6():

    sys = System(dq=1e-3)

    pendulums = CoupledPendulums("pendulums", k=1.0, r1=1.0, r2=1.0,
                                 l1=1.0, l2=1.0, m1=1.0, m2=1.0,
                                 th10=pi/4, w10=0.0, th20=-pi/4, w20=0.0,
                                 dq_w=1e-1, dq_th=1e-1)

    sys.add_devices(pendulums)

    tstop = 100.0
    dt = 1.0e-3
    dc = 0
    optimize_dq = 1
    ode = 1 
    qss = 1

    chk_ss_delay = 5.0
    #chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        pendulums.th1.set_state(pi/4)

    sys.schedule(event, tstop*0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":qss, "errorband":True}

    #sys.plotxy(pendulums.w1, pendulums.th1, arrows=False, ss_region=True)
    #sys.plotxy(pendulums.w2, pendulums.th2, arrows=False, ss_region=True)

    #sys.plotxyt(pendulums.w1, pendulums.th1, ss_region=True)
    #sys.plotxyt(pendulums.w2, pendulums.th2, ss_region=True)

    sys.plot(pendulums.w1, pendulums.w2, **plotargs)
    sys.plot(pendulums.th1, pendulums.th2, **plotargs)


if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    test6()
