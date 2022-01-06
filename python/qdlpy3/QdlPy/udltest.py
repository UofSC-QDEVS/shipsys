
from qdl import *
from models import *



def test1():

    sys = System(dq=1e-3)

    pendulum = Pendulum2("pendulum", mu=0.5, l=10.0, a0=1.0, w0=1.0,
                         dq_w=4e-2, dq_a=1e-2)

    sys.add_devices(pendulum)

    tstop = 60.0
    dt = 1.0e-3
    dc = 0
    optimize_dq = 1
    ode = 1 
    qss = 1

    chk_ss_delay = 10.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        pendulum.w.set_state(0.0)
        pendulum.a.set_state(-pi/4, quantize=True)

    #sys.schedule(event, tstop*0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":False, "errorband":True}

    sys.plot(pendulum.w, pendulum.a, **plotargs)

    sys.plotxy(pendulum.a, pendulum.w, arrows=False, ss_region=True)


def test2():

    sys = System(dq=1e-3)

    node = LimNode2("node", c=1e-3, g=2.0, h=0.0, v0=0.0, dq=1e-2)
    ground = LimNode2("ground", c=1e-3, g=1e3, h=0.0, v0=0.0, dq=1e-2)
    branch = LimBranch2("branch", l=1e-3, r=0.1, e=10.0, i0=0.0, dq=1e-2)

    sys.connect(branch.positive, node.positive)
    sys.connect(branch.negative, ground.positive)

    sys.add_devices(node, ground, branch)

    tstop = 0.1
    dt = 1.0e-3
    dc = 1
    optimize_dq = 0
    ode = 1 
    qss = 1

    chk_ss_delay = 10.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event1(sys):
        node.update_parameter("g", 1.0)

    def event2(sys):
        branch.update_parameter("e", 5.0)

    sys.schedule(event1, tstop*0.2)
    sys.schedule(event2, tstop*0.6)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":False, "errorband":True}

    sys.plot(node.v, branch.i, **plotargs)


def test3():

    sys = System(dq=1e-3)

    ws = 2*pi*60
    theta = 0.5

    dq_v = 1e0
    dq_i = 1e0

    node = LimNodeDQ2("node", c=1e-3, g=2.0, ws=ws, theta=theta, h=0.0, dq=dq_v)

    ground = LimNodeDQ2("ground", c=1e-3, g=1e3, ws=ws, theta=theta, h=0.0, dq=dq_v)

    branch = LimBranchDQ2("branch", l=1e-3, r=0.1, ws=ws, theta=theta, e=10.0, dq=dq_i)

    sys.connectdq(branch.positive, node.positive)
    sys.connectdq(branch.negative, ground.positive)

    sys.add_devices(node, ground, branch)

    tstop = 0.05
    dt = 1.0e-3
    dc = 1
    optimize_dq = 1
    ode = 1 
    qss = 1

    chk_ss_delay = 10.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event1(sys):
        node.update_parameter("g", 1.0)

    def event2(sys):
        branch.update_parameter("e", 5.0)

    sys.schedule(event1, tstop*0.2)
    sys.schedule(event2, tstop*0.6)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":False, "errorband":True}

    sys.plot(node.vd, node.vq, **plotargs)
    sys.plot(branch.id, branch.iq, **plotargs)


def test4():

    ws = 2*pi*60
    VLL = 4160.0  # bus nominal line-to-line rms voltage
    theta = 0.0

    sys = System(dq=1e-3)

    im = InductionMachineDQ2("im", ws=ws, dq_i=1e-2, dq_wr=1e-1)
    bus = LimNodeDQ2("bus", c=1e-3, g=0.0, ws=ws, theta=theta, h=1e3, vd0=VLL, vq0=VLL, dq=1e-1)
    source = LimBranchDQ2("source", l=2e-3, r=0.1, ws=ws, theta=theta, e=VLL, dq=1e-2)
    ground = LimNodeDQ2("ground", c=1e-3, g=0, ws=ws, theta=theta, h=0.0, dq=1e-2)

    sys.connectdq(im.terminal, bus.positive)
    sys.connectdq(source.positive, bus.positive)
    sys.connectdq(source.negative, ground.positive)

    sys.add_devices(im, bus, source, ground)

    tstop = 0.05
    dt = 1.0e-3
    dc = 1
    optimize_dq = 0
    ode = 1 
    qss = 0
    chk_ss_delay = 10.0
    chk_ss_delay = None

    sys.initialize(dt=dt, dc=dc)

    def event(sys):
        source.update_parameter("r", 0.2)

    sys.schedule(event, tstop*0.5)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":False, "errorband":True}

    sys.plot(im.wr, **plotargs)
    sys.plot(im.ids, im.iqs, **plotargs)
    sys.plot(bus.vd, bus.vq, **plotargs)


def test5():

    ws = 376.991
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

    sm = SyncMachineDQ2("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v, dq_th=dq_th, dq_wr=dq_wr, dq_f=dq_f)
    avr = AC8B("avr", VLL=VLL, vref=vref, dq_x1=dq_x1, dq_x2=dq_x2, dq_x3=dq_x3, dq_vfd=dq_vfd)
    bus1 = LimNodeDQ2("bus1", c=1e-3, g=1e-4, ws=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus2 = LimNodeDQ2("bus2", c=1e-3, g=1e-4, ws=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    cable = LimBranchDQ2("cable", l=2.3e-5, r=0.865e-2, ws=ws, dq=dq_i)
    load = LimBranchDQ2("load", l=3.6e-3, r=2.8, ws=ws, dq=dq_i)
    im = InductionMachineDQ2("im", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i/10, dq_wr=dq_wr, Tb=0.0)
    gnd = LimNodeDQ2("gnd", c=1e-3, g=0.0, ws=ws, vd0=0.0, vq0=0.0, dq=dq_v)

    sys.connect(sm.terminal, bus1.positive)

    sys.connect(sm.vterm, avr.vterm)
    sys.connect(avr.vfd, sm.vfd)

    sys.connect(cable.positive, bus1.positive)
    sys.connect(cable.negative, bus2.positive)
    sys.connect(load.positive, bus2.positive)
    sys.connect(load.negative, gnd.positive)
    sys.connect(im.terminal, bus2.positive)

    sys.add_devices(sm, bus1, cable, bus2, load, im, gnd)

    tstop = 0.01
    dt = 1.0e-4
    dc = 1
    optimize_dq = 0
    ode = 1 
    qss = 1

    chk_ss_delay = 1.0
    chk_ss_delay = None

    def event(sys):
        load.update_parameter("r", 2.6)

    sys.schedule(event, tstop*0.2)

    sys.Km = 2.0

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
            optimize_dq=optimize_dq, chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":True, "errorband":True}

    sys.plot(sm.wr,     im.wr,     sm.th,  **plotargs) # , pth=r".\plots\sm.pdf")
    sys.plot(avr.x1,    avr.x2,    avr.x3, **plotargs) # , pth=r".\plots\avr.pdf")
    sys.plot(cable.id,  cable.iq,          **plotargs) # , pth=r".\plots\cable.pdf")
    sys.plot(bus1.vd,   bus1.vq,           **plotargs) # , pth=r".\plots\bus1.pdf")
    sys.plot(bus2.vd,   bus2.vq,           **plotargs) # , pth=r".\plots\bus2.pdf")
    sys.plot(trload.id, trload.iq,         **plotargs) # , pth=r".\plots\tr.pdf")
    sys.plot(load.id,   load.iq,           **plotargs) # , pth=r".\plots\load.pdf")

    #sys.plotxy(sm.wr, sm.th, arrows=False, ss_region=True, auto_limits=True)


if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    #test4()
    test5()