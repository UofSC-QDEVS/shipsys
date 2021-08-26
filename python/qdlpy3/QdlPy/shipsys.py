
import os
import time

from qdl import *
from models import *


scenarios = [
#"IM_START",
"FAULT_BUS2",
#"LOAD_INCREASE",
#"VREF_INCREASE",
]

tstop = 1.08

outdir = r"D:\School\qdl"
#outdir = r"C:\Temp\qdl"

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

for scenario in scenarios:

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

    if scenario == "IM_START":
        im.connect(gnd)
        im.wr0 = 0.0
    else:
        im.connect(bus2)
        bus2.connect(im, terminal="i")

    load.connect(bus3, gnd)
    bus3.connect(load, terminal="i")

    trload.connect(bus3)
    bus3.connect(trload, terminal="i")

    dt = 1.0e-1
    dc = 1
    optimize_dq = 0
    ode = 1
    qss = 1

    def increase_load(sys):
        load.r *= 0.8

    def apply_fault(sys):
        bus2.g = 1e3

    def clear_fault(sys):
        bus2.g = 1e-4

    def trip_line(sys):
        cable12.l = 1e6

    def vref_increase(sys):
        avr.vref = 1.02

    def im_start(sys):
        im.connect(bus2)
        bus2.connect(im, terminal="i")

    if scenario == "LOAD_INCREASE":
        sys.schedule(increase_load, 1.0)

    if scenario == "FAULT_BUS2":
        sys.schedule(apply_fault, 1.0)
        sys.schedule(clear_fault, 1.0 + 5.0/60.0)
        #sys.schedule(trip_line, 1.0 + 5.0/60.0)

    if scenario == "VREF_INCREASE":  # SM vset change simulation:
        sys.schedule(vref_increase, 1.0)

    if scenario == "IM_START":
        sys.schedule(im_start, 1.0)

    sys.Km = 2.0

    sys.storage_type = StorageType.DEQUE

    sys.initialize(dt=dt, dc=dc, savedt=1e-9, tocsv=False, outdir=outdir)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="Radau",
            optimize_dq=optimize_dq)

    plotargs = {"plot_ode":ode,
                "plot_qss":qss,
                "plot_zoh":qss,
                "plot_upd":False,
                "errorband":False}

    #sys.plot(sm.wr, **plotargs)

    for atom in sys.state_atoms:

        print(f"writing {atom.device.name}.{atom.name} data to file...")

        dirpth = os.path.join(outdir, scenario, f"{int(tstop)}s")

        try:
            os.makedirs(dirpth)
        except:
            pass

        pth = os.path.join(dirpth, f"{atom.device.name}_{atom.name}")

        pth_tout = pth + "_tout.pickle"
        pth_qout = pth + "_qout.pickle"
        pth_tode = pth + "_tode.pickle"
        pth_xode = pth + "_xode.pickle"

        #pth_tzoh = pth + "_tzoh.pickle"
        #pth_qzoh = pth + "_qzoh.pickle"
        #pth_nupd = pth + "_nupd.pickle"

        with open(pth_tout, 'wb') as f: pickle.dump(atom.tout, f)
        with open(pth_qout, 'wb') as f: pickle.dump(atom.qout, f)
        with open(pth_tode, 'wb') as f: pickle.dump(atom.tode, f)
        with open(pth_xode, 'wb') as f: pickle.dump(atom.xode, f)

        #with open(pth_tzoh, 'wb') as f: pickle.dump(atom.tzoh, f)
        #with open(pth_qzoh, 'wb') as f: pickle.dump(atom.qzoh, f)
        #with open(pth_nupd, 'wb') as f: pickle.dump(atom.nupd, f)

    print("\nDONE.")

