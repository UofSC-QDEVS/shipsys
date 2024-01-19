
import os
import time
from glob import glob
import multiprocessing as mp

from collections import deque

from qdl import *
from models import *

OUTDIR = r"/work/navidg/shipsys"

def run():

    scenarios = [
    #"IM_START",
    #"LOAD_INCREASE",
    "VREF_INCREASE",
    #"FAULT_BUS2",
    ]

    tstop = 1.1
    
    ws = 2*60*PI
    vfdb = 90.1
    VLL = 4160.0
    vref = 1.0

    dq_i = 1e-2
    dq_v = 1e-2
    dq_wr = 1e-3
    dq_th = 1e-4
    dq_f = 1e-4
    dq_x1 = 1e-8
    dq_x2 = 1e-9
    dq_x3 = 1e-5
    dq_vfd = 1e-3
    R= 3.16
    Rs=31.8e-3
    r=0.865e-2
    divisor=5

    for scenario in scenarios:

        #ship1
        sys = System(qss_method=QssMethod.LIQSS1)

        sm1 = SyncMachineDQ("sm1", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v,
                            dq_th=dq_th, dq_wr=dq_wr, dq_f=dq_f)

        avr1 = AC8B("avr1", vref=vref, dq_x1=dq_x1, dq_x2=dq_x2, dq_x3=dq_x3,
                   dq_vfd=dq_vfd)

        bus1 = LimNodeDQ("bus1", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        bus2 = LimNodeDQ("bus2", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        bus3 = LimNodeDQ("bus3", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        cable12 = LimBranchDQ("cable12", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        cable13 = LimBranchDQ("cable13", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        cable23 = LimBranchDQ("cable23", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        load1 = LimBranchDQ("load1", l=3.6e-3, r=2.8/divisor, w=ws, dq=dq_i)
        trload1 = TRLoadDQ2("trload1", w=ws, vdc0=VLL, dq_i=dq_i/10, dq_v=dq_v/10,R=R/divisor)
        im1 = InductionMachineDQ("im1", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i/10, dq_wr=dq_wr, Tb=0.0, Rs=Rs/divisor)

        #ship2

        bus4 = LimNodeDQ("bus4", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        bus5 = LimNodeDQ("bus5", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        bus6 = LimNodeDQ("bus6", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
        cable45 = LimBranchDQ("cable45", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        cable46 = LimBranchDQ("cable46", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        cable56 = LimBranchDQ("cable56", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)
        load2 = LimBranchDQ("load2", l=3.6e-3, r=2.8/divisor, w=ws, dq=dq_i)
        trload2 = TRLoadDQ2("trload2", w=ws, vdc0=VLL, dq_i=dq_i/10, dq_v=dq_v/10,R=R/divisor)
        im2 = InductionMachineDQ("im2", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i/10, dq_wr=dq_wr, Tb=0.0, Rs=Rs/divisor)

        cable14 = LimBranchDQ("cable14", l=2.3e-5, r=r/divisor, w=ws, dq=dq_i)

        gnd = GroundNodeDQ("gnd")

        sys.add_devices(sm1, avr1, bus1, bus2, bus3, cable12, cable13, cable23, load1, trload1, im1, bus4, bus5, bus6, cable45, 
            cable46, cable56, cable14, load2, trload2, im2, gnd)

        sm1.connect(bus1, avr1)
        avr1.connect(bus1, sm1)
        bus1.connect(sm1, terminal="j")

        bus1.connect(cable12, terminal="i")
        cable12.connect(bus1, bus2)
        bus2.connect(cable12, terminal="j")

        bus1.connect(cable13, terminal="i")
        cable13.connect(bus1, bus3)
        bus3.connect(cable13, terminal="j")

        bus2.connect(cable23, terminal="i")
        cable23.connect(bus2, bus3)
        bus3.connect(cable23, terminal="j")


        bus4.connect(cable45, terminal="i")
        cable45.connect(bus4, bus5)
        bus5.connect(cable45, terminal="j")

        bus4.connect(cable46, terminal="i")
        cable46.connect(bus4, bus6)
        bus6.connect(cable46, terminal="j")

        bus5.connect(cable56, terminal="i")
        cable56.connect(bus5, bus6)
        bus6.connect(cable56, terminal="j")

        bus1.connect(cable14, terminal="i")
        cable14.connect(bus1, bus4)
        bus4.connect(cable14, terminal="j")

        if scenario == "IM_START":
            im.connect(gnd)
            im.wr0 = 0.0
        else:
            im1.connect(bus2)
            bus2.connect(im1, terminal="i")
            im2.connect(bus5)
            bus5.connect(im2, terminal="i")

        load1.connect(bus3, gnd)
        bus3.connect(load1, terminal="i")

        trload1.connect(bus3)
        bus3.connect(trload1, terminal="i")

        load2.connect(bus6, gnd)
        bus6.connect(load2, terminal="i")

        trload2.connect(bus6)
        bus6.connect(trload2, terminal="i")

        dt = 1e-4
        dc = 1
        optimize_dq = 0
        ode = 0
        qss = 1

        plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                    "plot_upd":False, "errorband":False}

        def increase_load(sys):
            load.r *= 0.999

        def apply_fault(sys):
            bus2.g = 1e3

        def clear_fault(sys):
            bus2.g = 1e-4

        def trip_line(sys):
            cable12.l = 1e6

        def vref_increase(sys):
            avr1.vref = 1.02

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

        sys.initialize(dt=dt, dc=dc, savedt=1e-7, tocsv=False, outdir=OUTDIR)

        tstops = [0.0, 4.0]

        sys.state_to_file(os.path.join(OUTDIR, scenario, f"scaling/2shipsconnected/state_{int(tstops[0]*1000)}.pickle"))

        for i, tstop in enumerate(tstops[1:]):

            the_path=os.path.join(OUTDIR, scenario, f"scaling/2shipsconnected/state_{int(tstops[i]*1000)}.pickle")

            sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="Radau",
                    optimize_dq=optimize_dq)

            for atom in sys.state_atoms:

                fullname =f"{atom.device.name}_{atom.name}"

                time_ms = f""

                print(f"writing {fullname} data to file up to time {time_ms}...")

                dirpth = os.path.join(OUTDIR, scenario)+'/scaling/2shipsconnected'

                try:
                    os.makedirs(dirpth)
                except:
                    pass

                pth = os.path.join(dirpth, f"{fullname}")

                if qss:
                    pth_tout = pth + "_tout.pickle"
                    pth_qout = pth + "_qout.pickle"
                    with open(pth_tout, 'wb') as f: pickle.dump(atom.tout, f)
                    with open(pth_qout, 'wb') as f: pickle.dump(atom.qout, f)

                if ode:
                    pth_tode = pth + "_tode.pickle"
                    pth_xode = pth + "_xode.pickle"
                    with open(pth_tode, 'wb') as f: pickle.dump(atom.tode, f)
                    with open(pth_xode, 'wb') as f: pickle.dump(atom.xode, f)

            print(f"\nDONE for time segment to time {time_ms} ms.")

            sys.state_to_file(os.path.join(OUTDIR, scenario, f"scaling/2shipsconnected/state_{int(tstops[0]*1000)}.pickle"))

        print("\nALL DONE.")


def combine_output():

    atompaths = {}
    
    paths = glob(os.path.join(OUTDIR, "FAULT_BUS2", "*.pickle"))

    for path in paths:
        dir_, filepath = os.path.split(path)
        filename, ext = os.path.splitext(filepath)
        fields = filename.split("_")
        if not len(fields) > 3: continue
        atom, state, ms, array = filename.split("_")
        atom_state = f"{atom}_{state}"
        if not atom_state in atompaths:
            atompaths[atom_state] = []
        atompaths[atom_state].append(path)

    for atom_state, paths in atompaths.items():

        data = deque()

        for path in paths:
            with open(path, "rb") as f:
                section = pickle.load(f)
            data.extend(section)

        dir_, filepath = os.path.split(path)
        filename, ext = os.path.splitext(filepath)
        atom, state, ms, array = filename.split("_")

        newpath = os.path.join(dir_, f"{atom}_{state}_{array}.pickle")

        with open(newpath, "wb") as f:
            pickle.dump(data, f)

run()

combine_output()

