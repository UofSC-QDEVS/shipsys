
import os
import time
from glob import glob
import multiprocessing as mp
import threading as th

from collections import deque

from qdl import *
from models import *

OUTDIR = r"C:\School\qdl"

scenario = "FAULT_BUS2_MP"


def run(tstart, tstop, start_data_path, stop_data_path):

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

    im.connect(bus2)
    bus2.connect(im, terminal="i")

    load.connect(bus3, gnd)
    bus3.connect(load, terminal="i")

    trload.connect(bus3)
    bus3.connect(trload, terminal="i")

    dt = 1e-4
    dc = 1
    optimize_dq = 0
    ode = 0
    qss = 1

    def apply_fault(sys):
        bus2.g = 1e3

    def clear_fault(sys):
        bus2.g = 1e-4

    def trip_line(sys):
        cable12.l = 1e6

    sys.schedule(apply_fault, 1.0)
    sys.schedule(clear_fault, 1.0 + 5.0/60.0)
    #sys.schedule(trip_line, 1.0 + 5.0/60.0)

    sys.Km = 2.0

    sys.storage_type = StorageType.DEQUE

    sys.print_states()

    if tstart == 0.0:
        sys.initialize(t0=tstart, dt=dt, dc=dc, savedt=1e-7, tocsv=False, outdir=OUTDIR)
    else:
        sys.state_from_file(os.path.join(OUTDIR, scenario, start_data_path))
        
    sys.clear_data_arrays()
  
    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="Radau", optimize_dq=optimize_dq)

    sys.print_states()
    sys.state_to_file(os.path.join(OUTDIR, scenario, stop_data_path))

    for atom in sys.state_atoms:

        fullname =f"{atom.device.name}_{atom.name}"

        time_ms = f"{int(tstop*1000)}"

        print(f"writing {fullname} data to file up to time {time_ms}...")

        dirpth = os.path.join(OUTDIR, scenario)

        try:
            os.makedirs(dirpth)
        except:
            pass

        pth = os.path.join(dirpth, f"{fullname}_{time_ms}")

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


if __name__ == "__main__":

    #with open(r"C:\School\qdl\FAULT_BUS2_MP\state_900.pickle", 'rb') as f:
    #    d = pickle.load(f)
    #
    #for a in d["atoms"].values():
    #    print(a["x"])

    times = [0.0, 0.9, 1.1, 1.5, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    for i in range(1, len(times)):
        p = th.Thread(target=run, args=(times[i-1], times[i],
                                        f"state_{int(times[i-1]*1000)}.pickle",
                                        f"state_{int(times[i]*1000)}.pickle"))
        p.start()
        p.join()



