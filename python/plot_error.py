
import os
from glob import glob
import pickle
from math import sqrt
from matplotlib import pylab as plt
from collections import OrderedDict as odict

import numpy as np
from scipy.signal import butter, lfilter, freqz, filtfilt


_EPS = 1.0e-12

atom_pretty = {
"avr.vfd"    : "AVR $v_{fd}$ (V)"               ,
"avr.x1"     : "AVR x1 (p.u.)"                  ,
"avr.x2"     : "AVR x2 (p.u.)"                  ,
"avr.x3"     : "AVR x3 (p.u.)"                  ,         
"bus1.vd"    : "Bus 1 $v_d$ (V)"                ,
"bus1.vq"    : "Bus 1 $v_q$ (V)"                ,
"bus2.vd"    : "Bus 2 $v_d$ (V)"                ,
"bus2.vq"    : "Bus 2 $v_q$ (V)"                ,
"bus3.vd"    : "Bus 3 $v_d$ (V)"                ,
"bus3.vq"    : "Bus 3 $v_q$ (V)"                ,
"cable12.id" : "Cable 1-2 $i_d$ (A)"            ,
"cable12.iq" : "Cable 1-2 $i_q$ (A)"            ,
"cable13.id" : "Cable 1-3 $i_d$ (A)"            ,
"cable13.iq" : "Cable 1-3 $i_q$ (A)"            ,
"cable23.id" : "Cable 2-3 $i_d$ (A)"            ,
"cable23.iq" : "Cable 2-3 $i_q$ (A)"            ,
"im.idr"     : "Ind. Mach. $i_{dr}$ (A)"        ,
"im.ids"     : "Ind. Mach. $i_{ds}$ (A)"        ,
"im.iqr"     : "Ind. Mach. $i_{qr}$ (A)"        ,
"im.iqs"     : "Ind. Mach. $i_{qs}$ (A)"        ,
"im.wr"      : "Ind. Mach. $\omega_r$ (rad/s)"  ,
"load.id"    : "RL Load $i_d$ (A)"              ,
"load.iq"    : "RL Load $i_q$ (A)"              ,
"sm.ffd"     : "Sync. Mach. $\psi_{fd}$ (Wb)"   ,
"sm.fkd"     : "Sync. Mach. $\psi_{kd}$ (Wb)"   ,
"sm.fkq"     : "Sync. Mach. $\psi_{kq}$ (Wb)"   ,
"sm.ids"     : "Sync. Mach. $i_{ds}$ (A)"       ,
"sm.iqs"     : "Sync. Mach. $i_{qs}$ (A)"       ,
"sm.th"      : "Sync. Mach. $\theta$ (rad)"     ,
"sm.wr"      : "Sync. Mach. $\omega_r$ (rad/s)" ,
"trload.id"  : "Rect. Load $i_d$ (A)"           ,
"trload.vdc" : "Rect. Load $v_{dc}$ (V)"        ,
}
    

def get_error(typ, tout, qout, tode, xode):

    # interpolate qss to ss time vector:
    # this function can only be called after state space and qdl simualtions
    # are complete

    qout_interp = np.interp(tode, tout, qout)

    if typ.lower().strip() == "l2":

        # calculate the L^2 relative error:
        #      ________________
        #     / sum((y - q)^2)
        #    /  --------------
        #  \/      sum(y^2)

        dy_sqrd_sum = 0.0
        y_sqrd_sum = 0.0

        for q, y in zip(qout_interp, xout):
            dy_sqrd_sum += (y - q)**2
            y_sqrd_sum += y**2

        return sqrt(dy_sqrd_sum / y_sqrd_sum)

    elif typ.lower().strip() == "nrmsd":   # <--- this is what we're using

        # calculate the normalized relative root mean squared error:
        #      ________________
        #     / sum((y - q)^2) 
        #    /  ---------------
        #  \/          N
        # -----------------------
        #       max(y) - min(y)

        dy_sqrd_sum = 0.0
        y_sqrd_sum = 0.0

        for q, y in zip(qout_interp, xode):
            dy_sqrd_sum += (y - q)**2
            y_sqrd_sum += y**2

        return sqrt(dy_sqrd_sum / len(qout_interp)) / (max(xode) - min(xode))


    elif typ.lower().strip() == "re":

        # Pointwise relative error
        # e = [|(y - q)| / |y|] 

        e = []

        for q, y in zip(qout_interp, xode):
            e.append(abs(y-q) / abs(y))

        return e

    elif typ.lower().strip() == "rpd":

        # Pointwise relative percent difference
        # e = [ 100% * 2 * |y - q| / (|y| + |q|)] 

        e = []

        for q, y in zip(qout_interp, xode):
            den = abs(y) + abs(q)
            if den >= _EPS: 
                e.append(100 * 2 * abs(y-q) / (abs(y) + abs(q)))
            else:
                e.append(0)

        return e

    return None

scenario = "LOAD_INCREASE"

dir_ = f"E:\\School\\qdl\\{scenario}\\60s"
files = glob(dir_+"/*.pickle")

atoms = {}

for file_ in files:

    head, tail = os.path.split(file_)
    filename, ext = os.path.splitext(tail)
    device, atom, array = filename.split("_")
    atomname =  device+"."+atom 

    if not atomname in atoms:
       atoms[atomname] = {}

    atoms[atomname][array] = file_

atomnames = list(atoms.keys())


def plot_error(data=None, sort=True):

    print(f"Normalized RMS Deviation for each QSS atom for scenario {scenario}")

    errors = data

    if not errors:
    
        for atomname in atomnames[:]:

            with open (atoms[atomname]["tout"], "rb") as f: tout = pickle.load(f)
            with open (atoms[atomname]["qout"], "rb") as f: qout = pickle.load(f)
            with open (atoms[atomname]["tode"], "rb") as f: tode = pickle.load(f)
            with open (atoms[atomname]["xode"], "rb") as f: xode = pickle.load(f)

            #error = get_error("re", tout, qout, tode, xode)
            #plt.plot(tode, error, label=atomname)
            #plt.title(f"Relative error {atomname}")
            #plt.show()

            error = get_error("nrmsd", tout, qout, tode, xode)

            errors[atomname] = error
            
            print("{:12s}{:12.4f} p.u.".format(atomname, error))

    if sort:
        errors_sorted = {k: v for k, v in sorted(errors.items(), key=lambda item: item[1])}

    else:
        errors_sorted = errors
        
    error_atomnames = []
    error_values = []

    for k, v in errors_sorted.items():
        error_atomnames.append(atom_pretty[k])
        error_values.append(v * 100.0)

    ax = plt.gca()


    ax.grid(which='major', axis='x')

    #ax.grid(which='major', color='#1f1f1f', linewidth=1.2)
    #ax.grid(which='minor', color='#3f3f3f', linewidth=0.6)
    #ax.minorticks_on()

    ax.tick_params(which='minor', bottom=True, left=False, top=False, right=False)

    plt.barh(error_atomnames, error_values) 

    plt.ylabel("State variable")
    plt.xlabel("Normalized RMS deviation (%)")

    plt.xlim(0.01, 10.0)

    plt.tight_layout()

    plt.show()


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filtfilt(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def plot_data(atomname, ylabel=None, order=6, filter_cutoff=None, step_qss=False, linewidth=1.0, report_error=False):

    with open (atoms[atomname]["tout"], "rb") as f: tout = pickle.load(f)
    with open (atoms[atomname]["qout"], "rb") as f: qout = pickle.load(f)
    with open (atoms[atomname]["tode"], "rb") as f: tode = pickle.load(f)
    with open (atoms[atomname]["xode"], "rb") as f: xode = pickle.load(f)


    qout_use = []
    tout_use = []

    if step_qss:

        q0 = qout[0]

        for t, q in zip(tout, qout):
            qout_use.append(q0)
            qout_use.append(q)
            tout_use.append(t)
            tout_use.append(t)
            q0 = q

    else:
        qout_use = qout
        tout_use = tout

    plt.plot(tout_use, qout_use, "b-", linewidth=linewidth, label="qss")

    if filter_cutoff:

        qout_interp = np.interp(tode, tout, qout)
        fs = len(tode) / 60.0
        qout_filt = butter_lowpass_filtfilt(qout_interp, filter_cutoff, fs, order)
        tout_filt = tode

        plt.plot(tout_filt, qout_filt, "r-", linewidth=linewidth*2.0, label="qss filtered ($f_c$={} Hz)".format(filter_cutoff))

    plt.plot(tode, xode, "k--", linewidth=linewidth*1.2, label="ode")

    plt.legend(loc="lower right")

    if ylabel: plt.ylabel(ylabel)

    plt.xlabel("t (s)")

    ax = plt.gca()

    ax.minorticks_on()

    ax.grid(which='major', linewidth=0.50)
    ax.grid(which='minor', linewidth=0.25)

    fig = plt.gcf()
    fig.set_size_inches(14.0, 6.0)

    plt.show()

    if report_error:
        print("{:12s}{:12.4f} p.u.".format(atomname, get_error("nrmsd", tout_use, qout_use, tode, xode)))
        if filter_cutoff:
            print("{:12s}{:12.4f} p.u. (filtered)".format(atomname, get_error("nrmsd", tout_filt, qout_filt, tode, xode)))

#plot_data("sm.wr",      title="Synchronous Machine Speed",        ylabel="$\omega_r$ (rad/s)") #, filter_cutoff=2.0)
#plot_data("cable23.iq", title="Cable 2-3 q-axis Current",         ylabel="$i_q$ (A)",          filter_cutoff=50.0)
#plot_data("trload.vdc", title="Transformer-Rectifier DC Voltage", ylabel="$v_{dc}$ (V)",       filter_cutoff=200.0)

#plot_data("sm.wr", title="", ylabel="", step_qss=True)

#best error:
#plot_data("load.iq", ylabel=atom_pretty["load.iq"], step_qss=True, linewidth=1.0)

#worst error:
plot_data("cable23.iq", ylabel=atom_pretty["cable23.iq"], step_qss=True, linewidth=1.0, filter_cutoff=50.0, report_error=True)

nrmsd_errors = odict((
("avr.vfd   ", 0.0359),
("avr.x1    ", 0.0486),
("avr.x2    ", 0.0526),
("avr.x3    ", 0.0486),
("bus1.vd   ", 0.0393),
("bus1.vq   ", 0.0196),
("bus2.vd   ", 0.0390),
("bus2.vq   ", 0.0196),
("bus3.vd   ", 0.0391),
("bus3.vq   ", 0.0194),
("cable12.id", 0.0448),
("cable12.iq", 0.0637),
("cable13.id", 0.0337),
("cable13.iq", 0.0482),
("cable23.id", 0.0466),
("cable23.iq", 0.0927),
("im.idr    ", 0.0125),
("im.ids    ", 0.0134),
("im.iqr    ", 0.0077),
("im.iqs    ", 0.0074),
("im.wr     ", 0.0071),
("load.id   ", 0.0033),
("load.iq   ", 0.0023),
("sm.ffd    ", 0.0363),
("sm.fkd    ", 0.0324),
("sm.fkq    ", 0.0196),
("sm.ids    ", 0.0270),
("sm.iqs    ", 0.0242),
("sm.th     ", 0.0196),
("sm.wr     ", 0.0308),
("trload.id ", 0.0316),
("trload.vdc", 0.0379),
))

#plot_error(data=nrmsd_errors)



