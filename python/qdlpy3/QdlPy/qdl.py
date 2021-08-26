"""Quantized DEVS-LIM modeling and simulation framework.
"""

_SYMPY = False

import os

if _SYMPY:
    import sympy as sp
    from sympy.solvers import solve
    from sympy.utilities.lambdify import lambdify, implemented_function
    from sympy import sin, cos, tan, atan2, acos, pi, sqrt

from glob import glob
import time
import pickle

import threading as th
import queue

# numerical math constants:

from math import pi    as PI
from math import sin   as SIN
from math import cos   as COS
from math import acos  as ACOS
from math import tan   as TAN
from math import acos  as ACOS
from math import atan2 as ATAN2
from math import sqrt  as SQRT
from math import floor as FLOOR
from math import ceil  as CEIL
from math import isclose as ISCLOSE

# temporary:

from math import pi, sin, cos, acos, tan, acos, atan2, sqrt, floor as FLOOR

from collections import deque
from collections import OrderedDict as odict
from array import array

import pandas as pd

import numpy as np
import numpy.linalg as la

from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes.formatter', useoffset=False)

from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

import lti


# ============================ Private Constants ===============================


_EPS = 1.0e-12
_INF = float('inf')
_MAXITER = 1000


# ============================ Public Constants ================================


DEF_DTMIN = 1e-12        # default minimum time step
DEF_DMAX = 1.0e5       # default maximum derivative (slew-rate)

MIN_DT_SAVE = 1.0e-15   # min time between saves (defines max time resolution)

PI_4 = float(pi / 4.0)
PI_3 = float(pi / 3.0)
PI5_6 = float(5.0 * pi / 6.0)
PI7_6 = float(7.0 * pi / 6.0)


# ============================= Enumerations ===================================


class StorageType:

    LIST = "LIST"
    ARRAY = "ARRAY"
    DEQUE = "DEQUE"


class SourceType:

    NONE = "NONE"
    CONSTANT = "CONSTANT"
    STEP = "STEP"
    SINE = "SINE"
    PWM = "PWM"
    RAMP = "RAMP"
    FUNCTION = "FUNCTION"


class OdeMethod:

    RK45 = "RK45"
    RK23 = "RK23"
    DOP853 = "DOP853"
    RADAU = "Radau"
    BDF = "BDF"
    LSODA = "LSODA"


class QssMethod:

    QSS1 = "QSS1"
    QSS2 = "QSS2"
    LIQSS1 = "LIQSS1"
    LIQSS2 = "LIQSS2"
    mLIQSS1 = "mLIQSS1"
    mLIQSS2 = "mLIQSS2"


# =============================== Globals ======================================


sys = None  # set by qdl.System constructor for visibility from fode function.
simtime = 0.0
simlock = th.Lock()


# ========================== Utility Funrctions ================================


def print_matrix_dots(m):
    s = ""
    for i in m.size(0):
        for j in m.size(1):
            if m[i,j]:
                s += "x"
            else:
                s += " "
        s += "\n"
    print(s)


# ============================= Qdl Model ======================================


class Atom(object):

    def __init__(self, name, x0=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=1e10, units="", qss_method=None):

        # params:

        self.name = name
        self.x0 = x0  # initial value of state
        self.dq = dq  # quantization step size

        self.dqmin = dqmin
        self.dqmax = dqmax
        self.dqerr = dqerr

        self.dtmin = dtmin

        self.dmax = dmax

        self.units = units

        self.qss_method = qss_method  # type QssMethod

        # simulation variables:

        # quantization step:

        self.dq0 = self.dq
        self.qlo = 0.0
        self.qhi = 0.0
        self.qdlo = 0.0
        self.qdhi = 0.0

        # time:

        self.tlast = 0.0    # last update time of internal state x
        self.tlastq = 0.0   # last update time of quantized state q
        self.tnext = 0.0    # next time to quantized state change

        # internal state:

        self.x = x0
        self.xlast = x0

        # internal state derivatives:

        self.dx = 0.0
        self.dxlast = 0.0
        self.ddx = 0.0
        self.ddxlast = 0.0

        # quantized state:

        self.q = x0
        self.qlast = x0
        self.qd = x0
        self.qdlast = x0

        self.a = 0.0
        self.u = 0.0
        self.du = 0.0

        self.triggered = False
        self.saveclock = 0.0

        # results data storage:

        # qss:
        self.tout = None  # output times quantized output
        self.qout = None  # quantized output
        self.tzoh = None  # zero-order hold output times quantized output
        self.qzoh = None  # zero-order hold quantized output
        self.evals = 0    # qss evals
        self.updates = 0  # qss updates

        # state space:
        self.tout_ss = None  # state space time output
        self.xout_ss = None  # state space value output
        self.updates_ss = 0  # state space update count

        # non-linear ode:
        self.tode = None  # state space time output
        self.xode = None  # state space value output
        self.updates_ode = 0  # state space update count

        # atom connections:

        self.broadcast_to = []  # push updates to
        self.connections = []   # recieve updates from

        # jacobian cell functions:

        self.jacfuncs = []
        self.derargfunc = None

        # parent object references:

        self.sys = None
        self.device = None

        # other:

        self.implicit = True

        self.tocsv = False
        self.outdir = None
        self.csv = None

    def add_connection(self, other, coefficient=1.0, coeffunc=None):

        connection = Connection(self, other, coefficient=coefficient,
                                coeffunc=coeffunc)

        connection.device = self.device

        self.connections.append(connection)

        return connection

    def add_jacfunc(self, other, func):

        self.jacfuncs.append((other, func))

    def set_state(self, value, quantize=False):

        self.x = float(value)

        if quantize:

            self.quantize(implicit=False)

        else:
            self.q = value
            self.qhi = self.q + self.dq
            self.qlo = self.q - self.dq

    def initialize(self, t0):

        self.tlast = t0
        self.tlastq = t0
        self.tnext = _INF

        self.saveclock = 0.0
        self.savetime = 0.0

        # init state:

        self.x = self.x0
        self.q = self.x
        self.qlast = self.x
        self.qd = 0.0
        self.qdlast = 0.0
        self.a = 0.0
        self.u = 0.0
        self.du = 0.0
        self.qsave = self.x
        self.xsave = self.x

        # init quantizer values:

        self.qhi = self.q + self.dq
        self.qlo = self.q - self.dq

        self.qdhi = self.qd + self.dq
        self.qdlo = self.qd - self.dq

        # init output:

        if self.sys.storage_type == StorageType.LIST:

            self.tout = []
            self.qout = []
            self.nupd = []
            self.tzoh = []
            self.qzoh = []

            self.tout_ss = []
            self.xout_ss = []
            self.nupd_ss = []

            self.tode = []
            self.xode = []
            self.uode = []

        elif self.sys.storage_type == StorageType.ARRAY:

            typecode = "d"

            self.tout = array(typecode)
            self.qout = array(typecode)
            self.nupd = array(typecode)
            self.tzoh = array(typecode)
            self.qzoh = array(typecode)

            self.tout_ss = array(typecode)
            self.xout_ss = array(typecode)
            self.nupd_ss = array(typecode)

            self.tode = array(typecode)
            self.xode = array(typecode)
            self.uode = array(typecode)

        elif self.sys.storage_type == StorageType.DEQUE:

            self.tout = deque()
            self.qout = deque()
            self.nupd = deque()
            self.tzoh = deque()
            self.qzoh = deque()

            self.tout_ss = deque()
            self.xout_ss = deque()
            self.nupd_ss = deque()

            self.tode = deque()
            self.xode = deque()
            self.uode = deque()

        self.updates = 0
        self.evals = 0
        self.updates_ss = 0
        self.updates_ode = 0

        self.tout.append(self.tlast)
        self.qout.append(self.qlast)
        self.nupd.append(0)
        self.tzoh.append(self.tlast)
        self.qzoh.append(self.qlast)

        self.tout_ss.append(self.tlast)
        self.xout_ss.append(self.qlast)
        self.nupd_ss.append(0)

        self.tode.append(self.tlast)
        self.xode.append(self.qlast)
        self.uode.append(0)

        self.tocsv = self.sys.tocsv
        self.outdir = self.sys.outdir

        if self.tocsv and self.outdir:

            self.csv = os.path.join(self.outdir, "{}_{}.csv".format(self.device.name, self.name))

            with open(self.csv, "w") as f:
                f.write("t,q,e,n\n")
                f.write("{},{},{}\n".format(self.tlast, self.qlast, 0, 0))

    def step(self, t):

        self.tlast = t
        self.evals += 1
        self.updates += 1
        self.dx = self.f()
        self.dint(t)
        self.q = self.x
        self.save_qss(t, force=True)
        self.qlast = self.q

    def dint(self, t):

        raise NotImplementedError()

    def quantize(self):

        raise NotImplementedError()

    def ta(self, t):

        raise NotImplementedError()

    def f(self, q, t=0.0):

        raise NotImplementedError()

    def df(self, q, d, t=0.0):

        raise NotImplementedError()

    def broadcast(self):

        for atom in self.broadcast_to:
            if atom is not self:
                atom.triggered = True

    def update_dq(self):

        if not self.dqerr:
            return
        else:
            if self.dqerr <= 0.0:
                return

        if not (self.dqmin or self.dqmax):
            return

        if (self.dqmax - self.dqmin) < _EPS:
            return

        self.dq = min(self.dqmax, max(self.dqmin, abs(self.dqerr * self.q)))

        self.qlo = self.q - self.dq
        self.qhi = self.q + self.dq

    def save_qss(self, t, force=False):

        self.updates += 1

        if self.q != self.qlast or force:

            self.tout.append(t)
            self.qout.append(self.q)
            self.nupd.append(self.updates)

            self.tzoh.append(t)
            self.qzoh.append(self.qzoh[-1])
            self.tzoh.append(t)
            self.qzoh.append(self.q)

    def save_ss(self, t, x):

        self.tout_ss.append(t)
        self.xout_ss.append(x)
        self.nupd_ss.append(self.updates_ss)
        self.updates_ss += 1

    def save_ode(self, t, x):

        self.tode.append(t)
        self.xode.append(x)
        self.uode.append(self.updates_ss)
        self.updates_ode += 1

    def get_error(self, typ="l2"):

        # interpolate qss to ss time vector:
        # this function can only be called after state space AND qdl simualtions
        # are complete

        qout_interp = numpy.interp(self.tout2, self.tout, self.qout)

        if typ.lower().strip() == "l2":

            # calculate the L**2 relative error:
            #      ________________
            #     / sum((y - q)**2)
            #    /  --------------
            #  \/      sum(y**2)

            dy_sqrd_sum = 0.0
            y_sqrd_sum = 0.0

            for q, y in zip(qout_interp, self.qout2):
                dy_sqrd_sum += (y - q)**2
                y_sqrd_sum += y**2

            return sqrt(dy_sqrd_sum / y_sqrd_sum)

        elif typ.lower().strip() == "nrmsd":   # <--- this is what we're using

            # calculate the normalized relative root mean squared error:
            #      ________________
            #     / sum((y - q)**2)
            #    /  ---------------
            #  \/          N
            # -----------------------
            #       max(y) - min(y)

            dy_sqrd_sum = 0.0
            y_sqrd_sum = 0.0

            for q, y in zip(qout_interp, self.qout2):
                dy_sqrd_sum += (y - q)**2
                y_sqrd_sum += y**2

            return sqrt(dy_sqrd_sum / len(qout_interp)) / (max(self.qout2)
                                                           - min(self.qout2))


        elif typ.lower().strip() == "re":

            # Pointwise relative error
            # e = [|(y - q)| / |y|]

            e = []

            for q, y in zip(qout_interp, self.qout2):
                e.append(abs(y-q) / abs(y))

            return e

        elif typ.lower().strip() == "rpd":

            # Pointwise relative percent difference
            # e = [ 100% * 2 * |y - q| / (|y| + |q|)]

            e = []

            for q, y in zip(qout_interp, self.qout2):
                den = abs(y) + abs(q)
                if den >= _EPS:
                    e.append(100 * 2 * abs(y-q) / (abs(y) + abs(q)))
                else:
                    e.append(0)

            return e

        return None

    def get_previous_state(self):

        if self.qout:
            if len(self.qout) >= 2:
                return self.qout[-2]
            else:
                return self.xlast
        else:
            return self.xlast

    def full_name(self):

        return self.device.name + "." + self.name

    def __repr__(self):

        return self.full_name()

    def __str__(self):

        return __repr__(self)


class SourceAtom(Atom):

    def __init__(self, name, source_type=SourceType.CONSTANT, x0=0.0, x1=0.0,
                 x2=0.0, xa=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0,
                 srcfunc=None, gainfunc=None, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e10, units=""):

        Atom.__init__(self, name=name, x0=x0, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.source_type = source_type
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.xa = xa
        self.freq = freq
        self.phi = phi
        self.duty = duty
        self.t1 = t1
        self.t2 = t2
        self.srcfunc = srcfunc
        self.gainfunc = gainfunc

        # source derived quantities:

        self.x = self.x0

        self.omega = 2.0 * pi * self.freq

        self.period = _INF
        if self.freq:
            self.period = 1.0 / self.freq

        if self.freq:
            self.T = 1.0 / self.freq

        if self.source_type == SourceType.RAMP:
            self.x0 = self.x1

        self.ramp_slope = 0.0
        if (self.t2 - self.t1) > 0:
            self.ramp_slope = (self.x2 - self.x1) / (self.t2 - self.t1)

    def dint(self, t):

        self.xprev = self.x

        if self.source_type == SourceType.FUNCTION:

            x = self.srcfunc(self.device, self.tlast)

        elif self.source_type == SourceType.CONSTANT:

            x = self.x0

        elif self.source_type == SourceType.STEP:

            if t < self.t1:
                x = self.x0
            else:
                x = self.x1

        elif self.source_type == SourceType.SINE:

            if t >= self.t1:
                x = self.x0 + self.xa * sin(self.omega * t + self.phi)
            else:
                x = self.x0

        elif self.source_type == SourceType.PWM:

            if self.duty <= 0.0:
                x = self.x2

            elif self.duty >= 1.0:
                x = self.x1

            else:

                w = t % self.period

                if ISCLOSE(w, self.period):
                    x = self.x1

                elif ISCLOSE(w, self.period * self.duty):
                    x = self.x2

                elif w < self.period * self.duty:
                    x = self.x1

                else:
                    x = self.x2

        elif self.source_type == SourceType.RAMP:

            if t <= self.t1:
                x = self.x1
            elif t <= self.t2:
                x = self.x1 + (t - self.t1) * self.dx
            else:
                x = self.x2

        elif self.source_type == SourceType.FUNCTION:

            x = self.srcfunc()

        if self.sys.enable_slewrate:
            if x > self.xprev:
                self.x = min(x, self.dmax * self.dq * (t - self.tlast) + self.xprev)
            elif x < self.u_prev:
                self.x = max(x, -self.dmax * self.dq * (t - self.tlast) + self.xprev)
        else:
            self.x = x

        if self.gainfunc:

            k = self.gainfunc(self.device)
            self.x *= k

        return self.x

    def quantize(self):

        self.q = self.x
        return False

    def ta(self, t):

        self.tnext = _INF

        if self.source_type == SourceType.FUNCTION:

            pass

        if self.source_type == SourceType.RAMP:

            if t < self.t1:
                self.tnext = self.t1

            elif t < self.t2:
                if self.dx > 0.0:
                    self.tnext = t + (self.q + self.dq - self.x) / self.dx
                elif self.dx < 0.0:
                    self.tnext = t + (self.q - self.dq - self.x) / self.dx
                else:
                    self.tnext = _INF

            else:
                self.tnext = _INF

        elif self.source_type == SourceType.STEP:

            if t < self.t1:
                self.tnext = self.t1
            else:
                self.tnext = _INF

        elif self.source_type == SourceType.SINE:

            if t < self.t1:

                self.tnext = self.t1

            else:

                w = t % self.T             # cycle time
                t0 = t - w                 # cycle start time
                theta = self.omega * w + self.phi  # wrapped angular position

                # value at current time w/o dc offset:
                x = self.xa * sin(2.0 * pi * self.freq * t)

                # determine next transition time. Saturate at +/- xa:

                # quadrant I
                if theta < pi/2.0:
                    self.tnext = (t0 + (asin(min(1.0, (x + self.dq) / self.xa)))
                                  / self.omega)

                # quadrant II
                elif theta < pi:
                    self.tnext = (t0 + self.T/2.0
                                  - (asin(max(0.0, (x - self.dq) / self.xa)))
                                  / self.omega)

                # quadrant III
                elif theta < 3.0*pi/2:
                    self.tnext = (t0 + self.T/2.0
                                  - (asin(max(-1.0, (x - self.dq) / self.xa)))
                                  / self.omega)

                # quadrant IV
                else:
                    self.tnext = (t0 + self.T
                                  + (asin(min(0.0, (x + self.dq) / self.xa)))
                                  / self.omega)

        elif self.source_type == SourceType.PWM:

            if self.duty <= 0.0 or self.duty >= 1.0:

                self.tnext = _INF

            else:

                w = t % self.period

                if ISCLOSE(w, self.period):
                    self.tnext = t + self.period * self.duty

                elif ISCLOSE(w, self.period * self.duty):
                    self.tnext = t + self.period - w

                elif w < self.period * self.duty:
                    self.tnext = t + self.period * self.duty - w

                else:
                    self.tnext = t + self.period - w


        elif self.source_type == SourceType.FUNCTION:

            pass
            #self.tnext = self.tlast + self.srcdt # <-- should we do this?

        #self.tnext = max(self.tnext, self.tlast + self.dtmin)

    def f(self, q, t):

        d = 0.0

        if self.source_type == SourceType.RAMP:

            d = self.ramp_slope

        elif self.source_type == SourceType.SINE:

            d = self.omega * self.xa * cos(self.omega * t + self.phi)

        elif self.source_type == SourceType.STEP:

            pass  # todo: sigmoid approx?

        elif self.source_type == SourceType.PWM:

            pass  # todo: sigmoid approx?

        elif self.source_type == SourceType.FUNCTION:

            d = 0.0  # todo: add a time derivative function delegate

        return d

    def df(self, u, du, t):

        d2 = 0.0

        if self.source_type == SourceType.RAMP:

            d2 = 0.0

        elif self.source_type == SourceType.SINE:

            d2 = -self.omega**2 * self.xa * sin(self.omega * t + self.phi)

        elif self.source_type == SourceType.STEP:

            pass  # todo: sigmoid approx?

        elif self.source_type == SourceType.PWM:

            pass  # todo: sigmoid approx?

        elif self.source_type == SourceType.FUNCTION:

            d2 = 0.0  # todo: add a 2nd time derivative function delegate

        return d2


class StateAtom(Atom):

    """ Qdl State Atom.
    """

    def __init__(self, name, x0=0.0, coefficient=0.0, coeffunc=None,
                 derfunc=None, der2func=None, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e10, units=""):

        Atom.__init__(self, name=name, x0=x0, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.derfunc = derfunc
        self.der2func = der2func

    def dint(self, t):

        self.x += self.dx * (t - self.tlast)

        return self.x

    def quantize(self, t=0.0, implicit=True):

        interp = False
        change = False

        self.qlast = self.q

        if self.qss_method == QssMethod.QSS1:

            if self.x >= self.q + self.dq:

                self.q += self.dq

            elif self.x <= self.q - self.dq:

                self.q -= self.dq

        elif self.qss_method == QssMethod.QSS2:

            if self.x >= self.q + self.dq:

                self.q += self.dq

            elif self.x <= self.q - self.dq:

                self.q -= self.dq

            if self.dx >= self.qd + self.dq:
            
                self.qd += self.dq
            
            elif self.dx <= self.qd - self.dq:
            
                self.qd -= self.dq

        elif self.qss_method == QssMethod.LIQSS1:

            # save previous derivative so we can see if the sign has changed:

            self.dxlast = self.dx

            # determine if the current internal state x is outside of the band:

            if self.x >= self.qhi:

                self.q = self.qhi
                self.qlo += self.dq
                change = True

            elif self.x <= self.qlo:

                self.q = self.qlo
                self.qlo -= self.dq
                change = True

            self.qhi = self.qlo + 2.0 * self.dq

            if change and self.implicit and implicit:

                # we've ventured out of (qlo, qhi) bounds:

                self.dx = self.f(self.q)

                # if the derivative has changed signs, then we know
                # we are in a potential oscillating situation, so
                # we will set the q such that the derivative ~= 0:

                if (self.dx * self.dxlast) < 0:

                    # derivative has changed sign:

                    flo = self.f(self.qlo)
                    fhi = self.f(self.qhi)
                    if flo != fhi:
                        a = (2.0 * self.dq) / (fhi - flo)
                        self.q = self.qhi - a * fhi
                        interp = True

            return interp

        elif self.qss_method == QssMethod.LIQSS2:

            """
            Line number reference paper:

            'Improving Linearly Implicit Quantized State System Methods'

            (Algorithm 5 listing)

            """

            ex = t - self.tlast

            # elapsed time since last qi update (10):

            eq = t - self.tlastq

            # store previous value of dxi/dt (8):

            self.dxlast = self.dx

            # affine coefficient projection (9):

            self.u += ex * self.du

            # store previous value of qi projected (11):

            self.qlast = self.q + eq * self.qd

            # h = MAX_2ND_ORDER_STEP_SIZE(xi) (12):

            h = self.max_2nd_order_stepsize()

            # 2ND_ORDER_STEP(x, h):

            qd = self.dx + h * self.ddx

            q = self.x + h * self.dx + h * h * self.ddx - h * self.q

            if self.q >= self.qhi:

                self.q = self.qhi
                self.qlo += self.dq

            elif self.q <= self.qlo:

                self.q = self.qlo
                self.qlo -= self.dq

            if self.qd >= self.qdhi:

                self.qd = self.qdhi
                self.qdlo += self.dq

            elif self.qd <= self.qdlo:

                self.qd = self.d
                self.qdlo -= self.dq

            self.qhi = self.qlo + 2.0 * self.dq
            self.qdhi = self.qdlo + 2.0 * self.dq

            return True

    def max_2nd_order_stepsize(self):

        h1, h2, h3, h4 = 0, 0, 0, 0

        den = self.a * self.qd + self.du - self.ddx

        if den == 0.0:

            return 0.0


        h1 = -(self.a * self.x + self.u - self.dx + self.a * self.dq) / den

        h2 = -(self.a * self.x + self.u - self.dx - self.a * self.dq) / den

        h3 = -(2 * self.a * self.x + 2 * self.u - 2 * self.dx + 2 * self.a * self.dq) / den

        h4 = -(2 * self.a * self.x + 2 * self.u - 2 * self.dx - 2 * self.a * self.dq) / den

        return min([h1, h2, h3, h4])

    def ta(self, t):

        if self.qss_method == QssMethod.QSS1:

            if self.dx > _EPS:
                self.tnext = t + (self.q + self.dq - self.x) / self.dx
            elif self.dx < -_EPS:
                self.tnext = t + (self.q - self.dq - self.x) / self.dx
            else:
                self.tnext = _INF

        elif self.qss_method == QssMethod.QSS2:

            ta1 = _INF
            ta2 = _INF

            if self.dx > _EPS:
                ta1 = t + (self.q + self.dq - self.x) / self.dx
            elif self.dx < -_EPS:
                ta1 = t + (self.q - self.dq - self.x) / self.dx

            if self.ddx > _EPS:
                ta2 = t + (self.qd + self.dq - self.dx) / self.ddx
            elif self.ddx < -_EPS:
                ta2 = t + (self.qd - self.dq - self.dx) / self.ddx

            self.tnext = min(ta1, ta2)

        elif self.qss_method == QssMethod.LIQSS1:

            if self.dx > _EPS:
                self.tnext = t + (self.qhi - self.x) / self.dx
            elif self.dx < -_EPS:
                self.tnext = t + (self.qlo - self.x) / self.dx
            else:
                self.tnext = _INF

        elif self.qss_method == QssMethod.LIQSS2:

            ta1 = _INF
            ta2 = _INF

            if self.dx > _EPS:
                ta1 = t + (self.qhi - self.x) / self.dx
            elif self.dx < -_EPS:
                ta1 = t + (self.qlo - self.x) / self.dx
            else:
                self.tnext = _INF

            if self.ddx > _EPS:
                ta2 = t + (self.qdhi - self.dx) / self.ddx
            elif self.ddx < -_EPS:
                ta2 = t + (self.qdlo - self.dx) / self.ddx

            self.tnext = min(ta1, ta2)

        self.tnext = max(self.tnext, t + self.dtmin)

    def compute_coefficient(self):

        if self.coeffunc:
            return self.coeffunc(self.device)
        else:
            return self.coefficient

    def f(self, q, t=0.0):

        if self.derfunc:
            if self.derargfunc:
                args = self.derargfunc(self.device)
                return self.derfunc(*args)
            else:
                return self.derfunc(self.device, q)

        d = self.compute_coefficient() * q

        for connection in self.connections:
            d += connection.value()

        return d

    def df(self, x, d, t=0.0):

        if self.derfunc:
            return self.der2func(self.device, x, d)

        d2 = self.compute_coefficient() * d

        for connection in self.connections:
            d2 += connection.value()

        return d2

    def set_state(self, x):

        self.x = x
        self.q = x
        self.qlo = x - self.dq
        self.qhi = x + self.dq


class System(object):

    def __init__(self, name="sys", qss_method=QssMethod.LIQSS1, dq=None, dqmin=None,
                 dqmax=None, dqerr=None, dtmin=None, dmax=None):

        global sys
        sys = self

        self.name = name
        self.qss_method = qss_method

        self.dtmin = DEF_DTMIN
        if dtmin:
            self.dtmin = dtmin

        self.dmax = DEF_DMAX
        if dmax:
            self.dmax = dmax

        # child elements:

        self.devices = []
        self.atoms = []
        self.state_atoms = []
        self.source_atoms = []
        self.n = 0
        self.m = 0

        # simulation variables:

        self.tstop = 0.0  # end simulation time
        self.time = 0.0   # current simulation time
        self.tsave = 0.0  # saved time for state restore
        self.iprint = 0   # for runtime updates
        self.dt = 1e-4
        self.enable_slewrate = False
        self.jacobian = None
        self.Km = 1.2

        self.savedt = 0.0   # max dt for saving data. Save all if zero

        self.tocsv = False  # send data to csv files
        self.outdir = None  # send data to csv files

        # events:

        self.events = {}

        # memory management:
        self.storage_type = StorageType.LIST

        self.show_time = False
        self.time_queue = queue.Queue()

    def schedule(self, func, t):

        if not t in self.events:
            self.events[t] = []

        self.events[t].append(func)

    def add_device(self, device):

        self.devices.append(device)

        for atom in device.atoms:

            if not atom.dq:
                atom.dq = self.dq

            #if not atom.dqmin:
            #    atom.dqmin = self.dqmin
            #
            #if not atom.dqmax:
            #    atom.dqmax = self.dqmax
            #
            #if not atom.dqerr:
            #    atom.dqerr = self.dqerr

            if not atom.dtmin:
                atom.dtmin = self.dtmin

            if not atom.dmax:
                atom.dmax = self.dmax

            if not atom.qss_method:
                atom.qss_method = self.qss_method

            atom.device = device
            atom.sys = self

            self.atoms.append(atom)

            if isinstance(atom, StateAtom):
                atom.index = self.n
                self.state_atoms.append(atom)
                self.n += 1

            elif isinstance(atom, SourceAtom):
                atom.index = self.m
                self.source_atoms.append(atom)
                self.m += 1

        setattr(self, device.name, device)

    def add_devices(self, *devices):

        for device in devices:
            device.setup_connections()

        for device in devices:
            device.setup_functions()

        for device in devices:
            self.add_device(device)

    def save_state(self):

        self.tsave = self.time

        for atom in self.atoms:
            atom.qsave = atom.q
            atom.xsave = atom.x

    def connect(self, from_port, to_port):

        from_port.connect(to_port)

    def restore_state(self):

        self.time = self.tsave

        for atom in self.atoms:
            atom.q = atom.qsave
            atom.x = atom.xsave

            atom.qhi = atom.q + atom.dq
            atom.qlo = atom.q - atom.dq

    def get_jacobian(self):

        jacobian = np.zeros((self.n, self.n))

        for atom in self.state_atoms:
            for other, func in atom.jacfuncs:
                if atom.derargfunc:
                    args = atom.derargfunc(atom.device)
                    jacobian[atom.index, other.index] = func(*args)
                else:
                    if atom is other:
                        jacobian[atom.index, other.index] = func(atom.device, atom.q)
                    else:
                        jacobian[atom.index, other.index] = func(atom.device, atom.q, other.index)

        return jacobian

    @staticmethod
    def fode(t, x, sys):

        """Returns array of derivatives from state atoms. This function must be
        a static method in order to be passed as a delgate to the
        scipy ode integrator function. Note that sys is a global module variable.
        """

        dx_dt = [0.0] * sys.n

        for atom in sys.state_atoms:
            atom.q = x[atom.index]

        for atom in sys.state_atoms:
            dx_dt[atom.index] = atom.f(atom.q, t)

        return dx_dt

    @staticmethod
    def fode2(x, t=0.0, sys=None):

        """Returns array of derivatives from state atoms. This function must be
        a static method in order to be passed as a delgate to the
        scipy ode integrator function. Note that sys is a global module variable.
        (not that this function differs from self.fode in the argument order).
        """

        dx_dt = [0.0] * sys.n

        for atom in sys.state_atoms:
            atom.q = x[atom.index]

        for atom in sys.state_atoms:
            dx_dt[atom.index] = atom.f(atom.q, t)

        return dx_dt

    def solve_dc(self, init=True, set=True):

        xi = [0.0]*self.n

        for atom in self.state_atoms:
            if init:
                xi[atom.index] = atom.x0
            else:
                xi[atom.index] = atom.x

        xdc = fsolve(self.fode2, xi, args=(0, sys), xtol=1e-12)

        for atom in self.state_atoms:
            if init:
                atom.x0 = xdc[atom.index]
            elif set:
                atom.x = xdc[atom.index]
                atom.q = atom.x

        return xdc

    def initialize(self, t0=0.0, dt=1e-4, dc=False, savedt=0.0, tocsv=False, outdir=None):

        self.time = t0
        self.dt = dt

        self.savedt = savedt

        self.tocsv = tocsv
        self.outdir = outdir

        self.dq0 = np.zeros((self.n, 1))

        for atom in self.state_atoms:
            self.dq0[atom.index] = atom.dq0

        if dc:
            self.solve_dc()

        for atom in self.state_atoms:
            atom.initialize(self.time)

        for atom in self.source_atoms:
            atom.initialize(self.time)

    def print_time(self):

        while self.show_time:
            time.sleep(1.0)
            print(simtime)

    #def run_threaded(self, tstop, ode=True, qss=True, verbose=2, qss_fixed_dt=None,
    #        ode_method="RK45", optimize_dq=False, chk_ss_delay=None):
    #
    #    args = (tstop, ode, qss, verbose, qss_fixed_dt, ode_method, optimize_dq, chk_ss_delay)
    #
    #    self.show_time = True
    #
    #    self.print_thread = th.Thread(target=self.print_time)
    #    self.run_thread = th.Thread(target=self.run_threaded, args=args)
    #
    #    self.print_thread.start()
    #    self.run_thread.start()
        
    def run(self, tstop, ode=True, qss=True, verbose=2, qss_fixed_dt=None,
            ode_method="RK45", optimize_dq=False, chk_ss_delay=None):

        self.tstop = tstop

        self.verbose = verbose
        self.calc_ss = False

        #self.show_time = True
        #self.print_thread = th.Thread(target=self.print_time)
        #self.print_thread.start()

        if optimize_dq or chk_ss_delay:
            self.calc_ss = True
            self.update_steadystate_distance()

        self.ode_method = ode_method

        # get the event times and event function lists, sorted by time:

        sorted_events = sorted(self.events.items())

        # add the last event (tstop, or end of simulation) to the lists:

        sorted_events.append((tstop, None))

        # loop through the event times and solve:

        print("Simulation started......")

        start_time = time.time()

        for event_time, events in sorted_events:

            if self.calc_ss:
                self.calc_steadystate()

            if optimize_dq:
                self.optimize_dq()
                self.update_steadystate_distance()

            if ode:

                print("ODE solution started to next event...")

                if qss: self.save_state()

                xi = [0.0]*self.n
                for atom in self.state_atoms:
                    xi[atom.index] = atom.x

                tspan = (self.time, event_time)

                soln = solve_ivp(self.fode, tspan, xi, ode_method, args=(sys,),
                                 max_step=self.dt)
                t = soln.t
                x = soln.y

                for i in range(len(t)):

                    for atom in self.state_atoms:
                        atom.q = x[atom.index, i]
                        atom.save_ode(t[i], atom.q)

                    for atom in self.source_atoms:
                        atom.save_ode(t[i], atom.q)
                        atom.dint(t[i])

                for atom in self.state_atoms:
                    xf = x[atom.index, -1]
                    atom.x = xf
                    atom.q = xf

                print("ODE solution completed to next event.")

            if qss:

                print("QSS solution started to next event...")

                if ode: self.restore_state()

                if self.qss_method == QssMethod.QSS1:

                    self.run_qss1()

                elif self.qss_method == QssMethod.QSS2:

                    self.run_qss2()

                elif self.qss_method == QssMethod.LIQSS1:

                    self.run_liqss1(event_time)

                elif self.qss_method == QssMethod.LIQSS2:

                    self.run_liqss2()

                print("QSS solution completed to next event.")

            if events:

                for event in events:
                    event(self)

            self.time = event_time

        #self.show_time = False
        #self.print_thread.join()

        print(f"Simulation complete. Total run time = {time.time() - start_time} s.")

    def run_qss1(self):

        t = self.time

        for atom in self.atoms:
            atom.dx = atom.f(atom.q, t)
            atom.ta(t)
            atom.tlast = t
            atom.save_qss(t)

        while(t < self.tstop):                      # 1

            if self.verbose == 2: print(t)

            atomi = None
            t = self.tstop

            for atom in self.atoms:
                if atom.tnext <= t:
                    t = atom.tnext                  # 2
                    atomi = atom                    # 3

            if not atomi: break

            e = t - atomi.tlast                     # 4
            atomi.x = atomi.x + atomi.dx * e        # 5
            atomi.quantize()                        # 6
            atomi.ta(t)                             # 7

            for atomj in atomi.broadcast_to:        # 8

                e = t - atomj.tlast                 # 9
                atomj.x = atomj.x + atomj.dx * e     # 10

                atomj.dx = atomj.f(atomj.q, t)       # 12
                atomj.ta(t)                         # 13

                if atomj is not atomi:
                    atomj.tlast = t                 # 11

            atomi.tlast = t                         # 13
            atomi.save_qss(t)

        for atom in self.atoms:

            atom.tlast = t
            atom.save_qss(t, force=True)

        self.time = self.tstop

    def run_qss2(self):

        t = self.time

        if t == self.tstart:
            for atom in self.atoms:
                atom.dx = atom.f(atom.q, t)
                atom.ddx = atom.df(atom.q, atom.dx, t)
                atom.ta(t)
                atom.tlast = t
                atom.save_qss(t)

        while(t < self.tstop):                                # 1

            if self.verbose == 2: print(t)

            atomi = None
            t = self.tstop

            for atom in self.atoms:
                if atom.tnext <= t:
                    t = atom.tnext                            # 2
                    atomi = atom                              # 3

            if not atomi: break

            e = t - atomi.tlast                               # 4
            ee = e**2

            atomi.x += atomi.dx * e + 0.5 * atomi.ddx * ee     # 6
            atomi.dx += atomi.ddx * e                          # 7

            if atomi.q >= atomi.q + atomi.dq:
                atomi.q += atomi.dq

            elif atomi.q <= atomi.q - atomi.dq:
                atomi.q -= atomi.dq

            atomi.qd = atomi.dx

            # 11:

            ta1 = _INF
            ta2 = _INF

            if atomi.dx > _EPS:
                ta1 = t + (atomi.q + atomi.dq - atomi.x) / atomi.dx
            elif atomi.dx < -_EPS:
                ta1 = t + (atomi.q - atomi.dq - atomi.x) / atomi.dx

            if atomi.ddx > _EPS:
                ta2 = t + (atomi.qd + atomi.dq - atomi.dx) / atomi.ddx
            elif atomi.ddx < -_EPS:
                ta2 = t + (atomi.qd - atomi.dq - atomi.dx) / atomi.ddx

            self.tnext = max(t, min(ta1, ta2))

            for atomj in atomi.broadcast_to:                   # 12

                e = t - atomj.tlast                            # 13
                ee = e**2
                atomj.x += atomj.dx * e + 0.5 * atomj.ddx * ee # 15

                atomj.dx = atomj.f(atomj.q, t)                 # 16
                atomj.ddx = atomj.df(atomj.q, atomj.dx, t)     # 17
                
                # 18:

                ta1 = _INF
                ta2 = _INF

                if atomj.dx > _EPS:
                    ta1 = t + (atomj.q + atomj.dq - atomj.x) / atomj.dx
                elif atomj.dx < -_EPS:
                    ta1 = t + (atomj.q - atomj.dq - atomj.x) / atomj.dx

                if atomi.ddx > _EPS:
                    ta2 = t + (atomj.qd + atomj.dq - atomj.dx) / atomj.ddx
                elif atomi.ddx < -_EPS:
                    ta2 = t + (atomj.qd - atomj.dq - atomj.dx) / atomj.ddx

                atomj.tnext = max(t, min(ta1, ta2))

                if atomj is not atomi:
                    atomj.tlast = t                            # 19

            atomi.tlast = t                                    # 21
            atomi.save_qss(t)

        for atom in self.atoms:

            atom.tlast = t
            atom.save_qss(t, force=True)

        self.time = self.tstop

    def run_liqss1(self, tnext):

        #global simtime

        t = self.time

        i = 0
        n = 1e6

        for atom in self.atoms:
            atom.dx = atom.f(atom.q, t)
            atom.ta(t)
            atom.tlast = t
            atom.save_qss(t)

        while(t < tnext):                           # 1

            if self.verbose == 2: print(t)

            atomi = None
            t = tnext

            for atom in self.atoms:
                if atom.tnext <= t:
                    t = atom.tnext                  # 2
                    atomi = atom                    # 3

            if not atomi: break

            e = t - atomi.tlast                     # 4
            atomi.x = atomi.x + atomi.dx * e        # 5
            atomi.quantize()                        # 6
            atomi.ta(t)                             # 7

            for atomj in atomi.broadcast_to:        # 8

                e = t - atomj.tlast                 # 9
                atomj.x = atomj.x + atomj.dx * e    # 10

                if atomj is not atomi:
                    atomj.tlast = t                 # 11

                atomj.dx = atomj.f(atomj.q, t)      # 12
                atomj.ta(t)                         # 13

            atomi.tlast = t                         # 13
            atomi.save_qss(t)
            
            #simtime = t

            i = i + 1
            if i > n:
                i = 0
                print(t)

        self.time = tnext

        for atom in self.atoms:

            atom.tlast = t
            atom.save_qss(t, force=True)

    def run_liqss1_test(self):

        t = self.time

        for atom in self.atoms:

            atom.dxlast = atom.dx
            atom.dx = atom.f(atom.q, t)
            atom.a = (atom.dx - atom.dxlast) / (2 * atom.dq)
            atom.u = atom.dx - atom.a * atom.q
            atom.tlast = t
            atom.ta(t)
            atom.save_qss(t)

        while(t < self.tstop):                      # 1

            if self.verbose == 2: print(t)

            atomi = None
            t = self.tstop

            for atom in self.atoms:
                if atom.tnext <= t:
                    t = atom.tnext                  # 2
                    atomi = atom                    # 3

            if not atomi: break

            e = t - atomi.tlast                     # 4

            atomi.x = atomi.x + atomi.dx * e        # 5

            atomi.qlast = atomi.q                   # 6

            atomi.dxlast = atomi.dx                 # 7

            dx_sign = 0
            if atomi.dx > 0: dx_sign = 1
            elif atomi.dx < 0: dx_sign = -1

            dx_plus = atomi.a * (atomi.x + dx_sign * atomi.dq) + atomi.u   # 8

            

            if atomi.x >= atomi.qhi:

                atomi.q = atomi.qhi
                atomi.qlo += atomi.dq

            elif atomi.x <= atomi.qlo:

                atomi.q = atomi.qlo
                atomi.qlo -= atomi.dq

            if atomi.dx * dx_sign < 0:

                atomi.q = -atomi.u / atomi.a

            atomi.qhi = atomi.qlo + 2.0 * atomi.dq

            # 7:

            if atomi.dx > _EPS:
                ta = t + (atomi.qhi - atomi.x) / atomi.dx
            elif atomi.dx < -_EPS:
                ta = t + (atomi.qlo - atomi.x) / atomi.dx
            else:
                ta = _INF

            atomi.tnext = max(t, ta)

            for atomj in atomi.broadcast_to:        # 8

                e = t - atomj.tlast                 # 9
                atomj.x = atomj.x + atomj.dx * e    # 10

                if atomj is not atomi:
                    atomj.tlast = t                 # 11

                atomj.dx = atomj.f(atomj.q, t)      # 12

                # 13:

                if atomj.dx > _EPS:
                    ta = t + (atomj.qhi - atomj.x) / atomj.dx
                elif atomj.dx < -_EPS:
                    ta = t + (atomj.qlo - atomj.x) / atomj.dx
                else:
                    ta = _INF

                atomj.tnext = max(t, ta)

            atomi.a = (atomi.dx - atomi.dxlast) / (2 * atomi.dq)  # 23

            atomi.u = atomi.dx - atomi.a * atomi.q  # 24

            atomi.tlast = t                         # 25

            atomi.save_qss(t)

        for atom in self.atoms:

            atom.tlast = t
            atom.save_qss(t, force=True)

        self.time = self.tstop

    def run_liqss2(self):

        t = self.time

        for atom in self.atoms:

            atom.dx = atom.f(atom.q, t)
            atom.ddx = atom.df(atom.dx, atom.q, t)

            atom.a = 0.0
            atom.u = atom.dx
            atom.du = atom.ddx

            atom.ta(t)
            atom.tlast = t
            atom.tlastq = t

            atom.save_qss(t)

        while(t < self.tstop):                      # 1

            if self.verbose == 2: print(t)

            atomi = None
            t = self.tstop

            for atom in self.atoms:
                if atom.tnext <= t:
                    t = atom.tnext                  # 2
                    atomi = atom                    # 3

            if not atomi: break

            e = t - atomi.tlast                     # 4
            ee = e**2

            atomi.x += atomi.dx * e + 0.5 * atomi.ddx * ee  # 6
            atomi.dx += atomi.ddx * e                       # 7

            atomi.quantize(t)                       # 8-13

            atomi.tlastq = t                        # 14

            atomi.ta(t)                             # 15

            for atomj in atomi.broadcast_to:        # 16

                e = t - atomj.tlast                 # 17-18

                atomj.x += atomj.dx * e + 0.5 * atomj.ddx * ee  # 18

                atomj.dx = atomj.f(atomj.q, t)               # 16

                atomj.ddx = atomj.df(atomj.q, atomj.dx, t)   # 17

                atomj.ta(t)                                  # 18

                if atomj is not atomi:
                    atomj.tlast = t                 # 23-25

            if atomi.q != atomi.qlast:

                atomi.a = (atomi.dx - atomi.dxlast) / (atomi.q - atomi.qlast)  # 26

                atomi.u = atomi.dx - atomi.a * atomi.q        # 27

                atomi.du = atomi.ddx - atomi.a * atomi.qd     # 28

            atomi.tlast = t

            atomi.save_qss(t)

        for atom in self.atoms:

            atom.tlast = t
            atom.save_qss(t, force=True)

        self.time = self.tstop

    def get_next(self):

        # Get next time and flag atoms to solve:

        tnext = _INF
        anext = None

        for atom in self.atoms:
            tnext = min(atom.tnext, tnext)
            anext = atom

        # limit by minimum time step:

        #tnext = max(tnext, self.time + _EPS)

        # limit by end of simulation section (to next scheduled event):

        tnext = min(tnext, self.tstop)

        return anext, tnext

    def advance(self):

        tnext = _INF

        for atom in self.atoms:
            tnext = min(atom.tnext, tnext)

        self.time = max(tnext, self.time + _EPS)
        self.time = min(self.time, self.tstop)

        if 0:  # method 1: all < tnext atoms w/ nested triggered (one iter)

            for atom in self.atoms:

                if atom.tnext <= self.time:

                    atom.update(self.time)

                    for atom in self.atoms:

                        if atom.triggered:

                            atom.update(self.time)

        if 0:  # method 2: all < tnext atoms then triggered (one iter)

            for atom in self.atoms:

                if atom.tnext <= self.time:

                    atom.update(self.time)

            for atom in self.atoms:

                if atom.triggered:

                    atom.update(self.time)


        if 0:  # method 3: all < tnext atoms w/ nested triggered (multi-iter)

            for atom in self.atoms:

                if atom.tnext <= self.time:

                    atom.update(self.time)

                    i = 0
                    while i < _MAXITER:
                        triggered = False
                        for atom in self.atoms:
                            if atom.triggered:
                                triggered = True
                                atom.update(self.time)
                        if not triggered:
                            break
                        i += 1

        if 0:  # method 4: first tnext atoms, then iterate over triggered:

            for atom in self.atoms:
                if atom.tnext <= self.time:
                    atom.update(self.time)

            i = 0
            while i < _MAXITER:
                triggered = False
                for atom in self.atoms:
                    if atom.triggered:
                        triggered = True
                        atom.update(self.time)
                if not triggered:
                    break
                i += 1

        if 1:  # method 5: full outer loop iterations with nexted triggered:

            i = 0
            while i < _MAXITER:
                triggered1 = False
                for atom in self.atoms:
                    if atom.tnext <= self.time:
                        atom.update(self.time)
                        triggered1 = True
                    j = 0
                    while j < _MAXITER:
                        triggered2 = False
                        for atom in self.atoms:
                            if atom.triggered:
                                triggered2 = True
                                atom.update(self.time)
                        if not triggered2:
                            break
                        j += 1
                i += 1

    def calc_steadystate(self):

        self.jac1 = self.get_jacobian()

        self.save_state()

        self.xf = self.solve_dc(init=False, set=False)

        for atom in self.state_atoms:
            atom.xf = self.xf[atom.index]

        self.jac2 = self.get_jacobian()

        self.restore_state()

    def update_steadystate_distance(self):

       dq0 = [0.0]*self.n
       for atom in self.state_atoms:
           dq0[atom.index] = atom.dq0

       self.steadystate_distance = la.norm(dq0) * self.Km

    def optimize_dq(self):

        if self.verbose:
            print("dq0 = {}\n".format(self.dq0))
            print("jac1 = {}\n".format(self.jac1))

        if 1:

            QQ0 = np.square(self.dq0)

            JTJ = self.jac1.transpose().dot(self.jac1)
            QQ = la.solve(JTJ, QQ0)
            dq1 = np.sqrt(np.abs(QQ))

            JTJ = self.jac2.transpose().dot(self.jac1)
            QQ = la.solve(JTJ, QQ0)
            dq2 = np.sqrt(np.abs(QQ))

        if 0:

            factor = 0.5

            E = np.zeros((self.n, self.n))

            dq1 = np.zeros((self.n, 1))
            dq2 = np.zeros((self.n, 1))

            for atom in self.state_atoms:
                for j in range(self.n):
                    if atom.index == j:
                        E[atom.index, atom.index] = (atom.dq0*factor)**2
                    else:
                        pass
                        E[atom.index, j] = (atom.dq0*factor)

            JTJ = self.jac1.transpose().dot(self.jac1)
            Q = la.solve(JTJ, E)

            for atom in self.state_atoms:
                dq = 999999.9
                for j in range(self.n):
                    if atom.index == j:
                       dqii = sqrt(abs(Q[atom.index, j]))
                       dqii = abs(Q[atom.index, j])
                       if dqii < dq:
                           dq = dqii
                    else:
                       dqij = abs(Q[atom.index, j])
                       if dqij < dq:
                           dq = dqij
                dq1[atom.index, 0] = dq

            JTJ = self.jac2.transpose().dot(self.jac2)
            Q = la.solve(JTJ, E)

            for atom in self.state_atoms:
                dq = 999999.9
                for j in range(self.n):
                    if atom.index == j:
                        dqii = sqrt(abs(Q[atom.index, j]))
                        dqii = abs(Q[atom.index, j])
                        if dqii < dq:
                            dq = dqii
                    else:
                        dqij = abs(Q[atom.index, j])
                        if dqij < dq:
                            dq = dqij
                dq2[atom.index, 0] = dq

        if self.verbose:
            print("at t=inf:")
            print("dq1 = {}\n".format(dq1))
            print("at t=0+:")
            print("dq2 = {}\n".format(dq2))

        for atom in self.state_atoms:

            #atom.dq = min(atom.dq0, dq1[atom.index, 0], dq2[atom.index, 0])
            atom.dq = min(dq1[atom.index, 0], dq2[atom.index, 0]) * 0.5

            atom.qhi = atom.q + atom.dq
            atom.qlo = atom.q - atom.dq

            if self.verbose:
                print("dq_{} = {} ({})\n".format(atom.full_name(), atom.dq, atom.units))

    def check_steadystate(self, t, apply_if_true=True):

        is_ss = False

        q = [0.0]*self.n
        for atom in self.state_atoms:
            q[atom.index] = atom.q

        qe = la.norm(np.add(q, -self.xf))

        if (qe < self.steadystate_distance):
            is_ss = True

        if is_ss and apply_if_true:

            for atom in self.state_atoms:
                atom.set_state(self.xf[atom.index])

            for atom in self.source_atoms:
                atom.dint(t)
                atom.q = atom.x

        return is_ss

    def plot_devices(self, *devices, plot_qss=True, plot_ss=False,
             plot_upd=False, plot_ss_updates=False, legend=False):

        for device in devices:
            for atom in devices.atoms:
                atoms.append(atom)

        self.plot(self, *atoms, plot_qss=plot_qss, plot_ss=plot_ss,
                  plot_upd=plot_upd,
                  plot_ss_updates=plot_ss_updates, legend=legend)

    def plot_old(self, *atoms, plot_qss=True, plot_ss=False,
             plot_upd=False, plot_ss_updates=False, legend=False):

        if not atoms:
            atoms = self.state_atoms

        c, j = 2, 1
        r = floor(len(atoms)/2) + 1

        for atom in atoms:

            ax1 = None
            ax2 = None

            plt.subplot(r, c, j)

            if plot_qss or plot_ss:

                ax1 = plt.gca()
                ax1.set_ylabel("{} ({})".format(atom.full_name(), atom.units),
                               color='b')
                ax1.grid()

            if plot_upd or plot_ss_updates:
                ax2 = ax1.twinx()
                ax2.set_ylabel('updates', color='r')

            if plot_qss:
                ax1.plot(atom.tzoh, atom.qzoh, 'b-', label="qss_q")

            if plot_ss:
                ax1.plot(atom.tout_ss, atom.xout_ss, 'c--', label="ss_x")

            if plot_upd:
                ax2.hist(atom.tout, 100)
                #ax2.plot(atom.tout, atom.nupd, 'r-', label="qss updates")

            if plot_ss_updates:
                ax2.plot(self.tout_ss, self.nupd_ss, 'm--', label="ss_upds")

            if ax1 and legend:
                ax1.legend(loc="upper left")

            if ax2 and legend:
                ax2.legend(loc="upper right")

            plt.xlabel("t (s)")

            j += 1

        plt.tight_layout()
        plt.show()

    def plot_groups(self, *groups, plot_qss=False, plot_ss=False, plot_ode=False):

        c, j = 1, 1
        r = CEIL(len(atoms) / c)

        for atoms in groups:

            plt.subplot(r, c, j)

            if plot_qss:

                for i, atom in enumerate(atoms):

                    color = "C{}".format(i)

                    lbl = "{} qss ({})".format(atom.full_name(), atom.units)

                    plt.plot(atom.tout, atom.qout,
                             marker='.',
                             markersize=4,
                             markerfacecolor='none',
                             markeredgecolor=color,
                             markeredgewidth=0.5,
                             linestyle='none',
                             label=lbl)

            if plot_ode:

                for i, atom in enumerate(atoms):

                    color = "C{}".format(i)

                    lbl = "{} ode ({})".format(atom.full_name(), atom.units)

                    plt.plot(atom.tode, atom.xode,
                             color=color,
                             alpha=0.6,
                             linewidth=1.0,
                             linestyle='dashed',
                             label=lbl)

            plt.legend(loc="lower right")
            plt.ylabel("atom state")
            plt.xlabel("t (s)")
            plt.grid()

            j += 1

        plt.tight_layout()
        plt.show()

    def plot(self, *atoms, plot_qss=False, plot_zoh=False, plot_ss=False, plot_ode=False,
             plot_upd=False, plot_ss_updates=False, legloc=None,
             plot_ode_updates=False, legend=True, errorband=False, upd_bins=1000,
             pth=None):

        c, j = 1, 1
        r = CEIL(len(atoms) / c)

        if r % c > 0.0: r += 1

        fig = plt.figure()

        for i, atom in enumerate(atoms):

            plt.subplot(r, c, j)

            ax1 = plt.gca()

            if atom.units:
                lbl = f"{atom.full_name()} ({atom.units})"
            else:
                lbl = atom.full_name()

            ax1.set_ylabel(lbl, color='tab:red')

            ax1.grid()

            ax2 = None

            if plot_upd or plot_ss_updates:

                ax2 = ax1.twinx()
                ylabel = "update frequency (Hz)"
                ax2.set_ylabel(ylabel, color='tab:blue')

            if plot_upd:

                dt = atom.tout[-1] / upd_bins

                label = "update frequency"
                #ax2.hist(atom.tout, upd_bins, alpha=0.5,
                #         color='b', label=label, density=True)

                n = len(atom.tout)
                bw = n**(-2/3)
                kde = gaussian_kde(atom.tout, bw_method=bw)
                t = np.arange(0.0, atom.tout[-1], dt/10)
                pdensity = kde(t) * n

                ax2.fill_between(t, pdensity, 0, lw=0,
                                 color='tab:blue', alpha=0.2,
                                 label=label)

            if plot_ss_updates:

                ax2.plot(self.tout_ss, self.nupd_ss, 'tab:blue', label="ss_upds")

            if plot_qss:

                #lbl = "{} qss ({})".format(atom.full_name(), atom.units)
                lbl = f"QSS ({self.qss_method})"

                ax1.plot(atom.tout, atom.qout,
                         marker='.',
                         markersize=4,
                         markerfacecolor='none',
                         markeredgecolor='tab:red',
                         markeredgewidth=0.5,
                         alpha=1.0,
                         linestyle='none',
                         label=lbl)

            if plot_zoh:

                lbl = f"QSS ({self.qss_method}) (ZOH)"

                ax1.plot(atom.tzoh, atom.qzoh, color="tab:red", linestyle="-",
                         alpha=0.5, label=lbl)

            if plot_ss:

                lbl = "State Space"

                ax1.plot(atom.tout_ss, atom.xout_ss,
                         color='r',
                         linewidth=1.0,
                         linestyle='dashed',
                         label=lbl)

            if plot_ode:

                lbl = f"ODE ({self.ode_method})"

                if errorband:

                    xhi = [x + atom.dq0 for x in atom.xode]
                    xlo = [x - atom.dq0 for x in atom.xode]


                    ax1.plot(atom.tode, atom.xode,
                             color='k',
                             alpha=0.6,
                             linewidth=1.0,
                             linestyle='dashed',
                             label=lbl)

                    lbl = "Error Band"

                    ax1.fill_between(atom.tode, xhi, xlo, color='k', alpha=0.1,
                                     label=lbl)

                else:

                    ax1.plot(atom.tode, atom.xode,
                             color='k',
                             alpha=0.6,
                             linewidth=1.0,
                             linestyle='dashed',
                             label=lbl)

            loc = "best"

            if legloc:
                loc = legloc

            lines1, labels1 = ax1.get_legend_handles_labels()

            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc=loc)
            else:
                ax1.legend(lines1, labels1, loc=loc)

            plt.xlabel("t (s)")
            j += 1

        plt.tight_layout()

        if pth:
            fig.savefig(pth)
        else:
            plt.show()

    def plot_csv(self, csvfiles, plot_upd=False, legloc=None, legend=True, upd_bins=1000, pth=None):

        # t,q,e,n

        c, j = 1, 1
        r = CEIL(len(csvfiles) / c)

        fig = plt.figure()

        for i, filepath in enumerate(csvfiles):

            plt.subplot(r, c, j)

            head, tail = os.path.split(filepath)
            filename, ext = os.path.splitext(tail)

            tout = []
            qout = []
            evals = []
            updates = []

            lines = []

            with open(filepath, "r") as f:
                lines = f.readlines()

            if not lines:
                return

            for line in lines[1:]:

                items = line.split(",")

                tout.append(items[0])
                qout.append(items[1])
                evals.append(items[2])

                if len(items) > 2:
                    updates.append(items[3])
                else:
                    updates.append(0)

            ax1 = plt.gca()
            ax1.set_ylabel("{}".format(filename), color='tab:red')
            ax1.grid()

            ax2 = None

            if plot_upd or plot_ss_updates:

                ax2 = ax1.twinx()
                ylabel = "update density ($s^{-1}$)"
                ax2.set_ylabel(ylabel, color='tab:blue')

            if plot_upd:

                dt = tout[-1] / upd_bins

                label = "update density"
                #ax2.hist(atom.tout, upd_bins, alpha=0.5,
                #         color='b', label=label, density=True)

                n = len(tout)
                bw = n**(-2/3)
                kde = gaussian_kde(tout, bw_method=bw)
                t = np.arange(0.0, tout[-1], dt/10)
                pdensity = kde(t) * n

                ax2.fill_between(t, pdensity, 0, lw=0,
                                 color='tab:blue', alpha=0.2,
                                 label=label)

            #if plot_ss_updates:
            #
            #    ax2.plot(self.tout_ss, self.nupd_ss, 'tab:blue', label="ss_upds")

            if plot_qss:

                #lbl = "{} qss ({})".format(atom.full_name(), atom.units)
                lbl = "qss"

                ax1.plot(tout, qout,
                         marker='.',
                         markersize=4,
                         markerfacecolor='none',
                         markeredgecolor='tab:red',
                         markeredgewidth=0.5,
                         alpha=1.0,
                         linestyle='none',
                         label=lbl)

            #if plot_zoh:
            #
            #    lbl = "qss (zoh)"
            #
            #    ax1.plot(atom.tzoh, atom.qzoh, color="tab:red", linestyle="-",
            #             alpha=0.5, label=lbl)

            #if plot_ss:
            #
            #    lbl = "ss"
            #
            #    ax1.plot(atom.tout_ss, atom.xout_ss,
            #             color='r',
            #             linewidth=1.0,
            #             linestyle='dashed',
            #             label=lbl)

            #if plot_ode:
            #
            #    lbl = "ode"
            #
            #    if errorband:
            #
            #        xhi = [x + atom.dq0 for x in atom.xode]
            #        xlo = [x - atom.dq0 for x in atom.xode]
            #
            #
            #        ax1.plot(atom.tode, atom.xode,
            #                 color='k',
            #                 alpha=0.6,
            #                 linewidth=1.0,
            #                 linestyle='dashed',
            #                 label=lbl)
            #
            #        lbl = "error band"
            #
            #        ax1.fill_between(atom.tode, xhi, xlo, color='k', alpha=0.1,
            #                         label=lbl)
            #
            #    else:
            #
            #        ax1.plot(atom.tode, atom.xode,
            #                 color='k',
            #                 alpha=0.6,
            #                 linewidth=1.0,
            #                 linestyle='dashed',
            #                 label=lbl)

            loc = "best"

            if legloc:
                loc = legloc

            lines1, labels1 = ax1.get_legend_handles_labels()

            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2, loc=loc)
            else:
                ax1.legend(lines1, labels1, loc=loc)

            plt.xlabel("t (s)")
            j += 1

        plt.tight_layout()

        if pth:
            fig.savefig(pth)
        else:
            plt.show()

    def plotxy(self, atomx, atomy, arrows=False, ss_region=False, auto_limits=False):

        ftheta = interp1d(atomx.tout, atomx.qout, kind='zero')
        fomega = interp1d(atomy.tout, atomy.qout, kind='zero')

        tboth = np.concatenate((atomx.tout, atomy.tout))
        tsort = np.sort(tboth)
        t = np.unique(tsort)

        x = ftheta(t)
        y = fomega(t)
        u = np.diff(x, append=x[-1])
        v = np.diff(y, append=x[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        if not auto_limits:
            r = max(abs(max(x)), abs(min(x)), abs(max(y)), abs(min(y)))

            dq = atomx.dq
            rx = r + (dq - r % dq) + dq * 5

            dx = rx*0.2 + (dq - rx*0.2 % dq)
            x_major_ticks = np.arange(-rx, rx, dx)
            x_minor_ticks = np.arange(-rx, rx, dx*0.2)

            dq = atomy.dq
            ry = r + (dq - r % dq) + dq * 5

            dy = ry*0.2 + (dq - ry*0.2 % dq)
            y_major_ticks = np.arange(-ry, ry, dy)
            y_minor_ticks = np.arange(-ry, ry, dy*0.2)

            ax.set_xticks(x_major_ticks)
            ax.set_xticks(x_minor_ticks, minor=True)
            ax.set_yticks(y_major_ticks)
            ax.set_yticks(y_minor_ticks, minor=True)

            plt.xlim([-rx, rx])
            plt.ylim([-ry, ry])

        if ss_region:
            dq = sqrt(atomx.dq**2 + atomx.dq**2) * self.Km
            region= plt.Circle((atomx.xf, atomy.xf), dq, color='k', alpha=0.2)
            ax.add_artist(region)

        if arrows:
            ax.quiver(x[:-1], y[:-1], u[:-1], v[:-1], color="tab:red",
                   units="dots", width=1, headwidth=10, headlength=10, label="qss")

            ax.plot(x, y, color="tab:red", linestyle="-")
        else:
            ax.plot(x, y, color="tab:red", linestyle="-", label="qss")

        ax.plot(atomx.xode, atomy.xode, color="tab:blue", linestyle="--", alpha=0.4, label="ode")

        ax.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
        ax.grid(b=True, which="minor", color="k", alpha=0.1, linestyle="-")

        plt.xlabel(atomx.full_name() + " ({})".format(atomx.units))
        plt.ylabel(atomy.full_name() + " ({})".format(atomy.units))

        ax.set_aspect("equal")

        plt.legend()
        plt.show()

    def plotxyt(self, atomx, atomy, arrows=True, ss_region=False):

        fx = interp1d(atomx.tout, atomx.qout, kind='zero')
        fy = interp1d(atomy.tout, atomy.qout, kind='zero')

        tboth = np.concatenate((atomx.tout, atomy.tout))
        tsort = np.sort(tboth)
        t = np.unique(tsort)

        x = fx(t)
        y = fy(t)
        u = np.diff(x, append=x[-1])
        v = np.diff(y, append=x[-1])

        fig = plt.figure()

        ax = plt.axes(projection="3d")

        dq = sqrt(atomx.dq**2 + atomx.dq**2) * self.Km

        def cylinder(center, r, l):
            x = np.linspace(0, l, 100)
            theta = np.linspace(0, 2*pi, 100)
            theta_grid, x_grid = np.meshgrid(theta, x)
            y_grid = r * np.cos(theta_grid) + center[0]
            z_grid = r * np.sin(theta_grid) + center[1]
            return x_grid, y_grid, z_grid

        Xc, Yc, Zc = cylinder((0.0, 0.0), 0.1, t[-1])

        ax.plot_surface(Xc, Yc, Zc, alpha=0.2)

        ax.scatter3D(t, x, y, c=t, cmap="hsv", marker=".")
        ax.plot3D(t, x, y)

        ax.plot3D(atomy.tode, atomx.xode, atomy.xode, color="tab:blue", linestyle="--", alpha=0.4, label="ode")

        ax.set_ylabel(atomx.full_name() + " ({})".format(atomx.units))
        ax.set_zlabel(atomy.full_name() + " ({})".format(atomy.units))
        ax.set_xlabel("t (s)")

        xmax = max(abs(min(x)), max(x))
        ymax = max(abs(min(y)), max(y))
        xymax = max(xmax, ymax)

        ax.set_xlim([0.0, t[-1]])
        ax.set_ylim([-xymax, xymax])
        ax.set_zlim([-xymax, xymax])

        plt.legend()

        plt.show()

    def plotxy2(self, atomsx, atomsy, arrows=True, ss_region=False):

        fx1 = interp1d(atomsx[0].tout, atomsx[0].qout, kind='zero')
        fx2 = interp1d(atomsx[1].tout, atomsx[1].qout, kind='zero')

        fy1 = interp1d(atomsy[0].tout, atomsy[0].qout, kind='zero')
        fy2 = interp1d(atomsy[1].tout, atomsy[1].qout, kind='zero')

        tall = np.concatenate((atomsx[0].tout, atomsx[1].tout,
                               atomsy[0].tout, atomsy[1].tout))

        tsort = np.sort(tall)

        t = np.unique(tsort)

        x1 = fx1(t)
        x2 = fx2(t)

        y1 = fy1(t)
        y2 = fy2(t)

        x = np.multiply(x1, x2)
        y = np.multiply(y1, y2)

        u = np.diff(x, append=x[-1])
        v = np.diff(y, append=x[-1])

        fig = plt.figure()
        ax = fig.add_subplot(111)

        r = max(abs(max(x)), abs(min(x)), abs(max(y)), abs(min(y)))

        dq = atomsx[0].dq
        rx = r + (dq - r % dq) + dq * 5

        dx = rx*0.2 + (dq - rx*0.2 % dq)
        x_major_ticks = np.arange(-rx, rx, dx)
        x_minor_ticks = np.arange(-rx, rx, dx*0.2)

        dq = atomsy[0].dq
        ry = r + (dq - r % dq) + dq * 5

        dy = ry*0.2 + (dq - ry*0.2 % dq)
        y_major_ticks = np.arange(-ry, ry, dy)
        y_minor_ticks = np.arange(-ry, ry, dy*0.2)

        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_major_ticks)
        ax.set_yticks(y_minor_ticks, minor=True)

        #plt.xlim([-rx, rx])
        #plt.ylim([-ry, ry])

        #if ss_region:
        #    dq = sqrt(atomx.dq**2 + atomx.dq**2) * self.Km
        #    region= plt.Circle((atomx.xf, atomy.xf), dq, color='k', alpha=0.2)
        #    ax.add_artist(region)

        if arrows:
            ax.quiver(x[:-1], y[:-1], u[:-1], v[:-1], color="tab:red",
                   units="dots", width=1, headwidth=10, headlength=10, label="qss")

            ax.plot(x, y, color="tab:red", linestyle="-")
        else:
            ax.plot(x, y, color="tab:red", linestyle="-", label="qss")

        xode = np.multiply(atomsx[0].xode, atomsx[1].xode)
        yode = np.multiply(atomsy[0].xode, atomsy[1].xode)

        ax.plot(xode, yode, color="tab:blue", linestyle="--", alpha=0.4, label="ode")

        #ax.grid(b=True, which="major", color="k", alpha=0.3, linestyle="-")
        #ax.grid(b=True, which="minor", color="k", alpha=0.1, linestyle="-")

        plt.xlabel("{} * {}".format(atomsx[0].name, atomsx[0].name))
        plt.ylabel("{} * {}".format(atomsy[0].name, atomsy[0].name))

        #ax.set_aspect("equal")

        plt.legend()
        plt.show()

    def __repr__(self):

        return self.name

    def __str__(self):

        return self.name


# ============================ Interfaces ======================================


class Device(object):

    """Collection of Atoms and Connections that comprise a device
    """

    def __init__(self, name):

        self.name = name
        self.atoms = []
        self.ports = []

    def add_atom(self, atom):

        self.atoms.append(atom)
        atom.device = self
        setattr(self, atom.name, atom)

    def add_atoms(self, *atoms):

        for atom in atoms:
            self.add_atom(atom)

    def add_port(self, name, typ, *connections):

        port = Port(name, typ, *connections)
        self.ports.append(port)
        port.device = self
        setattr(self, name, port)

    def setup_connections(self):

        pass

    def setup_functions(self):

        pass

    def __repr__(self):

        return self.name

    def __str__(self):

        return __repr__(self)


class Connection(object):

    """Connection between atoms.
    """

    def __init__(self, atom=None, other=None, coefficient=1.0, coeffunc=None,
                 valfunc=None, dvalfunc=None):

        self.atom = atom
        self.other = other
        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.valfunc = valfunc
        self.dvalfunc = dvalfunc

        self.device = None

        if atom and other:
            self.reset_atoms(atom, other)

    def reset_atoms(self, atom, other):

        self.atom = atom
        self.other = other

        self.other.broadcast_to.append(self.atom)

    def compute_coefficient(self):

        if self.coeffunc:
            return self.coeffunc(self.device)
        else:
            return self.coefficient

    def value(self):

        if self.other:

            if self.valfunc:

                return self.valfunc(self.other)

            else:

                return self.compute_coefficient() * self.other.q

                #if isinstance(self.other, StateAtom):
                #    return self.compute_coefficient() * self.other.q
                #
                #elif isinstance(self.other, SourceAtom):
                #    return self.compute_coefficient() * self.other.dint()
        else:

            return 0.0

    def dvalue(self):

        if self.other:

            if self.dvalfunc:

                return self.dvalfunc(self.other)

            else:

                return self.compute_coefficient() * self.other.dx

                #if isinstance(self.other, StateAtom):
                #    return self.compute_coefficient() * self.other.dx
                #
                #elif isinstance(self.other, SourceAtom):
                #    return self.compute_coefficient() * self.other.f(self.other.q)
        else:

            return 0.0


class PortConnection(object):

    def __init__(self, invar, outvar, state=None, sign=1, expr=""):

        self.invar = invar
        self.outvar = outvar
        self.state = state
        self.sign = sign
        self.expr = expr
        self.from_connections = []
        self.port = None


class Port(object):

    def __init__(self, name, typ="in", *connections):

        self.name = name
        self.typ = typ
        self.device = None

        if connections:
            self.connections = connections
            for connection in self.connections:
                connection.port = self
        else:
            self.connections = []

    def connect(self, other):

        if self.typ == "in":
            self.connections[0].from_connections.append(other.connections[0])

        elif self.typ == "out":
            other.connections[0].from_connections.append(self.connections[0])

        elif self.typ in ("inout"):
            self.connections[0].from_connections.append(other.connections[0])
            other.connections[0].from_connections.append(self.connections[0])

        elif self.typ in ("dq"):
            self.connections[0].from_connections.append(other.connections[0])
            other.connections[0].from_connections.append(self.connections[0])
            self.connections[1].from_connections.append(other.connections[1])
            other.connections[1].from_connections.append(self.connections[1])

        elif self.typ in ("abc"):
            self.connections[0].from_connections.append(other.connections[0])
            other.connections[0].from_connections.append(self.connections[0])
            self.connections[1].from_connections.append(other.connections[1])
            other.connections[1].from_connections.append(self.connections[1])
            self.connections[2].from_connections.append(other.connections[2])
            other.connections[2].from_connections.append(self.connections[2])


class SymbolicDevice(Device):

    def __init__(self, name):

        Device.__init__(self, name)

        self.states = odict()
        self.constants = odict()
        self.parameters = odict()
        self.algebraic = odict()
        self.diffeq = []
        self.dermap = odict()
        self.jacobian = odict()

    def add_state(self, name, dername, desc="", units="", x0=0.0, dq=1e-3):

        self.states[name] = odict()

        self.states[name]["name"] = name
        self.states[name]["dername"] = dername
        self.states[name]["desc"] = desc
        self.states[name]["units"] = units
        self.states[name]["x0"] = x0
        self.states[name]["dq"] = dq
        self.states[name]["device"] = self

        self.states[name]["sym"] = None
        self.states[name]["dersym"] = None
        self.states[name]["expr"] = None
        self.states[name]["atom"] = None

        self.dermap[dername] = name

    def add_constant(self, name, desc="", units="", value=None):

        self.constants[name] = odict()

        self.constants[name]["name"] = name
        self.constants[name]["desc"] = desc
        self.constants[name]["units"] = units
        self.constants[name]["value"] = value

        self.constants[name]["sym"] = None

    def add_parameter(self, name, desc="", units="", value=None):

        self.parameters[name] = odict()

        self.parameters[name]["name"] = name
        self.parameters[name]["desc"] = desc
        self.parameters[name]["units"] = units
        self.parameters[name]["value"] = value

        self.parameters[name]["sym"] = None

    def add_diffeq(self, equation):

        self.diffeq.append(equation)

    def add_algebraic(self, var, rhs):

        self.algebraic[var] = rhs

    def update_parameter(self, key, value):

        self.parameters[key]["value"] = value

    def add_input_port(self, name, var, sign=1):

        connection = PortConnection(var, sign=sign)
        self.add_port(name, "in", connection)

    def add_output_port(self, name, var=None, state=None, expr=""):

        connection = PortConnection(var, state=state, sign=sign, expr=expr)
        self.add_port(name, "out", connection)

    def add_electrical_port(self, name, input, output, sign=1, expr=""):

        connection = PortConnection(input, output, sign=sign, expr=expr)
        self.add_port(name, "inout", connection)

    def add_dq_port(self, name, inputs, outputs, sign=1, exprs=None):

        expr_d = ""
        expr_q = ""

        if exprs:
            expr_d, expr_q = exprs

        connection_d = PortConnection(inputs[0], outputs[0], sign=sign, expr=expr_d)
        connection_q = PortConnection(inputs[1], outputs[1], sign=sign, expr=expr_q)

        self.add_port(name, "dq", connection_d, connection_q)

    def setup_connections(self):

        for name, state in self.states.items():

            atom = StateAtom(name, x0=state["x0"], dq=state["dq"],
                             units=state["units"])

            atom.derargfunc = self.get_args

            self.add_atom(atom)

            self.states[name]["atom"] = atom

    def setup_functions(self):

        # 1. create sympy symbols:

        x = []
        dx_dt = []

        for name, state in self.states.items():

            sym = sp.Symbol(name)
            dersym = sp.Symbol(state["dername"])

            x.append(name)
            dx_dt.append(state["dername"])

            self.states[name]["sym"] = sym
            self.states[name]["dersym"] = dersym

        for name in self.constants:
            sp.Symbol(name)

        for name in self.parameters:
            sp.Symbol(name)

        for port in self.ports:
            for connection in port.connections:
                sp.Symbol(connection.invar)

        for var in self.algebraic:
            sp.Symbol(var)

        # 2. create symbolic derivative expressions:

        # 2a. substitute algebraic equations:

        n = len(self.algebraic)
        m = len(self.diffeq)

        algebraic = [[sp.Symbol(var), sp.sympify(expr)] for var, expr in self.algebraic.items()]


        for i in range(n-1):
             for j in range(i+1, n):
                 algebraic[j][1] = algebraic[j][1].subs(algebraic[i][0], algebraic[i][1])

        diffeq = self.diffeq.copy()

        for i in range(m):
            diffeq[i] = sp.sympify(diffeq[i])
            for var, expr in algebraic:
                diffeq[i] = diffeq[i].subs(var, expr)

        # 3. solve for derivatives:

        derexprs = solve(diffeq, *dx_dt, dict=True)

        for lhs, rhs in derexprs[0].items():

            dername = str(lhs)
            statename = self.dermap[dername]
            self.states[statename]["expr"] = rhs

        # 4. create atoms:

        ext_varnames = []

        ext_varsubs = {}

        external_vars = []

        # 4.a. set up ports:

        for port in self.ports:     # todo: other port types

            if port.typ == "inout":

                sign = 1

                for connection in port.connections:

                    varname = connection.invar
                    sign = connection.sign

                    for from_connection in connection.from_connections:

                        devicename = from_connection.port.device.name
                        varname = from_connection.outvar

                        mangeld_name = "{}_{}".format(devicename, varname)
                        ext_varnames.append(mangeld_name)

                        external_vars.append(varname)

            # make a sum expression for all input symbols:

            ext_varsubs[varname] = "(" + " + ".join(ext_varnames) + ")"

            if sign == -1:
                ext_varsubs[varname] = "-" + ext_varsubs[varname]

        # 4.b.

        argstrs = (list(self.constants.keys()) +
                   list(self.parameters.keys()) +
                   list(self.states.keys()) +
                   ext_varnames)

        argstr = " ".join(argstrs)
        argsyms = sp.var(argstr)

        for name, state in self.states.items():

            expr = state["expr"]

            for var, substr in ext_varsubs.items():
                subexpr = sp.sympify(substr)
                expr = expr.subs(var, subexpr)

            state["expr"] = expr

            func = lambdify(argsyms, expr, dummify=False)

            self.states[name]["atom"].derfunc = func

        for name in self.output_ports:

            statename = self.output_ports[name]["state"]
            state = self.states[statename]
            self.output_ports[name]["atom"] = state["atom"]

        # 5. connect atoms:

        for statex in self.states.values():

            for statey in self.states.values():

                f = statex["expr"]

                if statey["sym"] in f.free_symbols:

                    # connect:
                    statex["atom"].add_connection(statey["atom"])

                    # add jacobian expr:
                    df_dy = sp.diff(f, statey["sym"])

                    func = lambdify(argsyms, df_dy, dummify=False)

                    statex["atom"].add_jacfunc(statey["atom"], func)

            for var in external_vars:

                statey = self.states[var]

                f = statex["expr"]

                mangled_name = "{}_{}".format(statey["device"].name, statey["name"])
                mangled_symbol = sp.Symbol(mangled_name)

                if mangled_symbol in f.free_symbols:

                    # connect:
                    statex["atom"].add_connection(statey["atom"])

                    # jacobian expr:
                    df_dy = sp.diff(f, mangled_symbol)

                    func = lambdify(argsyms, df_dy, dummify=False)

                    statex["atom"].add_jacfunc(statey["atom"], func)

    @staticmethod
    def get_args(self):

        args = []

        for name, constant in self.constants.items():
            args.append(float(constant["value"]))

        for name, parameter in self.parameters.items():
            args.append(float(parameter["value"]))

        for name, state in self.states.items():
            args.append(float(state["atom"].q))

        for name, port in self.input_ports.items():
            for port2 in port["ports"]:
               args.append(port2["atom"].q)

        return args


def plot_csv(*csvfiles, plot_upd=False, legloc=None, legend=True, upd_bins=1000, pth=None):

    # t,q,e,n

    c, j = 1, 1
    r = len(csvfiles)/c

    if r % c > 0.0: r += 1

    fig = plt.figure()

    for i, filepath in enumerate(csvfiles):

        plt.subplot(r, c, j)

        head, tail = os.path.split(filepath)
        filename, ext = os.path.splitext(tail)

        tout = []
        qout = []
        evals = []
        updates = []

        lines = []

        with open(filepath, "r") as f:
            lines = f.readlines()

        if not lines:
            return

        for line in lines[1:]:

            items = line.split(",")

            tout.append(float(items[0]))
            qout.append(float(items[1]))
            evals.append(int(items[2]))

            if len(items) > 3:
                updates.append(int(items[3]))
            else:
                updates.append(0)

        ax1 = plt.gca()
        ax1.set_ylabel("{}".format(filename), color='tab:red')
        ax1.grid()

        ax2 = None

        if plot_upd or plot_ss_updates:

            ax2 = ax1.twinx()
            ylabel = "update frequency (Hz)"
            ax2.set_ylabel(ylabel, color='tab:blue')

        if plot_upd:

            dt = tout[-1] / upd_bins

            label = "update frequency"
            #ax2.hist(atom.tout, upd_bins, alpha=0.5,
            #         color='b', label=label, density=True)

            n = len(tout)
            bw = n**(-2/3)
            kde = gaussian_kde(tout, bw_method=bw)
            t = np.arange(0.0, tout[-1], dt/10)
            pdensity = kde(t) * n

            ax2.fill_between(t, pdensity, 0, lw=0,
                                color='tab:blue', alpha=0.2,
                                label=label)

        #if plot_ss_updates:
        #
        #    ax2.plot(self.tout_ss, self.nupd_ss, 'tab:blue', label="ss_upds")

        if True:

            #lbl = "{} qss ({})".format(atom.full_name(), atom.units)
            lbl = "qss"

            ax1.plot(tout, qout, marker='.', markersize=4, markerfacecolor='none',
                     markeredgecolor='tab:red', markeredgewidth=0.5, alpha=1.0,
                     linestyle='none', label=lbl)

        loc = "best"

        if legloc:
            loc = legloc

        lines1, labels1 = ax1.get_legend_handles_labels()

        if ax2:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1+lines2, labels1+labels2, loc=loc)
        else:
            ax1.legend(lines1, labels1, loc=loc)

        plt.xlabel("t (s)")
        j += 1

    plt.tight_layout()

    if pth:
        fig.savefig(pth)
    else:
        plt.show()
