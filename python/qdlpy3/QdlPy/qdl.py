"""Quantized DEVS-LIM modeling and simulation framework.
"""

from math import pi, sin, cos, sqrt, floor
from collections import OrderedDict as odict

import numpy as np
import numpy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

import lti
import funcss


# ============================ Private Constants ===============================


_EPS = 1.0e-9
_INF = float('inf')
_MAXITER = 1000


# ============================ Public Constants ================================


DEF_DQ = 1.0e-6        # default delta Q
DEF_DQMIN = 1.0e-6     # default minimum delta Q (for dynamic dq mode)
DEF_DQMAX = 1.0e-6     # default maximum delta Q (for dynamic dq mode)
DEF_DQERR = 1.0e-2     # default delta Q absolute error (for dynamic dq mode)
DEF_DTMIN = 1.0e-12    # default minimum time step
DEF_DMAX = 1.0e5       # default maximum derivative (slew-rate)



# =============================== Globals ======================================


sys = None  # set by qdl.System constructor for visibility from fode function.


# ============================= Enumerations ===================================


class SourceType:

    NONE = "NONE"
    CONSTANT = "CONSTANT"
    STEP = "STEP"
    SINE = "SINE"
    PWM = "PWM"
    RAMP = "RAMP"
    FUNCTION = "FUNCTION"


class LimAtomType:

    NONE = "NONE"
    LATENCY_NODE = "LATENCY_NODE"
    LATENCY_BRANCH = "LATENCY_BRANCH"
    GROUND_NODE = "GROUND_NODE"
    CURRENT_SOURCE = "CURRENT_SOURCE"
    VOLTAGE_SOURCE = "VOLTAGE_SOURCE"


# ============================= Qdl Model ======================================

# ========================= Global Functions ===================================


def fode(x, t, sys):

    """Returns array of derivatives from state atoms. This function must reside
    outside of the System class in order to be passed as a delgate to the
    scipy ode integrator function.
    """

    dx_dt = [0.0]*sys.n

    for atom in sys.state_atoms:
        atom.q = x[atom.index]

    for atom in sys.state_atoms:
        dx_dt[atom.index] = atom.f()

    return dx_dt


# ============================== Classes =======================================


class Atom(object):

    def __init__(self, name, lim_type=LimAtomType.GROUND_NODE, x0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None, dtmin=None, dmax=1e5,
                 units="", is_linear=True):

        self.x0 = x0
        self.lim_type = lim_type
        self.name = name
        self.dq = dq
        self.dqmin = dqmin
        self.dqmax = dqmax
        self.dqerr = dqerr
        self.dtmin = dtmin
        self.dmax = dmax
        self.units = units
        self.is_linear = is_linear

        # simulation variables:

        self.qlo = 0.0   
        self.qhi = 0.0    
        self.tlast = 0.0  
        self.tnext = 0.0  
        self.x = x0    
        self.d = 0.0      
        self.d0 = 0.0     
        self.q = x0      
        self.q0 = x0     
        self.triggered = False

        # results data storage:

        # qss:
        self.tout = None  # output times quantized output
        self.qout = None  # quantized output 
        self.tzoh = None  # zero-order hold output times quantized output
        self.qzoh = None  # zero-order hold quantized output 
        self.updates = 0  # qss updates

        # state space:
        self.tout_ss = None  # state space time output
        self.xout_ss = None  # state space value output
        self.updates_ss = 0  # state space update count

        # non-linear ode:
        self.tout_ode = None  # state space time output
        self.xout_ode = None  # state space value output
        self.updates_ode = 0  # state space update count

        # atom connections:

        self.broadcast_to = []  # push updates to
        self.connections = []   # recieve updates from

        # parent object references:

        self.sys = None
        self.device = None

        self.implicit = True

    def add_connection(self, other, coefficient=1.0, coeffunc=None,
                       coefobj=None):
        
        connection = Connection(self, other, coefficient=coefficient,
                                coeffunc=coeffunc, coefobj=coefobj)

        self.connections.append(connection)

        return connection

    def initialize(self, t0):

        self.tlast = t0
        self.time = t0
        self.tnext = _INF

        # init state:                        

        self.x = self.x0
        self.q = self.x0
        self.q0 = self.x0
        self.qsave = self.x0
        self.xsave = self.x0

        # init quantizer values:

        self.dq = self.dqmin
        self.qhi = self.q + self.dq
        self.qlo = self.q - self.dq

        # init output:

        self.updates = 0
        self.tout = [self.time]
        self.qout = [self.q0]
        self.nupd = [0]
        self.tzoh = [self.time]
        self.qzoh = [self.q0]

        self.updates_ss = 0
        self.tout_ss = [self.time]
        self.xout_ss = [self.q0]
        self.nupd_ss = [0]

        self.updates_ode = 0
        self.tout_ode = [self.time]
        self.xout_ode = [self.q0]
        self.nupd_ode = [0]

    def update(self, time):

        self.time = time
        self.updates += 1
        self.triggered = False  # reset triggered flag

        self.d = self.f()
        self.d = max(self.d, -self.dmax)
        self.d = min(self.d, self.dmax)

        self.dint()
        self.quantize()
        self.ta()

        # trigger external update if quantized output changed:
        
        if self.q != self.q0:
            self.save()
            self.q0 = self.q
            self.broadcast()
            self.update_dq()

    def step(self, time):

        self.time = time
        self.updates += 1
        self.d = self.f()
        self.dint()
        self.q = self.x
        self.save()
        self.q0 = self.q

    def dint(self):

        raise NotImplementedError()

    def quantize(self):

        raise NotImplementedError()

    def ta(self):

        raise NotImplementedError()

    def f(self, q=None):

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

    def save(self):
    
        if self.time != self.tout[-1]:

            self.tout.append(self.time)           
            self.qout.append(self.q)
            self.nupd.append(self.updates)

            self.tzoh.append(self.time)           
            self.qzoh.append(self.q0)
            self.tzoh.append(self.time)           
            self.qzoh.append(self.q)

    def save_ss(self, t, x):

        self.tout_ss.append(t)           
        self.xout_ss.append(x)
        self.nupd_ss.append(self.updates_ss)
        self.updates_ss += 1

    def save_ode(self, t, x):

        self.tout_ode.append(t)           
        self.xout_ode.append(x)
        self.nupd_ode.append(self.updates_ss)
        self.updates_ode += 1

    def get_error(self, typ="l2"):

        # interpolate qss to ss time vector:
        # this function can only be called after state space AND qdl simualtions
        # are complete

        qout_interp = numpy.interp(self.tout2, self.tout, self.qout)

        if typ.lower().strip() == "l2":

            # calculate the L^2 relative error:
            #      ________________
            #     / sum((y - q)^2)
            #    /  --------------
            #  \/      sum(y^2)

            dy_sqrd_sum = 0.0
            y_sqrd_sum = 0.0

            for q, y in zip(qout_interp, self.qout2):
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

            for q, y in zip(qout_interp, self.qout2):
                dy_sqrd_sum += (y - q)**2
                y_sqrd_sum += y**2

            return sqrt(dy_sqrd_sum / len(qout_interp)) / (max(self.qout2) - min(self.qout2))


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
                return self.x0
        else:
            return self.x0

    def full_name(self):

        return self.device.name + "." + self.name

    def __repr__(self):

        return self.full_name()

    def __str__(self):

        return __repr__(self)


class SourceAtom(Atom):

    def __init__(self, name, lim_type=LimAtomType.NONE,
                 source_type=SourceType.CONSTANT, 
                 x0=0.0, x1=0.0, x2=0.0, xa=0.0, freq=0.0, phi=0.0, duty=0.0,
                 t1=0.0, t2=0.0, srcfunc=None, srcobj=None, 
                 dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e5, units="", is_linear=True):

        Atom.__init__(self, name=name, lim_type=lim_type, x0=x0, dq=dq,
                      dqmin=dqmin, dqmax=dqmax, dqerr=dqerr, dtmin=dtmin,
                      dmax=dmax, units=units, is_linear=is_linear)

        self.source_type = source_type
        self.x1 = x1
        self.x2 = x2
        self.xa = xa
        self.freq = freq
        self.phi = phi
        self.duty = duty
        self.t1 = t1
        self.t2 = t2 
        self.srcfunc = srcfunc
        self.srcobj = srcobj

        # source derived quantities:

        self.omega = 2.0 * pi * self.freq

        if self.freq:
            self.T = 1.0 / self.freq

        if self.source_type == SourceType.RAMP:
            self.x0 = self.x1

        self.ramp_slope = 0.0
        if (self.t2 - self.t1) > 0:
            self.ramp_slope = (self.x2 - self.x1) / (self.t2 - self.t1)

    def dint(self):

        if self.source_type == SourceType.FUNCTION:

            self.x = self.srcfunc(self.srcobj, self.time)

        elif self.source_type == SourceType.CONSTANT:

            self.x = self.x0

        elif self.source_type == SourceType.STEP:

            if self.time < self.t1:
                self.x = self.x0
            else:
                self.x = self.x1

        elif self.source_type == SourceType.SINE:

            if self.time >= self.t1:
                self.x = self.x0 + self.xa * sin(self.omega * self.time + self.phi)
            else:
                self.x = self.x0

        elif self.source_type == SourceType.PWM:

            pass # todo

        elif self.source_type == SourceType.RAMP:

            if self.time <= self.t1:
                self.x = self.x1
            elif self.time <= self.t2:
                self.x = self.x1 + (self.time - self.t1) * self.d 
            else:
                self.x = self.x2

        elif self.source_type == SourceType.FUNCTION:

            self.x = self.srcfunc()

        self.tlast = self.time

        return self.x

    def quantize(self):
        
        self.q = self.x
        return False

    def ta(self):

        self.tnext = _INF

        if self.source_type == SourceType.FUNCTION:

            pass

        if self.source_type == SourceType.RAMP:

            if self.time < self.t1:
                self.tnext = self.t1

            elif self.time < self.t2:
                if self.d > 0.0:
                    self.tnext = self.time + (self.q + self.dq - self.x) / self.d
                elif self.d < 0.0:
                    self.tnext = self.time + (self.q - self.dq - self.x) / self.d
                else:
                    self.tnext = _INF

            else:
                self.tnext = _INF

        elif self.source_type == SourceType.STEP:

            if self.time < self.t1:
                self.tnext = self.t1
            else:
                self.tnext = _INF

        elif self.source_type == SourceType.SINE:

            if self.time < self.t1:

                self.tnext = self.t1

            else: 

                w = self.time % self.T             # cycle time
                t0 = self.time - w                 # cycle start time
                theta = self.omega * w + self.phi  # wrapped angular position

                # value at current time w/o dc offset:
                x = self.xa * sin(2.0 * pi * self.freq * self.time)

                # determine next transition time. Saturate at +/- xa:
            
                if theta < pi/2.0:      # quadrant I
                    self.tnext = t0 + (asin(min(1.0, (x + self.dq)/self.xa))) / self.omega

                elif theta < pi:        # quadrant II
                    self.tnext = t0 + self.T/2.0 - (asin(max(0.0, (x - self.dq)/self.xa))) / self.omega

                elif theta < 3.0*pi/2:  # quadrant III
                    self.tnext = t0 + self.T/2.0 - (asin(max(-1.0, (x - self.dq)/self.xa))) / self.omega

                else:                   # quadrant IV
                    self.tnext = t0 + self.T + (asin(min(0.0, (x + self.dq)/self.xa))) / self.omega

        elif self.source_type == SourceType.FUNCTION:

            self.tnext = self.time + self.srcdt

        self.tnext = max(self.tnext, self.tlast + self.dtmin)

    def f(self, q=None):

        if not q:
            q = self.q

        d = 0.0

        if self.source_type == SourceType.RAMP:

            d = self.ramp_slope

        elif self.source_type == SourceType.SINE:

            d = self.omega * self.xa * cos(self.omega * self.time + self.phi)

        elif self.source_type == SourceType.STEP:

            pass  # todo: sigmoid approx.

        elif self.source_type == SourceType.PWM:

            pass  # todo: sigmoid approx.

        return d


class StateAtom(Atom):

    """ Qdl State Atom.
    """

    def __init__(self, name, lim_type=LimAtomType.NONE, x0=0.0, coefficient=0.0,
                 coeffunc=None, coefobj=None, derfunc=None, derobj=None,
                 dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e5, units="", is_linear=True):

        Atom.__init__(self, name=name, lim_type=lim_type, x0=x0, dq=dq, dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                      dtmin=dtmin, dmax=dmax, units=units, is_linear=is_linear)

        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.coefobj = coefobj

        self.derfunc = derfunc
        self.derobj = derobj

    def dint(self):

        self.x += self.d * (self.time - self.tlast)

        self.tlast = self.time

        return self.x

    def quantize(self):
        
        interp = False
        change = False

        self.d0 = self.d

        # derivative based:

        if self.x >= self.qhi:

            self.q = self.qhi
            self.qlo += self.dq
            change = True

        elif self.x <= self.qlo:

            self.q = self.qlo
            self.qlo -= self.dq
            change = True

        self.qhi = self.qlo + 2.0 * self.dq

        if change and self.implicit:  # we've ventured out of (qlo, qhi) bounds

            self.d = self.f()

            # if the derivative has changed signs, then we know 
            # we are in a potential oscillating situation, so
            # we will set the q such that the derivative ~= 0:

            if (self.d * self.d0) < 0:  # if derivative has changed sign
                flo = self.f(self.qlo) 
                fhi = self.f(self.qhi)
                if flo != fhi:
                    a = (2.0 * self.dq) / (fhi - flo)
                    self.q = self.qhi - a * fhi
                    interp = True

        return interp

    def ta(self):

        if self.d > 0.0:
            self.tnext = self.time + (self.qhi - self.x) / self.d
        elif self.d < 0.0:
            self.tnext = self.time + (self.qlo - self.x) / self.d
        else:
            self.tnext = _INF

        self.tnext = max(self.tnext, self.tlast + self.dtmin)

    def compute_coefficient(self):

        if self.coeffunc:
            return self.coeffunc(self.coefobj)
        else:
            return self.coefficient

    def f(self, q=None):

        if not q:
            q = self.q

        if self.derfunc:
            return self.derfunc(self.derobj, q)

        d = self.compute_coefficient() * q

        for connection in self.connections:
            d += connection.value()

        return d


class System(object):

    def __init__(self, name="sys", dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None, print_time=False):
        
        global sys
        sys = self

        self.name = name

        # qss solution parameters:

        self.dq = DEF_DQ
        if dq:
            self.dq = dq

        self.dqmin = DEF_DQMIN
        if dqmin:
            self.dqmin = dqmin
        elif dq:
            self.dqmin = dq

        self.dqmax = DEF_DQMAX
        if dqmax:
            self.dqmax = dqmax
        elif dq:
            self.dqmax = dq

        self.dqerr = DEF_DQERR
        if dqerr:
            self.dqerr = dqerr

        self.dtmin = DEF_DTMIN
        if dtmin:
            self.dtmin = dtmin

        self.dmax = DEF_DMAX
        if dmax:
            self.dmax = dmax

        # child elements:

        self.devices = []
        self.atoms = []

        # simulation variables:

        self.tstop = 0.0  # end simulation time
        self.time = 0.0   # current simulation time
        self.iprint = 0   # for runtime updates
        self.print_time = print_time

        # state space model:

        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.x0 = None
        self.x = None
        self.u0 = None
        self.n = 0
        self.m = 0
        self.dt = -1.0
        self.time_ss = 0.0
        self.dt = 1e-4
        self.eye = None
        self.state_atoms = []
        self.source_atoms = []

        self.time_ode = 0.0

        self.is_linear = True

    def add_device(self, device):

        self.devices.append(device)

        for atom in device.atoms:

            if not atom.dq:
                atom.dq = self.dq

            if not atom.dqmin:
                atom.dqmin = self.dqmin

            if not atom.dqmax:
                atom.dqmax = self.dqmax

            if not atom.dqerr:
                atom.dqerr = self.dqerr

            if not atom.dtmin:
                atom.dtmin = self.dtmin

            if not atom.dmax:
                atom.dmax = self.dmax

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

            if not atom.is_linear:
                self.is_linear = False

        setattr(self, device.name, device)

    def add_devices(self, *devices):

        for device in devices:
            self.add_device(device)

    def store_q(self):

        for atom in self.atoms:
            atom.qsave = atom.q

    def restore_q(self):

        for atom in self.atoms:
            atom.q = atom.qsave

    def store_x(self):

        for atom in self.atoms:
            atom.xsave = atom.x

    def restore_x(self):

        for atom in self.atoms:
            atom.x = atom.xsave

    def init_qss(self, t0=0.0, dc=False):

        self.is_linear = True
        for atom in self.atoms:   
            if not atom.is_linear:
                self.is_linear = False
        if dc:
            self.solve_dc()

        self.time = t0

        for atom in self.atoms:   
            atom.initialize(t0)

    def init_ss(self, t0=0.0, dt=1e-4, dc=False):

        self.time_ss = t0
        self.dt = dt

        self.is_linear = True
        for atom in self.atoms:   
            if not atom.is_linear:
                self.is_linear = False

        if dc:
            self.solve_dc()
        else:
            self.build_ss(reset_state=True)

        self.x = self.x0
        self.u = self.u0

        for atom in self.atoms:   
            atom.initialize(t0)

    def init_ode(self, t0=0.0, dt=1e-4, dc=False):

        self.time_ode = t0
        self.dt = dt

        if dc:
            self.solve_dc()
            self.x = self.x0
        else:
            self.x0 = [0.0]*self.n
            for atom in self.state_atoms:
                self.x0[atom.index] = atom.x0
            self.x = self.x0

        for atom in self.atoms:   
            atom.initialize(t0)

    def build_ss(self, reset_state=False):

        self.a = np.zeros((self.n, self.n))
        self.b = np.zeros((self.n, self.m))
        self.u0 = np.zeros((self.m, 1))

        if reset_state:
            self.x0 = np.zeros((self.n, 1))

        self.eye = np.eye(self.n)

        for atom in self.source_atoms:

            self.u0[atom.index, 0] = atom.x0

        for atom in self.state_atoms:

            i = atom.index

            if reset_state:
                self.x0[i] = atom.x0

            self.a[i, i] = atom.compute_coefficient()

            for connection in atom.connections:

                if connection.other in self.state_atoms:
                    j = connection.other.index
                    self.a[i, j] = connection.compute_coefficient()

                else:
                    k = connection.other.index
                    self.b[i, k] = connection.compute_coefficient()

        self.apr = la.inv(self.eye - self.dt * self.a)
        self.bpr = np.dot(self.apr, self.dt * self.b)

    def update_ss(self):

        for atom in self.state_atoms:

            i = atom.index
            self.a[i, i] = atom.compute_coefficient()

            for connection in atom.connections:

                if connection.other in self.state_atoms:
                    j = connection.other.index
                    self.a[i, j] = connection.compute_coefficient()

                else:
                    k = connection.other.index
                    self.b[i, k] = connection.compute_coefficient()

        self.apr = la.inv(self.eye - self.dt * self.a)
        self.bpr = np.dot(self.apr, self.dt * self.b)

    def step_ss(self):

        if not self.is_linear:
            self.update_ss()

        self.x = np.dot(self.apr, self.x) + np.dot(self.bpr, self.u)

        for atom in self.state_atoms:
            atom.save_ss(self.time_ss, self.x[atom.index, 0])

    def solve_dc(self, maxitr=1e2, maxerr=1e-6):

        """solves for the dc steady-state.
        """

        # state space method (linear only):
        #self.build_ss(reset_state=True)
        #x = la.solve(self.a, -1.0 * np.dot(self.b, self.u0))

        x0 = [0.0]*self.n

        for atom in self.state_atoms:
            x0[atom.index] = atom.x0

        x = fsolve(fode, x0, args=(0, sys))
        
        self.x0 = x

        for atom in self.state_atoms:
            atom.x0 = x[atom.index]
        
    def run_ss(self, tstop):

        self.tstop = tstop

        self.store_q()

        print("State Space Simulation started...")

        self.update_ss()

        while self.time_ss < self.tstop:
            self.step_ss()
            self.time_ss += self.dt

        print("State Space Simulation complete.")

        self.restore_q()

    def run_ode(self, tstop):

        self.tstop = tstop

        self.store_q()

        for atom in self.atoms:
            atom.dint()
            atom.quantize()

        print("Non-linear ODE Simulation started...")

        t = np.arange(self.time_ode, self.tstop, self.dt)
        y = odeint(fode, self.x, t, args=(sys,))

        for atom in self.state_atoms:
            for i, tval in enumerate(t):
                atom.save_ode(tval, y[i, atom.index])
            self.x[atom.index] = y[-1, atom.index]

        self.time_ode = self.tstop

        print("Non-linear ODE Simulation complete.")

        self.restore_q()

    def run_qss(self, tstop, verbose=False):

        self.tstop = tstop

        print("QSS Simulation started...")

        # start by updating all atoms:

        for i in range(1):
            for atom in self.atoms:
                atom.update(self.time)
                atom.save()

        # now iterate over atoms until nothing triggered:

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

        # main simulation loop:

        tlast = self.time
        last_print_time =  self.time

        while self.time < self.tstop:
            self.advance()
            if verbose and self.time-last_print_time > 0.1:
                print("t = {0:5.2f} s".format(self.time))
                last_print_time = self.time
            tlast = self.time

        # force time to tstop and do one update at time = tstop:

        self.time = self.tstop

        for atom in self.atoms:
            atom.update(self.time)
            atom.save()

        print("QSS Simulation complete.")

    def advance(self):

        tnext = _INF

        for atom in self.atoms:
            tnext = min(atom.tnext, tnext)

        self.time = max(tnext, self.time + _EPS)
        self.time = min(self.time, self.tstop)

        for atom in self.atoms:
            if atom.tnext <= self.time or self.time >= self.tstop:
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

    def plot_devices(self, *devices, plot_qss=True, plot_ss=False,
             plot_qss_updates=False, plot_ss_updates=False, legend=False):

        for device in devices:
            for atom in devices.atoms:
                atoms.append(atom)

        self.plot(self, *atoms, plot_qss=plot_qss, plot_ss=plot_ss,
                  plot_qss_updates=plot_qss_updates,
                  plot_ss_updates=plot_ss_updates, legend=legend)

    def plot(self, *atoms, plot_qss=True, plot_ss=False,
             plot_qss_updates=False, plot_ss_updates=False, legend=False):

        if not atoms:
            atoms = self.state_atoms

        c, j = 2, 1
        r = floor(len(atoms)/2) + 1

        for atom in atoms:

            ax1 = None
            ax2 = None
    
            if plot_qss or plot_ss:

                plt.subplot(r, c, j)
                ax1 = plt.gca()
                ax1.set_ylabel("{} ({})".format(atom.full_name(), atom.units), color='b')
                ax1.grid()

            if plot_qss:
                ax1.plot(atom.tzoh, atom.qzoh, 'b-', label="qss_q")

            if plot_ss:
                ax1.plot(atom.tout_ss, atom.xout_ss, 'c--', label="ss_x")

            if plot_qss_updates or plot_ss_updates:
                ax2 = ax1.twinx()
                ax2.set_ylabel('total updates', color='r')
                
            if plot_qss_updates:
                ax2.plot(atom.tout, atom.nupd, 'r-', label="qss_upds")

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

    def plot_groups(self, *groups, plot_qss=False, plot_ss=False, plot_ode=False, legend=True):

        #mpl.style.use('seaborn')

        if len(groups) > 1:
            c, j = 2, 1
            r = floor(len(groups)/2) + 1

        for group in groups:

            if len(groups) > 1:
                plt.subplot(r, c, j)

            for i, atom in enumerate(group):

                color = 'C{}'.format(i)

                if plot_qss:

                    lbl = "{} qss ({})".format(atom.full_name(), atom.units)

                    plt.plot(atom.tout, atom.qout,
                             marker='.',
                             markersize=6,
                             markerfacecolor='none',
                             markeredgecolor=color,
                             markeredgewidth=0.8,
                             linestyle='none',
                             label=lbl)

                if plot_ss:

                    lbl = "{} ss ({})".format(atom.full_name(), atom.units)

                    plt.plot(atom.tout_ss, atom.xout_ss, 
                             color=color,
                             linewidth=0.8,
                             linestyle='dashed',
                             label=lbl)

                if plot_ode:

                    lbl = "{} ode ({})".format(atom.full_name(), atom.units)

                    plt.plot(atom.tout_ode, atom.xout_ode, 
                             color=color,
                             linewidth=0.8,
                             linestyle='dashed',
                             label=lbl)

            if legend:
                plt.legend()

            plt.xlabel("t (s)")
            plt.grid()
 
            if len(groups) > 1:
                j += 1

        plt.tight_layout()
        plt.show()

    def __repr__(self):

        return self.name

    def __str__(self):

        return self.name


# ========================== Interface Model ===================================


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

    def add_port(self, port):
        
        self.ports.append(port)
        port.device = self
        setattr(self, port.name, port)

    def add_ports(self, *ports):
        
        for port in ports:
            self.add_port(port)

    def __repr__(self):

        return self.name

    def __str__(self):

        return __repr__(self)


class Port(object):

    """Connection between devices.
    """

    def __init__(self, name, atom):

        self.name = name
        self.atom = atom
        self.device = None

    def connect(self,):

        pass                                          

    def __repr__(self):

        return self.device.name + "." + self.name

    def __str__(self):

        return __repr__(self)


class Connection(object):

    """Connection between atoms.
    """

    def __init__(self, atom, other, coefficient=1.0, coeffunc=None, coefobj=None):

        self.atom = atom
        self.other = other
        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.coefobj = coefobj

        self.other.broadcast_to.append(self.atom)

    def compute_coefficient(self):

        if self.coeffunc:
            if self.coefobj:
                return self.coeffunc(self.coefobj)
        else:
            return self.coefficient                                                  

    def value(self):

        return self.compute_coefficient() * self.other.q


# ============================ Basic Devices ===================================


class GroundNode(Device):

    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.atom = SourceAtom(name="source", source_type=SourceType.CONSTANT,
                                 lim_type=LimAtomType.GROUND_NODE, x0=0.0,
                                 units="V", is_linear=True)

        self.add_atom(self.atom)


class ConstantSourceNode(Device):

    def __init__(self, name="source", v0=0.0):

        Device.__init__(self, name)

        self.atom = SourceAtom(name="source", source_type=SourceType.CONSTANT,
                                 lim_type=LimAtomType.VOLTAGE_SOURCE, x0=v0,
                                 units="V", is_linear=True)

        self.add_atom(self.atom)


class LimNode(Device):

    """Generic LIM Lantency Node with G, C, I, B and S components.
                                
                       \       
               i_i2(t)  ^   ... 
                         \   
                 i_i1(t)  \     i_ik(t)
                ----<------o------>----
                           |
                           |   ^
                           |   | i_i(t)
           .--------.------+------------.--------------.
           |        |      |    i_b =   |      i_s =   |       +
          ,-.      <.     _|_   b_k *  ,^.     s_pq * ,^.     
    H(t) ( ^ )   G <.   C ___  v_k(t) < ^ >  i_pq(t) < ^ >   v(t)
          `-'      <.      |           `.' ^          `.' ^   
           |        |      |            |   \          |   \   -
           '--------'------+------------'----\---------'    \
                          _|_                 \              '---- port_s
                           -                   '---- port_b   

    isum = -(h + i_s + i_b) + v*G + v'*C

    v'= 1/C * (isum + h + i_s + i_b - v*G) 

    """

    def __init__(self, name, c, g=0.0, h0=0.0, v0=0.0, source_type=SourceType.CONSTANT,
                 h1=0.0, h2=0.0, ha=0.0, freq=0.0, phi=0.0, duty=0.0,
                 t1=0.0, t2=0.0, dq=None, is_linear=True):

        Device.__init__(self, name)

        self.c = c
        self.g = g

        self.source = SourceAtom("source", lim_type=LimAtomType.CURRENT_SOURCE,
                                 source_type=source_type, x0=h0, x1=h1,
                                 x2=h2, xa=ha, freq=freq, phi=phi, duty=duty,
                                 t1=t1, t2=t2, dq=dq, units="A")

        self.atom = StateAtom("state", lim_type=LimAtomType.LATENCY_NODE,
                              x0=0.0, coeffunc=self.aii, coefobj=self, 
                              dq=dq, units="V")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii, coefobj=self)

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.atom.add_connection(branch.atom, coeffunc=self.aij, coefobj=self)
        elif terminal == "j":
            self.atom.add_connection(branch.atom, coeffunc=self.aji, coefobj=self)
        
    @staticmethod
    def aii(self):
        
        return -self.g / self.c

    @staticmethod
    def bii(self):
        
        return 1.0 / self.c

    @staticmethod
    def aij(self):
        
        return -1.0 / self.c

    @staticmethod
    def aji(self):
        
        return 1.0 / self.c


class LimBranch(Device):

    """Generic LIM Lantency Branch with R, L, V, T and Z components.
    
                      +                v_ij(t)              -
    
                                       i(t) --> 
    
                                     v_t(t) =        v_z(t) =
               v(t)  +   -   +   -  T_ijk * v_k(t)  Z_ijpq * i_pq(t)
    positive   ,-.     R       L         ,^.           ,^.      negative
        o-----(- +)---VVV-----UUU-------<- +>---------<- +>-------o
               `-'                       `.'           `.'  
                                          ^             ^
                                          |             |
                                       port_t         port_z
    
    vij = -(v + vt + vz) + i*R + i'*L

    i'= 1/L * (vij + v + vt + vz - i*R) 

    """

    def __init__(self, name, l, r=0.0, e0=0.0, i0=0.0,
                 source_type=SourceType.CONSTANT, e1=0.0, e2=0.0, ea=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 is_linear=True):

        Device.__init__(self, name)

        self.l = l
        self.r = r

        self.source = SourceAtom("source", lim_type=LimAtomType.VOLTAGE_SOURCE,
                                 source_type=source_type, x0=e0, x1=e1,
                                 x2=e2, xa=ea, freq=freq, phi=phi, duty=duty,
                                 t1=t1, t2=t2, dq=dq, units="V")

        self.atom = StateAtom("state", lim_type=LimAtomType.LATENCY_BRANCH,
                              x0=0.0, coeffunc=self.aii, coefobj=self, 
                              dq=dq, units="A")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii, coefobj=self)

    def connect(self, inode, jnode):

        self.atom.add_connection(inode.atom, coeffunc=self.aij, coefobj=self)
        self.atom.add_connection(jnode.atom, coeffunc=self.aji, coefobj=self)
        
    @staticmethod
    def aii(self):
        
        return -self.r / self.l

    @staticmethod
    def bii(self):
        
        return 1.0 / self.l

    @staticmethod
    def aij(self):
        
        return 1.0 / self.l

    @staticmethod
    def aji(self):
        
        return -1.0 / self.l


# ============================= DQ Devices =====================================


class GroundNodeDQ(Device):

    """
          noded    nodeq
            o        o
            |        |
      .-----+--------+-----.
      |     |        |     |
      |    _|_      _|_    |
      |     -        -     |
      '--------------------'
    
    """


    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.atomd = SourceAtom(name="vd", source_type=SourceType.CONSTANT,
                                lim_type=LimAtomType.GROUND_NODE, x0=0.0,
                                units="V", is_linear=True)

        self.atomq = SourceAtom(name="vq", source_type=SourceType.CONSTANT,
                                lim_type=LimAtomType.GROUND_NODE, x0=0.0,
                                units="V", is_linear=True)

        self.add_atoms(self.atomd, self.atomq)


class LimBranchDQ(Device):
    
    """RLV Branch DQ Dynamic Phasor Model.

                 .-------------------------------------.
                 |  vd(t)    id*(r + w*L)      id'*L   |
                 |   ,-.    +            -    +     -  |
     inodep  o---|--(- +)---------VVV-----------UUU----|---o  jnodep  
                 |   `-'             id -->            |
                 |                                     |
                 |  vq(t)    iq*(r + w*L)      iq'*L   |
                 |   ,-.    +            -    +     -  |
     inodeq  o---|--(- +)---------VVV-----------UUU----|---o  jnodeq
                 |   `-'             iq -->            |
                 |                                     |
                 '-------------------------------------'
    """

    def __init__(self, name, l, r=0.0, vd0=0.0, vq0=0.0, w=60.0*pi, id0=0.0, iq0=0.0,
                 source_type=SourceType.CONSTANT, 
                 vd1=0.0, vd2=0.0, vda=0.0, freqd=0.0, phid=0.0, dutyd=0.0,
                 td1=0.0, td2=0.0, vq1=0.0, vq2=0.0, vqa=0.0,
                 freqq=0.0, phiq=0.0, dutyq=0.0, tq1=0.0, tq2=0.0, dq=None):

        Device.__init__(self, name)

        self.l = l
        self.r = r
        self.w = w
        self.id0 = id0
        self.iq0 = iq0

        self.sourced = SourceAtom("vd", lim_type=LimAtomType.VOLTAGE_SOURCE,
                                 source_type=source_type, x0=vd0, x1=vd1,
                                 x2=vd2, xa=vda, freq=freqd, phi=phid, duty=dutyd,
                                 t1=td1, t2=td2, dq=dq, units="V")

        self.sourceq = SourceAtom("vq", lim_type=LimAtomType.VOLTAGE_SOURCE,
                                 source_type=source_type, x0=vq0, x1=vq1,
                                 x2=vq2, xa=vqa, freq=freqq, phi=phiq, duty=dutyq,
                                 t1=tq1, t2=tq2, dq=dq, units="V")

        self.atomd = StateAtom("id", lim_type=LimAtomType.LATENCY_BRANCH,
                              x0=id0, coeffunc=self.aii, coefobj=self, 
                              dq=dq, units="A")

        self.atomq = StateAtom("iq", lim_type=LimAtomType.LATENCY_BRANCH,
                              x0=iq0, coeffunc=self.aii, coefobj=self, 
                              dq=dq, units="A")

        self.add_atoms(self.sourced, self.sourceq, self.atomd, self.atomq)

        self.atomd.add_connection(self.sourced, coeffunc=self.bii, coefobj=self)
        self.atomq.add_connection(self.sourceq, coeffunc=self.bii, coefobj=self)

    def connect(self, inode, jnode):

        self.atomd.add_connection(inode.atomd, coeffunc=self.aij, coefobj=self)
        self.atomd.add_connection(jnode.atomd, coeffunc=self.aji, coefobj=self)
        self.atomq.add_connection(inode.atomq, coeffunc=self.aij, coefobj=self)
        self.atomq.add_connection(jnode.atomq, coeffunc=self.aji, coefobj=self)

    @staticmethod
    def u(self, t):

        return 0.0  # todo

    @staticmethod
    def aii(self):
        
        return -(self.r + self.w * self.l) / self.l

    @staticmethod
    def bii(self):
        
        return 1.0 / self.l

    @staticmethod
    def aij(self):
        
        return 1.0 / self.l

    @staticmethod
    def aji(self):
        
        return -1.0 / self.l


class LimNodeDQ(Device):
    
    """
                        inoded                                inodeq
                           o                                     o
                           |   | isumd                           |   | isumq  
                           |   v                                 |   v
    .----------------------+-------------------------------------+-------------.
    |                      |                                     |             |
    |          .-----------+---------.                 .---------+---------.   |
    |          |           |         |                 |         |         |   |
    |  vd *  ^ <.       ^ _|_       ,-.      vq *  ^ <.       ^ _|_       ,-.  |
    |(g+w*C) | <. vd'*C | ___ id(t)( ^ )   (g+w*C) | <. vq'*C | ___ iq(t)( ^ ) |
    |        | <.       |  |        `-'            | <.       |  |        `-'  |
    |          |           |         |                |          |         |   |
    |          '-----------+---------'                '----------+---------'   |     
    |                     _|_                                   _|_            |
    |                      -                                     -             |
    '--------------------------------------------------------------------------'
    """

    def __init__(self, name, c, g=0.0, id0=0.0, iq0=0.0, w=60.0*pi, vd0=0.0, vq0=0.0,
                 source_type=SourceType.CONSTANT, 
                 id1=0.0, id2=0.0, ida=0.0, freqd=0.0, phid=0.0, dutyd=0.0,
                 td1=0.0, td2=0.0, iq1=0.0, iq2=0.0, iqa=0.0,
                 freqq=0.0, phiq=0.0, dutyq=0.0, tq1=0.0, tq2=0.0, dq=None):

        Device.__init__(self, name)

        self.c = c
        self.g = g
        self.w = w
        self.vd0 = vd0
        self.vq0 = vq0

        self.sourced = SourceAtom("id", lim_type=LimAtomType.CURRENT_SOURCE,
                                  source_type=source_type, x0=id0, x1=id1,
                                  x2=id2, xa=ida, freq=freqd, phi=phid, duty=dutyd,
                                  t1=td1, t2=td2, dq=dq, units="A")

        self.sourceq = SourceAtom("iq", lim_type=LimAtomType.CURRENT_SOURCE,
                                  source_type=source_type, x0=iq0, x1=iq1,
                                  x2=iq2, xa=iqa, freq=freqq, phi=phiq, duty=dutyq,
                                  t1=tq1, t2=tq2, dq=dq, units="A")

        self.atomd = StateAtom("stated", lim_type=LimAtomType.LATENCY_NODE,
                               x0=vd0, coeffunc=self.aii, coefobj=self, 
                               dq=dq, units="V")

        self.atomq = StateAtom("stateq", lim_type=LimAtomType.LATENCY_NODE,
                              x0=vq0, coeffunc=self.aii, coefobj=self, 
                              dq=dq, units="V")

        self.add_atoms(self.sourced, self.sourceq, self.atomd, self.atomq)

        self.atomd.add_connection(self.sourced, coeffunc=self.bii, coefobj=self)
        self.atomq.add_connection(self.sourceq, coeffunc=self.bii, coefobj=self)

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.atomd.add_connection(branch.atomd, coeffunc=self.aij, coefobj=self)
            self.atomq.add_connection(branch.atomq, coeffunc=self.aij, coefobj=self)
        elif terminal == "j":
            self.atomd.add_connection(branch.atomd, coeffunc=self.aji, coefobj=self)
            self.atomq.add_connection(branch.atomq, coeffunc=self.aji, coefobj=self)

    @staticmethod
    def aii(self):
        
        return -(self.g + self.w * self.c) / self.c

    @staticmethod
    def bii(self):
        
        return 1.0 / self.c

    @staticmethod
    def aij(self):
        
        return -1.0 / self.c

    @staticmethod
    def aji(self):
        
        return 1.0 / self.c


class SyncMachineReducedDQ(Device):

    """Synchronous Machine Reduced DQ Model

    Includes a built-in turbine/governor physics and control model

    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, wmb=60.0*pi,
                 P=4.00, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fkq0=0.0, fkd0=0.0, ffd0=0.0, wrm0=0.0,
                 th0=0.0, dq=1e-3):

        self.name = name

        # sm params:

        self.Psm  = Psm  
        self.VLL  = VLL  
        self.wmb  = wmb 
        self.P    = P    
        self.pf   = pf  
        self.rs   = rs  
        self.Lls  = Lls 
        self.Lmq  = Lmq 
        self.Lmd  = Lmd 
        self.rkq  = rkq 
        self.Llkq = Llkq
        self.rkd  = rkd 
        self.Llkd = Llkd
        self.rfd  = rfd 
        self.Llfd = Llfd
        self.vfdb = vfdb

        # turbine/governor params:

        self.Kp = Kp
        self.Ki = Ki
        self.J  = J 

        # intial conditions:

        self.fkq0 = fkq0 
        self.fkd0 = fkd0 
        self.ffd0 = ffd0 
        self.wrm0 = wrm0 
        self.th0  = th0   

        # qdl params:

        self.dq = dq

        # call super:

        Device.__init__(self, name)

        # derived:

        self.Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)
        self.Ld = Lls + (Lmd * Llfd * Llkd) / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd)

        # atoms:

        self.fkq = StateAtom("fkq", x0=fkq0, derfunc=self.dfkq, derobj=self, units="Wb",    dq=dq)
        self.fkd = StateAtom("fkd", x0=fkd0, derfunc=self.dfkd, derobj=self, units="Wb",    dq=dq)
        self.ffd = StateAtom("ffd", x0=ffd0, derfunc=self.dffd, derobj=self, units="Wb",    dq=dq)
        self.wrm = StateAtom("wrm", x0=wrm0, derfunc=self.dwrm, derobj=self, units="rad/s", dq=dq)
        self.th  = StateAtom("th",  x0=th0,  derfunc=self.dth,  derobj=self, units="rad",   dq=dq)

        self.add_atoms(self.fkq, self.fkd, self.ffd, self.wrm, self.th)

        self.fkd.add_connection(self.ffd)
        self.ffd.add_connection(self.fkd)
        self.wrm.add_connection(self.fkq, self.fkd, self.ffd, self.th)
        self.th.add_connection(self.wrm)

        self.atomd = SourceAtom("vd", lim_type=LimAtomType.VOLTAGE_SOURCE,
                                source_type=SourceType.FUNCTION,
                                srcfunc=self.vds, srcobj=self, units="V")

        self.atomq = SourceAtom("vq", lim_type=LimAtomType.VOLTAGE_SOURCE,
                                source_type=SourceType.FUNCTION,
                                srcfunc=self.vqs, srcobj=self, units="V")

        # viewables:

        #self.vd = qdl.Atom("sm.vd", source_type=qdl.SourceType.FUNCTION, srcfunc=self.vds, srcdt=1e-1, units="V")
        #self.vq = qdl.Atom("sm.vq", source_type=qdl.SourceType.FUNCTION, srcfunc=self.vqs, srcdt=1e-1, units="V")
        #self.te = qdl.Atom("sm.te", source_type=qdl.SourceType.FUNCTION, srcfunc=self.Te,  srcdt=1e-1, units="N.m")
        #self.tm = qdl.Atom("sm.tm", source_type=qdl.SourceType.FUNCTION, srcfunc=self.Tm,  srcdt=1e-1, units="N.m")
        #
        #self.add_atoms(self.vd, self.vq, self.te, self.tm)

        # branch connections:
        self.branchd_connections = []
        self.branchq_connections = []

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            bd = self.fkd.add_connection(branch.atomd, coefficient=1.0)
            bq = self.fkq.add_connection(branch.atomq, coefficient=1.0)
        elif terminal == "j":
            bd = self.fkd.add_connection(branch.atomd, coefficient=-1.0)
            bq = self.fkq.add_connection(branch.atomq, coefficient=-1.0)

        self.branchd_connections.append(bd)
        self.branchq_connections.append(bq)

    def iqs(self):

        isum = 0.0
        for connection in self.branchq_connections:
            isum -= connection.value()
        return isum
        
    def ids(self):

        isum = 0.0
        for connection in self.branchd_connections:
            isum -= connection.value()
        return isum

    @staticmethod
    def vqs(self, t):
        return (self.rs * self.iqs() + self.wrm.q * self.Ld * self.ids() +
                self.wrm.q * self.fd())

    @staticmethod
    def vds(self, t):
        return (self.rs * self.ids() - self.wrm.q * self.Lq * self.iqs() -
                self.wrm.q * self.fq())

    def vfd(self):
        return self.vfdb

    def fq(self):
        return self.Lmq / (self.Lmq + self.Llkq) * self.fkq.q

    def fd(self):
        return (self.Lmd * (self.fkd.q / self.Llkd + self.ffd.q / self.Llfd) /
                (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))

    def fqs(self):
        return self.Lq * self.iqs() + self.fq()

    def fds(self):
        return self.Ld * self.ids() + self.fd()

    def Te(self):
        return 3.0 * self.P / 4.0 * (self.fds() * self.iqs() - self.fqs() * self.ids())

    def Tm(self, wrm):
        return self.Kp * (self.wmb - wrm) + self.Ki * self.th.q

    @staticmethod
    def dfkq(self, fkq):
        return (-self.rkq / self.Llkq * (fkq - self.Lq * self.iqs() -
                self.fq() + self.Lls * self.iqs()))
    
    @staticmethod
    def dfkd(self, fkd):
        return (-self.rkd / self.Llkd * (fkd - self.Ld * self.ids() + 
                self.fd() - self.Lls * self.ids()))

    @staticmethod
    def dffd(self, ffd):
        return (self.vfd() - self.rfd / self.Llfd * (ffd - self.Ld *
                self.ids() + self.fd() - self.Lls * self.ids()))

    @staticmethod
    def dwrm(self, wrm):
        return (self.Te() - self.Tm(wrm)) / self.J

    @staticmethod
    def dth(self, th):
        return self.wmb - self.wrm.q


# =============================== Tests ========================================


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

    sys = System(dq=1e-3)

    ground = GroundNode("ground")
    node1 = LimNode("node1", 1.0, 1.0, 0.0)
    node2 = LimNode("node2", 1.0, 1.0, 0.0)
    node3 = LimNode("node3", 1.0, 1.0, 0.0)

    branch1 = LimBranch("branch1", 1.0, 0.1, 10.0)
    branch2 = LimBranch("branch2", 1.0, 0.1, 0.0)
    branch3 = LimBranch("branch2", 1.0, 0.1, 0.0)

    sys.add_devices(ground, node1, branch1, node2, node3, branch2, branch3)

    # inode, jnode
    branch1.connect(ground, node1)
    branch2.connect(node1, node2)
    branch3.connect(node2, node3)

    node1.connect(branch1, terminal="j")
    node1.connect(branch2, terminal="i")

    node2.connect(branch2, terminal="j")
    node2.connect(branch3, terminal="i")

    node3.connect(branch3, terminal="j")

    if 1: # nl test:

        def aii_nl(self):
            return -(self.r) / self.l - self.state.q

        branch1.state.coeffunc = aii_nl
        branch1.state.is_linear = False

    tstop = 20.0
    dc = True

    #sys.init_ss(dt=1.0e-3, dc=dc)
    sys.init_ode(dt=1.0e-3, dc=dc)
    sys.init_qss(dc=dc)

    #sys.run_ss(2.0)
    sys.run_ode(2.0)
    sys.run_qss(2.0)

    node2.g = 1000.0
    
    #sys.run_ss(2.1)
    sys.run_ode(2.1)
    sys.run_qss(2.1)
    
    node2.g = 1.0
    
    #sys.run_ss(tstop)
    sys.run_ode(tstop)
    sys.run_qss(tstop)

    sys.plot_groups((node1.atom, node2.atom, node3.atom), plot_ode=True)
    sys.plot_groups((branch1.atom, branch2.atom, branch3.atom), plot_ode=True)


def test3():

    sys = System(dq=1e-3)

    ground = GroundNodeDQ("ground")
    node1 = LimNodeDQ("node1", 0.001, 0.01, 0.0, 0.0)
    branch1 = LimBranchDQ("branch1", 0.001, 0.01, 10.0, 2.0)

    sys.add_devices(ground, node1, branch1)

    branch1.connect(ground, node1)
    node1.connect(branch1, terminal='j')

    if 1: # nl test:

        def aii_nl(self):
            return -(self.r + self.l * self.w + self.id.q*0.2) / self.l

        branch1.id.coeffunc = aii_nl
        branch1.id.is_linear = False

    tstop = 0.1
    dc = True

    sys.init_ode(dt=1.0e-4, dc=dc)
    sys.init_qss(dc=dc)

    sys.run_ode(tstop*0.1)
    sys.run_qss(tstop*0.1)

    branch1.sourced.x0 = 12.0
    branch1.sourceq.x0 = 3.0

    sys.run_ode(tstop)
    sys.run_qss(tstop)

    sys.plot_groups((node1.atomd, node1.atomq), plot_ode=True)
    sys.plot_groups((branch1.atomd, branch1.atomq), plot_ode=True)


def test4():

    sys = System(dq=1e-3)

    sm = SyncMachineReducedDQ("sm")
    load = LimBranchDQ("load", 0.001, 0.01, 10.0, 2.0)
    gnd = GroundNodeDQ("gnd")   

    sys.add_devices(sm, load, gnd)

    load.connect(gnd, sm)
    sm.connect(load, terminal='j')

    tstop = 0.01
    dc = False

    sys.init_ode(dt=1.0e-4, dc=dc)
    #sys.init_qss(dc=dc)

    sys.run_ode(tstop)
    #sys.run_qss(tstop)

    sys.plot_groups((sm.fkq, sm.fkd, sm.ffd), plot_ode=True, plot_qss=False)
    sys.plot_groups((sm.wrm, sm.th), plot_ode=True, plot_qss=False)
    sys.plot_groups((load.atom, load.atom), plot_ode=True, plot_qss=False)


if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    test4()

     