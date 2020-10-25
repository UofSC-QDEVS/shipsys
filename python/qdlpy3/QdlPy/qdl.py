"""Quantized DEVS-LIM modeling and simulation framework.
"""

from math import pi, sin, cos, acos, tan, acos, atan2, sqrt, floor
from collections import OrderedDict as odict

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
import funcss


# ============================ Private Constants ===============================


_EPS = 1.0e-9
_INF = float('inf')
_MAXITER = 1000


# ============================ Public Constants ================================


#DEF_DQ = 1.0e-6        # default delta Q
#DEF_DQMIN = 1.0e-6     # default minimum delta Q (for dynamic dq mode)
#DEF_DQMAX = 1.0e-6     # default maximum delta Q (for dynamic dq mode)
#DEF_DQERR = 1.0e-2     # default delta Q absolute error (for dynamic dq mode)
DEF_DTMIN = 1.0e-12    # default minimum time step
DEF_DMAX = 1.0e5       # default maximum derivative (slew-rate)

PI_3 = pi/3.0
PI5_6 = 5.0 * pi / 6.0
PI7_6 = 7.0 * pi / 6.0



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


# ============================= Qdl Model ======================================


class Atom(object):

    def __init__(self, name, x0=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=1e10, units=""):

        self.x0 = x0
        self.name = name
        self.dq = dq
        self.dqmin = dqmin
        self.dqmax = dqmax
        self.dqerr = dqerr
        self.dtmin = dtmin
        self.dmax = dmax  # will be scaled by self.dq
        self.units = units

        # simulation variables:

        self.dq0 = self.dq
        self.qlo = 0.0   
        self.qhi = 0.0 
        self.time = 0.0
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

        # jacobian cell functions:

        self.jacfuncs = []

        # parent object references:

        self.sys = None
        self.device = None

        self.implicit = True

    def add_connection(self, other, coefficient=1.0, coeffunc=None):
        
        connection = Connection(self, other, coefficient=coefficient,
                                coeffunc=coeffunc)

        connection.device = self.device

        self.connections.append(connection)

        return connection

    def add_jacfunc(self, other, func):

        self.jacfuncs.append((other, func))

    def set_state(self, value, quantize=False):

        self.x = value

        if quantize:

            self.quantize(implicit=False)

        else:
            self.q = value
            self.qhi = self.q + self.dq
            self.qlo = self.q - self.dq

    def initialize(self, t0):

        self.tlast = t0
        self.time = t0
        self.tnext = _INF

        # init state:                        

        if isinstance(self, StateAtom):
            self.x = self.x0

        if isinstance(self, SourceAtom):
            self.dint()

        self.q = self.x
        self.q0 = self.x
        self.qsave = self.x
        self.xsave = self.x

        # init quantizer values:

        #self.dq = self.dqmin
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

        #if self.sys.enable_slewrate:
        #    self.d = max(self.d, -self.dmax*self.dq)
        #    self.d = min(self.d, self.dmax*self.dq)

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

    def save(self, force=False):
    
        if self.time != self.tout[-1] or force:

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

    def __init__(self, name, source_type=SourceType.CONSTANT, u0=0.0, u1=0.0,
                 u2=0.0, ua=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0,
                 srcfunc=None, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e10, units=""):

        Atom.__init__(self, name=name, x0=u0, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.source_type = source_type
        self.u0 = u0
        self.u1 = u1
        self.u2 = u2
        self.ua = ua
        self.freq = freq
        self.phi = phi
        self.duty = duty
        self.t1 = t1
        self.t2 = t2 
        self.srcfunc = srcfunc

        # source derived quantities:

        self.u = self.u0

        self.omega = 2.0 * pi * self.freq

        if self.freq:
            self.T = 1.0 / self.freq

        if self.source_type == SourceType.RAMP:
            self.u0 = self.u1

        self.ramp_slope = 0.0
        if (self.t2 - self.t1) > 0:
            self.ramp_slope = (self.u2 - self.u1) / (self.t2 - self.t1)

    def dint(self):

        self.u_prev = self.u

        if self.source_type == SourceType.FUNCTION:

            u = self.srcfunc(self.device, self.time)

        elif self.source_type == SourceType.CONSTANT:

            u = self.u0

        elif self.source_type == SourceType.STEP:

            if self.time < self.t1:
                u = self.u0
            else:
                u = self.u1

        elif self.source_type == SourceType.SINE:

            if self.time >= self.t1:
                u = self.u0 + self.ua * sin(self.omega * self.time + self.phi)
            else:
                u = self.u0

        elif self.source_type == SourceType.PWM:

            pass # todo

        elif self.source_type == SourceType.RAMP:

            if self.time <= self.t1:
                u = self.u1
            elif self.time <= self.t2:
                u = self.u1 + (self.time - self.t1) * self.d 
            else:
                u = self.u2

        elif self.source_type == SourceType.FUNCTION:

            u = self.srcfunc()

        if self.sys.enable_slewrate:
            if u > self.u_prev:
                self.u = min(u, self.dmax * self.dq * (self.time - self.tlast) + self.u_prev)
            elif u < self.u_prev:
                self.u = max(u, -self.dmax * self.dq * (self.time - self.tlast) + self.u_prev)
        else:
            self.u = u

        self.tlast = self.time

        self.x = self.u
        self.q = self.u

        return self.u

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
                    self.tnext = self.time + (self.q + self.dq - self.u)/self.d
                elif self.d < 0.0:
                    self.tnext = self.time + (self.q - self.dq - self.u)/self.d
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
                u = self.ua * sin(2.0 * pi * self.freq * self.time)

                # determine next transition time. Saturate at +/- xa:
            
                # quadrant I
                if theta < pi/2.0:      
                    self.tnext = (t0 + (asin(min(1.0, (u + self.dq)/self.ua)))
                                  / self.omega)

                # quadrant II
                elif theta < pi:        
                    self.tnext = (t0 + self.T/2.0
                                  - (asin(max(0.0, (u - self.dq)/self.ua)))
                                  / self.omega)

                # quadrant III
                elif theta < 3.0*pi/2:  
                    self.tnext = (t0 + self.T/2.0
                                  - (asin(max(-1.0, (u - self.dq)/self.ua)))
                                  / self.omega)

                # quadrant IV
                else:                   
                    self.tnext = (t0 + self.T
                                  + (asin(min(0.0, (u + self.dq)/self.ua)))
                                  / self.omega)

        elif self.source_type == SourceType.FUNCTION:

            pass
            #self.tnext = self.time + self.srcdt # <-- should we do this?

        self.tnext = max(self.tnext, self.tlast + self.dtmin)

    def f(self, q=None):

        if not q:
            q = self.q

        d = 0.0

        if self.source_type == SourceType.RAMP:

            d = self.ramp_slope

        elif self.source_type == SourceType.SINE:

            d = self.omega * self.ua * cos(self.omega * self.time + self.phi)

        elif self.source_type == SourceType.STEP:

            pass  # todo: sigmoid approx.

        elif self.source_type == SourceType.PWM:

            pass  # todo: sigmoid approx.

        elif self.source_type == SourceType.FUNCTION:

            d = 0.0  # todo: add a time derivative function delegate

        return d


class StateAtom(Atom):

    """ Qdl State Atom.
    """

    def __init__(self, name, x0=0.0, coefficient=0.0, coeffunc=None,
                 derfunc=None, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=1e10, units=""):

        Atom.__init__(self, name=name, x0=x0, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.derfunc = derfunc

    def dint(self):

        self.x += self.d * (self.time - self.tlast)

        self.tlast = self.time

        return self.x

    def quantize(self, implicit=True):
        
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

        if change and self.implicit and implicit:  # we've ventured out of (qlo, qhi) bounds

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
            return self.coeffunc(self.device)
        else:
            return self.coefficient

    def f(self, q=None):

        if not q:
            q = self.q

        if self.derfunc:
            return self.derfunc(self.device, q)

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

        #self.dq = DEF_DQ
        #if dq:
        #    self.dq = dq
        #
        #self.dqmin = DEF_DQMIN
        #if dqmin:
        #    self.dqmin = dqmin
        #elif dq:
        #    self.dqmin = dq
        #
        #self.dqmax = DEF_DQMAX
        #if dqmax:
        #    self.dqmax = dqmax
        #elif dq:
        #    self.dqmax = dq
        #
        #self.dqerr = DEF_DQERR
        #if dqerr:
        #    self.dqerr = dqerr
        #
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
        self.print_time = print_time
        self.dt = 1e-4
        self.enable_slewrate = False
        self.jacobian = None
        self.Km = 1.2

        # events:

        self.events = {}

    def schedule(self, func, time):

        if not time in self.events:
            self.events[time] = []

        self.events[time].append(func)

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
            self.add_device(device)

    def save_state(self):

        self.tsave = self.time

        for atom in self.atoms:
            atom.qsave = atom.q
            atom.xsave = atom.x

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
            dx_dt[atom.index] = atom.f()

        return dx_dt

    @staticmethod
    def fode2(x, t=0.0, sys=None):

        """Returns array of derivatives from state atoms. This function must be
        a static method in order to be passed as a delgate to the
        scipy ode integrator function. Note that sys is a global module variable.
        """

        y = [0.0] * sys.n

        for atom in sys.state_atoms:
            atom.q = x[atom.index]

        for atom in sys.state_atoms:
            y[atom.index] = atom.f()

        return y

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

    def initialize(self, t0=0.0, dt=1e-4, dc=False):

        self.time = t0
        self.dt = dt

        self.dq0 = np.zeros((self.n, 1))

        for atom in self.state_atoms:
            self.dq0[atom.index] = atom.dq0

        if dc:
            self.solve_dc()

        for atom in self.state_atoms:   
            atom.initialize(self.time)

        for atom in self.source_atoms:   
            atom.initialize(self.time)

    def run(self, tstop, ode=True, qss=True, verbose=True, qss_fixed_dt=None,
            ode_method="RK45", optimize_dq=False, chk_ss_delay=None):

        self.verbose = verbose
        self.calc_ss = False

        if optimize_dq or chk_ss_delay:
            self.calc_ss = True
            self.update_steadystate_distance()

        # get the event times and event function lists, sorted by time:

        sorted_events = sorted(self.events.items())

        # add the last tstop event to the lists:

        sorted_events.append((tstop, None))

        # loop through the event times and solve:
                                       
        for time, events in sorted_events:

            if self.calc_ss:
                self.calc_steadystate()

            if optimize_dq:
                self.optimize_dq()
                self.update_steadystate_distance()

            self.tstop = time

            if ode:

                print("ODE Simulation started...")

                self.save_state()
                self.enable_slewrate = False

                xi = [0.0]*self.n
                for atom in self.state_atoms:
                    xi[atom.index] = atom.x

                tspan = (self.time, self.tstop)

                soln = solve_ivp(self.fode, tspan, xi, ode_method, args=(sys,),
                                 max_step=self.dt)

                t = soln.t
                x = soln.y

                for i in range(len(t)):

                    for atom in self.state_atoms:
                        atom.q = x[atom.index, i]
                        atom.save_ode(t[i], atom.q)

                    for atom in self.source_atoms:
                        atom.save_ode(t[i], atom.dint())

                for atom in self.state_atoms:
                    xf = x[atom.index, -1]
                    atom.x = xf
                    atom.q = xf

                for atom in self.source_atoms:
                    atom.dint()
                    atom.q = atom.q

                self.time = self.tstop
                self.enable_slewrate = True

                print("Non-linear ODE Simulation completed.")

            if qss:

                print("QSS Simulation started...")

                if ode: self.restore_state()

                # start by updating all atoms:

                for atom in self.atoms:
                    atom.update(self.time)
                    atom.save(force=True)

                if qss_fixed_dt:

                    while(self.time <= self.tstop):

                        for atom in self.source_atoms:
                            atom.step(self.time)

                        for atom in self.state_atoms:
                            atom.step(self.time)

                        self.time += qss_fixed_dt

                else:
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
                    interval = (self.tstop - self.time) * 0.02

                    chk_ss_clock = 0.0

                    while self.time < self.tstop:

                        self.advance()

                        if verbose and self.time-last_print_time > interval:
                            print("t = {0:5.2f} s".format(self.time))
                            last_print_time = self.time

                        if chk_ss_delay:

                            chk_ss_clock += self.time - tlast

                            if not self.check_steadystate(apply_if_true=False):
                                chk_ss_clock = 0.0

                            if chk_ss_clock >= chk_ss_delay:
                                self.check_steadystate(apply_if_true=True)

                        tlast = self.time

                    self.time = self.tstop

                    for atom in self.atoms:
                        atom.update(self.time)
                        atom.save()

                print("QSS Simulation completed.")

            if events:

                for event in events:
                    event(self)

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

        QQ0 = np.square(self.dq0)

        JJ1 = np.square(self.jac1)
        QQ1 = la.solve(JJ1, QQ0)
        dq1 = np.sqrt(np.abs(QQ1))

        if self.verbose:

            print("at event:")
            print("J = {}\n".format(self.jac1))
            print("J1*J1 = {}\n".format(JJ1))
            print("dq0.**2 = {}\n".format(QQ0))
            print("dq1.**2 = {}\n".format(QQ1))
            print("dq1 = {}\n".format(dq1))

        JJ2 = np.square(self.jac2)
        QQ2 = la.solve(JJ2, QQ0)
        dq2 = np.sqrt(np.abs(QQ2))

        if self.verbose:

            print("at steady-state:")
            print("J2 = {}\n".format(self.jac2))
            print("J2*J2 = {}\n".format(JJ2))
            print("dq0.**2 = {}\n".format(QQ0))
            print("dq2.**2 = {}\n".format(QQ2))
            print("dq2 = {}\n".format(dq2))

        for atom in self.state_atoms:

            atom.dq = min(atom.dq0, dq1[atom.index, 0], dq2[atom.index, 0])
            #atom.dq = min(dq1[atom.index, 0], dq2[atom.index, 0])

            atom.qhi = atom.q + atom.dq
            atom.qlo = atom.q - atom.dq

            if self.verbose:
                print("dq_{} = {} ({})\n".format(atom.full_name(), atom.dq, atom.units))

    def check_steadystate(self, apply_if_true=True):

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
                atom.dint()
                atom.q = atom.x

        return is_ss

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

    def plot_old(self, *atoms, plot_qss=True, plot_ss=False,
             plot_qss_updates=False, plot_ss_updates=False, legend=False):

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

            if plot_qss_updates or plot_ss_updates:
                ax2 = ax1.twinx()
                ax2.set_ylabel('updates', color='r')

            if plot_qss:
                ax1.plot(atom.tzoh, atom.qzoh, 'b-', label="qss_q")

            if plot_ss:
                ax1.plot(atom.tout_ss, atom.xout_ss, 'c--', label="ss_x")
                
            if plot_qss_updates:
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
        r = len(groups)/c

        if r % c > 0.0: r += 1

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

                    plt.plot(atom.tout_ode, atom.xout_ode, 
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
             plot_qss_updates=False, plot_ss_updates=False, legloc=None,
             plot_ode_updates=False, legend=True, errorband=False, upd_bins=1000):

        c, j = 1, 1
        r = len(atoms)/c

        if r % c > 0.0: r += 1

        for i, atom in enumerate(atoms):

            plt.subplot(r, c, j)

            ax1 = plt.gca()
            ax1.set_ylabel("{} ({})".format(atom.full_name(),
                            atom.units), color='tab:red')
            ax1.grid()

            ax2 = None

            if plot_qss_updates or plot_ss_updates:

                ax2 = ax1.twinx()
                ylabel = "updates density ($s^{-1}$)"
                ax2.set_ylabel(ylabel, color='tab:blue')

            if plot_qss_updates:

                dt = atom.tout[-1] / upd_bins

                label = "update density"
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
                lbl = "qss"

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
                
                lbl = "qss (zoh)"

                ax1.plot(atom.tzoh, atom.qzoh, color="tab:red", linestyle="-",
                         alpha=0.5, label=lbl)

            if plot_ss:

                lbl = "ss"

                ax1.plot(atom.tout_ss, atom.xout_ss, 
                         color='r',
                         linewidth=1.0,
                         linestyle='dashed',
                         label=lbl)

            if plot_ode:

                lbl = "ode"

                if errorband:

                    xhi = [x + atom.dq0 for x in atom.xout_ode]
                    xlo = [x - atom.dq0 for x in atom.xout_ode]


                    ax1.plot(atom.tout_ode, atom.xout_ode, 
                             color='k',
                             alpha=0.6,
                             linewidth=1.0,
                             linestyle='dashed',
                             label=lbl)

                    lbl = "error band"

                    ax1.fill_between(atom.tout_ode, xhi, xlo, color='k', alpha=0.1, 
                                     label=lbl)

                else:

                    ax1.plot(atom.tout_ode, atom.xout_ode, 
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
        plt.show()

    def plotxy(self, atomx, atomy, arrows=True, ss_region=False):

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

        ax.plot(atomx.xout_ode, atomy.xout_ode, color="tab:blue", linestyle="--", alpha=0.4, label="ode")

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
        
        ax.plot3D(atomy.tout_ode, atomx.xout_ode, atomy.xout_ode, color="tab:blue", linestyle="--", alpha=0.4, label="ode")
        
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

        xode = np.multiply(atomsx[0].xout_ode, atomsx[1].xout_ode)
        yode = np.multiply(atomsy[0].xout_ode, atomsy[1].xout_ode)

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

    def __init__(self, atom=None, other=None, coefficient=1.0, coeffunc=None, valfunc=None):

        self.atom = atom
        self.other = other

        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.valfunc = valfunc

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
                if isinstance(self.other, StateAtom):
                    return self.compute_coefficient() * self.other.q

                elif isinstance(self.other, SourceAtom):
                    return self.compute_coefficient() * self.other.dint()
        else:
            return 0.0


# ============================ Basic Devices ===================================


class GroundNode(Device):

    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.atom = SourceAtom(name="source", source_type=SourceType.CONSTANT,
                               u0=0.0, units="V", dq=1.0)

        self.add_atom(self.atom)


class ConstantSourceNode(Device):

    def __init__(self, name="source", v0=0.0):

        Device.__init__(self, name)

        self.atom = SourceAtom(name="source", source_type=SourceType.CONSTANT,
                               u0=v0, units="V", dq=1.0)

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

    def __init__(self, name, c, g=0.0, h0=0.0, v0=0.0,
                 source_type=SourceType.CONSTANT, h1=0.0, h2=0.0, ha=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None):

        Device.__init__(self, name)

        self.c = c
        self.g = g

        self.source = SourceAtom("source", source_type=source_type, u0=h0,
                                 u1=h1, u2=h2, ua=ha, freq=freq, phi=phi,
                                 duty=duty, t1=t1, t2=t2, dq=dq, units="A")

        self.atom = StateAtom("v", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="V")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii)

        self.voltage = self.atom

        self.atom.add_jacfunc(self.atom, self.aii)

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.atom.add_connection(branch.atom, coeffunc=self.aij)
            self.atom.add_jacfunc(branch.atom, self.aij)

        elif terminal == "j":
            self.atom.add_connection(branch.atom, coeffunc=self.aji)
            self.atom.add_jacfunc(branch.atom, self.aji)
        
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
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None):

        Device.__init__(self, name)

        self.l = l
        self.r = r

        self.source = SourceAtom("source", source_type=source_type, u0=e0,
                                 u1=e1, u2=e2, ua=ea, freq=freq, phi=phi,
                                 duty=duty, t1=t1, t2=t2, dq=dq, units="V")

        self.atom = StateAtom("i", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="A")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii)

        self.i = self.atom

        self.atom.add_jacfunc(self.atom, self.aii)

    def connect(self, inode, jnode):

        self.atom.add_connection(inode.atom, coeffunc=self.aij)
        self.atom.add_connection(jnode.atom, coeffunc=self.aji)

        self.atom.add_jacfunc(inode.atom, self.aij)
        self.atom.add_jacfunc(jnode.atom, self.aji)
        
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
          noded     nodeq
            o         o
            |         |
            '----+----'
                _|_
                 - 
    """


    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.atomd = SourceAtom(name="vd", source_type=SourceType.CONSTANT,
                                u0=0.0, units="V", dq=1.0)

        self.atomq = SourceAtom(name="vq", source_type=SourceType.CONSTANT,
                                u0=0.0, units="V", dq=1.0)

        self.add_atoms(self.atomd, self.atomq)


class SourceNodeDQ(Device):

    """
          noded       nodeq
            o           o
            |           |          
        +  ,-.      +  ,-.
       Vd (   )    Vq (   )
        -  `-'      -  `-'
            |           |
            '-----+-----'
                 _|_
                  - 
    """


    def __init__(self, name, voltage, th0=0.0):

        Device.__init__(self, name)

        self.v = voltage

        self.vd0 = v * sin(th0)
        self.vq0 = v * cos(th0)
        self.th_atom = None

        self.vd = SourceAtom(name="vd", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_vd,
                             u0=self.vd0, units="V", dq=1.0)

        self.vq = SourceAtom(name="vq", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_vq,
                             u0=self.vq0, units="V", dq=1.0)

        self.add_atoms(self.vd, self.vq)

        self.atomd = self.vd
        self.atomq = self.vq

    def connect_theta(self, th_atom):

        self.th_atom = th_atom

    @staticmethod
    def get_vd(self, t):

        if self.th_atom:
            return self.v * cos(self.th_atom.q)
        else:
            return self.vd0

    @staticmethod
    def get_vq(self, t):

        if self.th_atom:
            return self.v * sin(self.th_atom.q)
        else:
            return self.vq0


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

    def __init__(self, name, l, r=0.0, vd0=0.0, vq0=0.0, w=60.0*pi, id0=0.0,
                 iq0=0.0, source_type=SourceType.CONSTANT, 
                 vd1=0.0, vd2=0.0, vda=0.0, freqd=0.0, phid=0.0, dutyd=0.0,
                 td1=0.0, td2=0.0, vq1=0.0, vq2=0.0, vqa=0.0,
                 freqq=0.0, phiq=0.0, dutyq=0.0, tq1=0.0, tq2=0.0, dq=1e0):

        Device.__init__(self, name)

        self.l = l
        self.r = r
        self.w = w
        self.id0 = id0
        self.iq0 = iq0

        dmax = 1e8

        self.sourced = SourceAtom("vd", source_type=source_type, u0=vd0, u1=vd1,
                                 u2=vd2, ua=vda, freq=freqd, phi=phid, dmax=dmax,
                                 duty=dutyd, t1=td1, t2=td2, dq=dq, units="V")

        self.sourceq = SourceAtom("vq", source_type=source_type, u0=vq0, u1=vq1,
                                 u2=vq2, ua=vqa, freq=freqq, phi=phiq, dmax=dmax,
                                 duty=dutyq, t1=tq1, t2=tq2, dq=dq, units="V")

        self.atomd = StateAtom("id", x0=id0, coeffunc=self.aii, dq=dq, dmax=dmax,
                               units="A")

        self.atomq = StateAtom("iq", x0=iq0, coeffunc=self.aii, dq=dq, dmax=dmax,
                               units="A")

        self.add_atoms(self.sourced, self.sourceq, self.atomd, self.atomq)

        self.atomd.add_connection(self.sourced, coeffunc=self.bii)
        self.atomq.add_connection(self.sourceq, coeffunc=self.bii)

    def connect(self, inodedq, jnodedq):

        self.atomd.add_connection(inodedq.atomd, coeffunc=self.aij)
        self.atomd.add_connection(jnodedq.atomd, coeffunc=self.aji)
        self.atomq.add_connection(inodedq.atomq, coeffunc=self.aij)
        self.atomq.add_connection(jnodedq.atomq, coeffunc=self.aji)

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

    def __init__(self, name, c, g=0.0, id0=0.0, iq0=0.0, w=60.0*pi, vd0=0.0,
                 vq0=0.0, source_type=SourceType.CONSTANT, id1=0.0, id2=0.0,
                 ida=0.0, freqd=0.0, phid=0.0, dutyd=0.0, td1=0.0, td2=0.0,
                 iq1=0.0, iq2=0.0, iqa=0.0, freqq=0.0, phiq=0.0, dutyq=0.0,
                 tq1=0.0, tq2=0.0, dq=None):

        Device.__init__(self, name)

        self.c = c
        self.g = g
        self.w = w
        self.vd0 = vd0
        self.vq0 = vq0

        self.hd = SourceAtom("hd", source_type=source_type, u0=id0, u1=id1,
                                  u2=id2, ua=ida, freq=freqd, phi=phid,
                                  duty=dutyd, t1=td1, t2=td2, dq=dq, units="A")

        self.hq = SourceAtom("hq", source_type=source_type, u0=iq0, u1=iq1,
                                  u2=iq2, ua=iqa, freq=freqq, phi=phiq,
                                  duty=dutyq, t1=tq1, t2=tq2, dq=dq, units="A")

        self.vd = StateAtom("vd", x0=vd0, coeffunc=self.aii, dq=dq, units="V")

        self.vq = StateAtom("vq", x0=vq0, coeffunc=self.aii, dq=dq, units="V")

        self.add_atoms(self.hd, self.hq, self.vd, self.vq)

        self.vd.add_connection(self.hd, coeffunc=self.bii)
        self.vq.add_connection(self.hq, coeffunc=self.bii)

        self.atomd = self.vd
        self.atomq = self.vq

    def connect(self, device, terminal="i"):

        if terminal == "i":
            self.vd.add_connection(device.atomd, coeffunc=self.aij)
            self.vq.add_connection(device.atomq, coeffunc=self.aij)
        elif terminal == "j":
            self.vd.add_connection(device.atomd, coeffunc=self.aji)
            self.vq.add_connection(device.atomq, coeffunc=self.aji)

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


class SyncMachineDQ(Device):

    """Synchronous Machine Reduced DQ Model as voltage source

    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, ws=60.0*pi,
                 P=4.00, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fkq0=0.0, fkd0=0.0, ffd0=0.0, wr0=60.0*pi,
                 th0=0.0, iqs0=0.0, ids0=0.0, dq_flux=1e-2, dq_wr=1e-1, dq_th=1e-3,
                 dq_v=1e0):

        self.name = name

        # sm params:

        self.Psm  = Psm  
        self.VLL  = VLL  
        self.ws   = ws  
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
        self.wr0 = wr0 
        self.th0  = th0  
        
        dmax = 1e8

        # call super:

        Device.__init__(self, name)

        # derived:

        self.Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)

        self.Ld = (Lls + (Lmd * Llfd * Llkd)
                   / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd))

        # source atoms:

        self.atomd = SourceAtom("vd", source_type=SourceType.FUNCTION, 
                                srcfunc=self.vds, dq=dq_v, units="V", dmax=dmax)

        self.atomq = SourceAtom("vq", source_type=SourceType.FUNCTION, 
                                srcfunc=self.vqs, dq=dq_v, units="V", dmax=dmax)

        # state atoms:

        self.fkq = StateAtom("fkq", x0=fkq0, derfunc=self.dfkq, units="Wb", dq=dq_flux, dmax=dmax)
        self.fkd = StateAtom("fkd", x0=fkd0, derfunc=self.dfkd, units="Wb", dq=dq_flux, dmax=dmax)
        self.ffd = StateAtom("ffd", x0=ffd0, derfunc=self.dffd, units="Wb", dq=dq_flux, dmax=dmax)
        self.wr = StateAtom("wr", x0=wr0, derfunc=self.dwr, units="rad/s", dq=dq_wr, dmax=dmax)
        self.th = StateAtom("th", x0=th0, derfunc=self.dth, units="rad", dq=dq_th, dmax=dmax)

        self.add_atoms(self.atomd, self.atomq, self.fkq, self.fkd, self.ffd,
                       self.wr, self.th)

        # vds:
        self.atomd.add_connection(self.wr)
        self.atomd.add_connection(self.fkd)
        self.atomd.add_connection(self.ffd)

        # vqs:
        self.atomq.add_connection(self.wr)
        self.atomq.add_connection(self.fkq)

        self.fkd.add_connection(self.ffd)

        self.ffd.add_connection(self.fkd)

        self.wr.add_connection(self.fkq)
        self.wr.add_connection(self.fkd)
        self.wr.add_connection(self.ffd)
        self.wr.add_connection(self.th)

        self.th.add_connection(self.wr)

        # for branch connections:
        self.connectiond = []
        self.connectionq = []

    def connect(self, branch, terminal="i"):

        if terminal == "i":

            self.connectiond.append(self.atomd.add_connection(branch.atomd, 1.0))
            self.connectionq.append(self.atomq.add_connection(branch.atomq, 1.0))

        elif terminal == "j":

            self.connectiond.append(self.atomd.add_connection(branch.atomd, -1.0))
            self.connectionq.append(self.atomq.add_connection(branch.atomq, -1.0))

        self.fkq.add_connection(branch.atomq)
        self.fkd.add_connection(branch.atomd)
        self.ffd.add_connection(branch.atomd)
        self.wr.add_connection(branch.atomd)
        self.wr.add_connection(branch.atomq)

    def ids(self):
        if self.connectiond: 
            return sum([x.value() for x in self.connectiond])
        else:
            return 0.0
        
    def iqs(self):
        if self.connectionq: 
            return sum([x.value() for x in self.connectionq])
        else:
            return 0.0

    def vfd(self):
        return self.vfdb

    def fq(self, fkq):
        return self.Lmq / (self.Lmq + self.Llkq) * fkq

    def fd(self, fkd, ffd):
        return (self.Lmd * (fkd / self.Llkd + ffd / self.Llfd) /
                (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))

    def fqs(self, fkq):
        return self.Lq * self.iqs() + self.fq(fkq)

    def fds(self, fkd, ffd):
        return self.Ld * self.ids() + self.fd(fkd, ffd)

    def Te(self, fkq, fkd, ffd):
        return (3.0 * self.P / 4.0 * (self.fds(fkd, ffd) * self.iqs() 
                                      - self.fqs(fkq) * self.ids()))

    def Tm(self, wr, th):
        return -(self.Kp * (self.ws - wr) + self.Ki * th)

    @staticmethod
    def vqs(self, t):  # vqs relies on wr, fkd, ffd, ext_d, ext_q
        return (self.rs * self.iqs() + self.wr.q * self.Ld * self.ids() +
                self.wr.q * self.fd(self.fkd.q, self.ffd.q))

    @staticmethod
    def vds(self, t):  # vds relies on wr, fkq, ext_d, ext_q
        return (self.rs * self.ids() - self.wr.q * self.Lq * self.iqs() +
                self.wr.q * self.fq(self.fkq.q))
    
    @staticmethod      
    def dfkq(self, fkq):  # fkq relies on ext_q
        return (-self.rkq / self.Llkq * (fkq - self.Lq * self.iqs() -
                self.fq(fkq) + self.Lls * self.iqs()))

    @staticmethod
    def dfkd(self, fkd):  # fkd relies on ffd, ext_d
        return (-self.rkd / self.Llkd * (fkd - self.Ld * self.ids() + 
                self.fd(fkd, self.ffd.q) - self.Lls * self.ids()))

    @staticmethod
    def dffd(self, ffd):  # ffd relies on fkd, ext_d
        return (self.vfd() - self.rfd / self.Llfd * (ffd - self.Ld *
                self.ids() + self.fd(self.fkd.q, ffd) - self.Lls *
                self.ids()))

    @staticmethod
    def dwr(self, wr):  # wr relies on fkq, fkd, ffd, wr, ext_q, ext_d
        return (self.Te(self.fkq.q, self.fkd.q, self.ffd.q)
                - self.Tm(wr, self.th.q)) / self.J

    @staticmethod
    def dth(self, th): # th relies on wr
        return self.ws - self.wr.q


class SyncMachineDQ2(Device):

    """Synchronous Machine Reduced DQ Model as voltage source behind impedance.

    
    [ vqs ] = [ Rs     wr*Ld ] * [ iqs ] + [ Lls  0   ] * [ diqs_dt ] + [ wr*fd ]
    [ vds ]   [-wr*Lq  Rs    ]   [ ids ]   [ 0    Lls ]   [ dids_dt ]   [-wr*fq ]

    vqs = Rs*iqs + wr*Ld*ids + Lls*diqs + wr*fd
    vds = Rs*ids - wr*Lq*iqs + Lls*dids - wr*fq

    diqs = (1/Lls) * (vqs - Rs*iqs - wr*Ld*ids - wr*fd)
    dids = (1/Lls) * (vds - Rs*ids + wr*Lq*iqs + wr*fq)  

                               wr*Ld*ids 
                  Rs     Lls      ,^.
            .----VVVV----UUUU----<- +>----o (j)
            |         --->        `.'     +
          +,-.        iqs                vqs    
    wr*fq (   )                           -
          -`-'                            o
     (i)   _|_                           _|_
            -                             -

                               wr*Lq*iqs
                  Rs     Lls      ,^.
            .----VVVV----UUUU----<+ ->----o (j)
            |         --->        `.'     +
          +,-.        ids                vds   
    wr*fd (   )                           -
          -`-'                            o
      (i)  _|_                           _|_
            -                             -


    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, ws=60.0*pi,
                 P=4.00, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fkq0=0.0, fkd0=0.0, ffd0=0.0, wr0=60.0*pi,
                 th0=0.0, iqs0=0.0, ids0=0.0, dq_current=1e-1, dq_flux=1e-2,
                 dq_wr=1e-1, dq_th=1e-3, dq_v=1e0):

        self.name = name

        # sm params:

        self.Psm  = Psm  
        self.VLL  = VLL  
        self.ws   = ws  
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

        self.ids0 = ids0
        self.iqs0 = iqs0
        self.fkq0 = fkq0 
        self.fkd0 = fkd0 
        self.ffd0 = ffd0 
        self.wr0 = wr0 
        self.th0  = th0  
        
        dmax = 1e8

        # call super:

        Device.__init__(self, name)

        # derived:

        self.Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)

        self.Ld = (Lls + (Lmd * Llfd * Llkd)
                   / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd))

        # port atoms:

        self.ids = StateAtom("ids", x0=ids0, derfunc=self.dids, units="A", dq=dq_current, dmax=dmax)
        self.iqs = StateAtom("iqs", x0=iqs0, derfunc=self.diqs, units="A", dq=dq_current, dmax=dmax)
       
        # state atoms:

        self.fkq = StateAtom("fkq", x0=fkq0, derfunc=self.dfkq, units="Wb", dq=dq_flux, dmax=dmax)
        self.fkd = StateAtom("fkd", x0=fkd0, derfunc=self.dfkd, units="Wb", dq=dq_flux, dmax=dmax)
        self.ffd = StateAtom("ffd", x0=ffd0, derfunc=self.dffd, units="Wb", dq=dq_flux, dmax=dmax)
        self.wr = StateAtom("wr", x0=wr0, derfunc=self.dwr, units="rad/s", dq=dq_wr, dmax=dmax)
        self.th = StateAtom("th", x0=th0, derfunc=self.dth, units="rad", dq=dq_th, dmax=dmax)

        self.add_atoms(self.ids, self.iqs, self.fkq, self.fkd, self.ffd,
                       self.wr, self.th)

        # iqs relies on ext_q, wr, ids, fkq
        # ids relies on ext_d, wr, iqs, fkd  
        # fkq relies on iqs
        # fkd relies on ffd, ids
        # ffd relies on fkd, ids
        # wr relies on fkq, fkd, ffd, wr, iqs, ids
        # th relies on wr

        self.iqs.add_connection(self.wr)
        self.iqs.add_connection(self.ids)
        self.iqs.add_connection(self.fkq)

        self.ids.add_connection(self.wr)
        self.ids.add_connection(self.iqs)
        self.ids.add_connection(self.fkd)

        self.fkq.add_connection(self.iqs)

        self.fkd.add_connection(self.ffd)
        self.fkd.add_connection(self.ids)

        self.ffd.add_connection(self.fkd)
        self.ffd.add_connection(self.ids)

        self.wr.add_connection(self.ids)
        self.wr.add_connection(self.iqs)
        self.wr.add_connection(self.fkq)
        self.wr.add_connection(self.fkd)
        self.wr.add_connection(self.ffd)
        self.wr.add_connection(self.th)

        self.th.add_connection(self.wr)

        # for terminal connection:

        self.otherd = None
        self.otherq = None

        self.atomd = self.ids
        self.atomq = self.iqs

    def connect(self, device):

        self.otherd = self.atomd.add_connection(device.atomd, 1.0)
        self.otherq = self.atomq.add_connection(device.atomq, -1.0)

    def vds(self):
        return self.otherd.value()
        
    def vqs(self):
        return self.otherq.value()

    def vfd(self):
        return self.vfdb

    def fq(self, fkq):
        return self.Lmq / (self.Lmq + self.Llkq) * fkq

    def fd(self, fkd, ffd):
        return (self.Lmd * (fkd / self.Llkd + ffd / self.Llfd) /
                (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))

    def fqs(self, fkq, iqs):
        return self.Lq * iqs + self.fq(fkq)

    def fds(self, fkd, ffd, ids):
        return self.Ld * ids + self.fd(fkd, ffd)

    def Te(self, fkq, fkd, ffd, iqs, ids):
        return (3.0 * self.P / 4.0 * (self.fds(fkd, ffd, ids) * iqs 
                - self.fqs(fkq, iqs) * ids))

    def Tm(self, wr, th):
        return -(self.Kp * (self.ws - wr) + self.Ki * th)

    @staticmethod
    def diqs(self, iqs):  # iqs relies on ext_q, wr, ids, fkq

        # diqs = (1/Lls) * (vqs - Rs*iqs - wr*Ld*ids - wr*fd)

        return (1.0 / self.Lls) * (self.vqs()
                                   - self.rs * iqs
                                   - self.wr.q * self.Ld * self.ids.q
                                   - self.wr.q * self.fd(self.fkd.q, self.ffd.q))
    @staticmethod
    def dids(self, ids):  # ids relies on ext_d, wr, iqs, fkd

       # dids = (1/Lls) * (vds - Rs*ids + wr*Lq*iqs + wr*fq) 

        return (1.0 / self.Lls) * (self.vds()
                                   - self.rs * ids
                                   + self.wr.q * self.Lq * self.iqs.q
                                   + self.wr.q * self.fq(self.fkq.q))
    
    @staticmethod      
    def dfkq(self, fkq):  # fkq relies on iqs
        return (-self.rkq / self.Llkq * (fkq - self.Lq * self.iqs.q -
                self.fq(fkq) + self.Lls * self.iqs.q))

    @staticmethod
    def dfkd(self, fkd):  # fkd relies on ffd, ids
        return (-self.rkd / self.Llkd * (fkd - self.Ld * self.ids.q + 
                self.fd(fkd, self.ffd.q) - self.Lls * self.ids.q))

    @staticmethod
    def dffd(self, ffd):  # ffd relies on fkd, ids
        return (self.vfd() - self.rfd / self.Llfd * (ffd - self.Ld *
                self.ids.q + self.fd(self.fkd.q, ffd) - self.Lls *
                self.ids.q))

    @staticmethod
    def dwr(self, wr):  # wr relies on fkq, fkd, ffd, wr, iqs, ids
        return ((1.0 / self.J) 
                * (self.Te(self.fkq.q, self.fkd.q, self.ffd.q, self.iqs.q, self.ids.q)
                - self.Tm(wr, self.th.q)))

    @staticmethod
    def dth(self, th): # th relies on wr
        return self.ws - self.wr.q


class SyncMachineDQ3(Device):

    """7th order Synchronous Machine Model (5 fluxes, speed and angle states)

    vd = id * Rs - wr * fq + d/dt * fd
    vq = iq * Rs + wr * fd + d/dt * fq
    vfd = d/dt * ffd + Rfd * ifd
    0 = d/dt * fkd + Rkd * ikd
    0 = d/dt * fkq + Rkq * ikq

    [ fd  ]   [ Lmd+Lls  Lmd       Lmd      ]   [ id  ]
    [ fkd ] = [ Lmd      Llkd+Lmd  Lmd      ] * [ ikd ]
    [ ffd ]   [ Lmd      Lmd       Llfd+Lmd ]   [ ifd ]

    [ fq  ] = [ Lmq+Lls  Lmq     ] * [ iq  ]
    [ fkq ]   [ Lmq      Lmq+Lkq ]   [ ikq ]

    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, ws=60.0*pi,
                 P=4.00, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fq0=0.0, fkq0=0.0, fd0=0.0, fkd0=0.0,
                 ffd0=0.0, wr0=60.0*pi, th0=0.0, iqs0=0.0, ids0=0.0,
                 dq_current=1e-1, dq_flux=1e-2, dq_wr=1e-1, dq_th=1e-3, dq_v=1e0):

        self.name = name

        # sm params:

        self.Psm  = Psm  
        self.VLL  = VLL  
        self.ws   = ws  
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

        self.fq0  = fq0
        self.fkq0 = fkq0
        self.fd0  = fd0
        self.fkd0 = fkd0
        self.ffd0 = ffd0
        self.wr0 = wr0 
        self.th0  = th0  
        
        dmax = 1e8

        # call super:

        Device.__init__(self, name)

        # derived:

        dend = Llkd*Lls*Lmd+Llfd*Lls*Lmd+Llfd*Llkd*Lmd+Llfd*Llkd*Lls
        denq = (Lls*Lmq+Llkq*Lmq+Llkq*Lls)

        self.sdd = (Llkd*Lmd+Llfd*Lmd+Llfd*Llkd)/dend
        self.sdk = -Llfd*Lmd/dend
        self.sdf = -Llkd*Lmd/dend
        
        self.skd = -Llfd*Lmd/dend
        self.skkd = (Lls*Lmd+Llfd*Lmd+Llfd*Lls)/dend
        self.skf = -Lls*Lmd/dend
        
        self.sfd = -Llkd*Lmd/dend
        self.sfk = -Lls*Lmd/dend
        self.sff = (Lls*Lmd+Llkd*Lmd+Llkd*Lls)/dend
        
        self.sqq = (Lmq+Llkq)/denq
        self.sqk = -Lmq/denq

        self.skq = -Lmq/denq
        self.skkq = (Lmq+Lls)/denq
        
        self.Lq = Lls+(Lmq*Llkq)/(Llkq+Lmq)
        self.Ld = (Lls+(Lmd*Llfd*Llkd)/(Lmd*Llfd+Lmd*Llkd+Llfd*Llkd))

        # port atoms:

        self.ids = SourceAtom("ids", x0=ids0, derfunc=self.id, units="A", dq=dq_current, dmax=dmax)
        self.iqs = SourceAtom("iqs", x0=iqs0, derfunc=self.iq, units="A", dq=dq_current, dmax=dmax)
       
        # state atoms:

        self.fq = StateAtom("fq", x0=fq0, derfunc=self.dfq, units="Wb", dq=dq_flux, dmax=dmax)
        self.fd = StateAtom("fd", x0=fd0, derfunc=self.dfd, units="Wb", dq=dq_flux, dmax=dmax)
        self.fkq = StateAtom("fkq", x0=fkq0, derfunc=self.dfkq, units="Wb", dq=dq_flux, dmax=dmax)
        self.fkd = StateAtom("fkd", x0=fkd0, derfunc=self.dfkd, units="Wb", dq=dq_flux, dmax=dmax)
        self.ffd = StateAtom("ffd", x0=ffd0, derfunc=self.dffd, units="Wb", dq=dq_flux, dmax=dmax)
        self.wr = StateAtom("wr", x0=wr0, derfunc=self.dwr, units="rad/s", dq=dq_wr, dmax=dmax)
        self.th = StateAtom("th", x0=th0, derfunc=self.dth, units="rad", dq=dq_th, dmax=dmax)

        self.add_atoms(self.ids, self.iqs, self.fkq, self.fkd, self.ffd,
                       self.wr, self.th)

        # iqs relies on ext_q, wr, ids, fkq
        # ids relies on ext_d, wr, iqs, fkd  
        # fkq relies on iqs
        # fkd relies on ffd, ids
        # ffd relies on fkd, ids
        # wr relies on fkq, fkd, ffd, wr, iqs, ids
        # th relies on wr

        self.iqs.add_connection(self.wr)
        self.iqs.add_connection(self.ids)
        self.iqs.add_connection(self.fkq)

        self.ids.add_connection(self.wr)
        self.ids.add_connection(self.iqs)
        self.ids.add_connection(self.fkd)

        self.fkq.add_connection(self.iqs)

        self.fkd.add_connection(self.ffd)
        self.fkd.add_connection(self.ids)

        self.ffd.add_connection(self.fkd)
        self.ffd.add_connection(self.ids)

        self.wr.add_connection(self.ids)
        self.wr.add_connection(self.iqs)
        self.wr.add_connection(self.fkq)
        self.wr.add_connection(self.fkd)
        self.wr.add_connection(self.ffd)
        self.wr.add_connection(self.th)

        self.th.add_connection(self.wr)

        # for terminal connection:

        self.otherd = None
        self.otherq = None

        self.atomd = self.ids
        self.atomq = self.iqs

    def connect(self, device):

        self.otherd = self.atomd.add_connection(device.atomd, 1.0)
        self.otherq = self.atomq.add_connection(device.atomq, -1.0)

    def vds(self):
        return self.otherd.value()
        
    def vqs(self):
        return self.otherq.value()

    def vfd(self):
        return self.vfdb

    def iq(self, fq, fkq):
        return self.sqq * fq + self.sqk * fkq

    def id(self, fd, fkd, ffd):
        return self.sdd * fd + self.sdk * fkd + self.sdf * ffd

    def ikd(self, fd, fkd, ffd):
        return self.skd * fd + self.skkd * fkd + self.skf * ffd

    def ifd(self, fd, fkd, ffd):
        return self.sfd * fd + self.sfk * fkd + self.sff * ffd

    def ikq(self, fq, fkq):
        return self.skq * fq + self.skkq * fkq

    def fq(self, fkq):
        return self.Lmq / (self.Lmq + self.Llkq) * fkq

    def fd(self, fkd, ffd):
        return (self.Lmd * (fkd / self.Llkd + ffd / self.Llfd) /
                (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))

    def Te(self, fq, fd, fkq, fkd, ffd):
        return (3.0 * self.P / 4.0
                * ((self.Ld * self.id(fd, fkd, ffd)
                + self.fd(fkd, ffd)) * self.iq(fq, fkq)
                - (self.Lq * self.iq(fq, fkq)
                + self.fq(fkq)) * self.id(fd, fkd, ffd)))

    def Tm(self, wr, th):
        return -(self.Kp * (self.ws - wr) + self.Ki * th)

    @staticmethod
    def dfd(self, fd):    # fd relies on fkd, ffd, fq
        return (self.vds() - self.id(fd, self.fkd.q, self.ffd.q) * self.rs
                + self.wr.q * self.fq.q)
    
    @staticmethod
    def dfq(self, fq):    # fq relies on fkq, fd
        return (self.vqs() - self.iq(fq, self.fkq.q) * self.rs
                - self.wr.q * self.fd.q)

    @staticmethod
    def dffd(self, ffd):  # fkq relies on fq
        return self.vfd() - self.ifd(self.fd.q, self.fkd.q, ffd) * self.rfd

    @staticmethod
    def dfkd(self, fkd):  # fkd relies on fd, ffd 
        return -self.ikd(self.fd.q, fkd, self.ffd.q) * self.rkd

    @staticmethod
    def dfkq(self, fkq):  # fkq relies on fq
        return -self.ikq(self.fq.q, fkq) * self.rkq

    @staticmethod
    def dwr(self, wr):    # wr relies on fq, fq, fkq, fkd, ffd
        return ((1.0 / self.J)
                * (self.Te(self.fq.q, self.fd.q, self.fkq.q, self.fkd.q, self.ffd.q)
                   - self.Tm(wr, self.th.q)))

    @staticmethod
    def dth(self, th): # th relies on wr
        return self.ws - self.wr.q


class SyncMachineDQ4(Device):

    """Synchronous Machine Reduced DQ Model as Voltage source behind a
    Lim Latency Branch.

    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, ws=60.0*pi,
                 P=4.00, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fkq0=0.0, fkd0=0.0, ffd0=0.0, wr0=60.0*pi,
                 th0=0.0, iqs0=0.0, ids0=0.0, dq_i=1e-2, dq_f=1e-2, dq_wr=1e-1,
                 dq_th=1e-3, dq_v=1e0):

        self.name = name

        # sm params:

        self.Psm  = Psm  
        self.VLL  = VLL  
        self.ws   = ws  
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

        self.iqs0 = iqs0 
        self.ids0 = ids0 
        self.fkq0 = fkq0 
        self.fkd0 = fkd0 
        self.ffd0 = ffd0 
        self.wr0 = wr0 
        self.th0  = th0  
        
        dmax = 1e8

        # call super:

        Device.__init__(self, name)

        # derived:

        self.Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)
        self.Ld = (Lls + (Lmd * Llfd * Llkd) / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd))

        # state atoms:

        self.ids = StateAtom("ids", x0=ids0, derfunc=self.dids, units="A",     dq=dq_i)
        self.iqs = StateAtom("iqs", x0=iqs0, derfunc=self.diqs, units="A",     dq=dq_i)
        self.fkq = StateAtom("fkq", x0=fkq0, derfunc=self.dfkq, units="Wb",    dq=dq_f)
        self.fkd = StateAtom("fkd", x0=fkd0, derfunc=self.dfkd, units="Wb",    dq=dq_f)
        self.ffd = StateAtom("ffd", x0=ffd0, derfunc=self.dffd, units="Wb",    dq=dq_f)
        self.wr  = StateAtom("wr",  x0=wr0,  derfunc=self.dwr,  units="rad/s", dq=dq_wr)
        self.th  = StateAtom("th",  x0=th0,  derfunc=self.dth,  units="rad",   dq=dq_th)

        self.add_atoms(self.ids, self.iqs, self.fkq, self.fkd, self.ffd, self.wr, self.th)

        # atom connections:

        self.ids.add_connection(self.iqs)
        self.ids.add_connection(self.wr)
        self.ids.add_connection(self.fkq)
        self.ids.add_connection(self.fkq)
        self.iqs.add_connection(self.ids)
        self.iqs.add_connection(self.wr)
        self.iqs.add_connection(self.fkd)
        self.iqs.add_connection(self.ffd)
        self.fkd.add_connection(self.ffd)
        self.fkd.add_connection(self.ids)
        self.fkq.add_connection(self.iqs)
        self.ffd.add_connection(self.ids)
        self.ffd.add_connection(self.fkd)
        self.wr.add_connection(self.ids)
        self.wr.add_connection(self.iqs)
        self.wr.add_connection(self.fkq)
        self.wr.add_connection(self.fkd)
        self.wr.add_connection(self.ffd)
        self.wr.add_connection(self.th)
        self.th.add_connection(self.wr)

        # jacobian:

        self.ids.add_jacfunc(self.ids, self.jids_ids) 
        self.ids.add_jacfunc(self.iqs, self.jids_iqs) 
        self.ids.add_jacfunc(self.fkq, self.jids_fkq)  
        self.ids.add_jacfunc(self.wr , self.jids_wr ) 
        self.iqs.add_jacfunc(self.ids, self.jiqs_ids) 
        self.iqs.add_jacfunc(self.iqs, self.jiqs_iqs)  
        self.iqs.add_jacfunc(self.fkd, self.jiqs_fkd) 
        self.iqs.add_jacfunc(self.ffd, self.jiqs_ffd) 
        self.iqs.add_jacfunc(self.wr , self.jiqs_wr )
        self.fkq.add_jacfunc(self.iqs, self.jfkq_iqs) 
        self.fkq.add_jacfunc(self.fkq, self.jfkq_fkq) 
        self.fkd.add_jacfunc(self.ids, self.jfkd_ids)
        self.fkd.add_jacfunc(self.fkd, self.jfkd_fkd) 
        self.fkd.add_jacfunc(self.ffd, self.jfkd_ffd) 
        self.ffd.add_jacfunc(self.ids, self.jffd_ids) 
        self.ffd.add_jacfunc(self.fkd, self.jffd_fkd) 
        self.ffd.add_jacfunc(self.ffd, self.jffd_ffd) 
        self.wr.add_jacfunc(self.wr  ,  self.jwr_wr ) 
        self.wr.add_jacfunc(self.th  ,  self.jwr_th )
        self.th.add_jacfunc(self.wr  ,  self.jth_wr ) 

        # ports:

        self.atomd = self.iqs
        self.atomq = self.ids

        self.termd = None  # terminal voltage d connection
        self.termq = None  # terminal voltage q connection

        self.input = None  # avr

    def connect(self, bus, avr=None):

        self.termd = self.iqs.add_connection(bus.atomq)
        self.termq = self.ids.add_connection(bus.atomd)

        if avr:
            self.input = self.ffd.add_connection(avr.vfd, self.vfdb)
            avr.vd = avr.x1.add_connection(bus.vd, 1.0 / self.VLL)
            avr.vq = avr.x1.add_connection(bus.vq, 1.0 / self.VLL)

    def vtermd(self):
        return self.termd.value()
        
    def vtermq(self):
        return self.termq.value()

    def vfd(self):
        if self.input:
            return self.input.value()
        else:
           return self.vfdb # no feedback control

    @staticmethod
    def jids_ids(self, ids):
        return -self.rs/self.Lls
    
    @staticmethod
    def jids_iqs(self, ids, iqs):
        return -(self.Lq * self.wr.q) / self.Lls
    
    @staticmethod
    def jids_fkq(self, ids, fkq):
        return (self.Lmq * self.wr.q) / (self.Lls * (self.Lmq + self.Llkq))
    
    @staticmethod
    def jids_wr(self, ids, wr):
        return (((self.Lmq * self.fkq.q) / (self.Lmq + self.Llkq)
                - self.Lq * self.iqs.q) / self.Lls)
    
    @staticmethod
    def jiqs_ids(self, iqs, ids):
        return (self.Ld * self.wr.q) / self.Lls
    
    @staticmethod
    def jiqs_iqs(self, iqs):
        return -self.rs / self.Lls
    
    @staticmethod
    def jiqs_fkd(self, iqs, fkd):
        return ((self.Lmd * self.wr.q) / (self.Llkd * self.Lls
                * (self.Lmd / self.Llkd + self.Lmd / self.Llfd + 1)))
    
    @staticmethod
    def jiqs_ffd(self, iqs, ffd):
        return ((self.Lmd * self.wr.q) / (self.Llfd * self.Lls
                * (self.Lmd / self.Llkd + self.Lmd / self.Llfd + 1)))
    
    @staticmethod
    def jiqs_wr(self, iqs, wr):
        return ((self.Ld * self.ids.q + (self.Lmd * (self.fkd.q / self.Llkd
                + self.ffd.q / self.Llfd)) / (self.Lmd / self.Llkd
                + self.Lmd / self.Llfd + 1)) / self.Lls)

    @staticmethod
    def jfkq_iqs(self, fkq, iqs):
        return -((self.Lls - self.Lq) * self.rkq) / self.Llkq
    
    @staticmethod
    def jfkq_fkq(self, fkq):
        return -((1 - self.Lmq / (self.Lmq + self.Llkq)) * self.rkq) / self.Llkq
 
    @staticmethod
    def jfkd_ids(self, fkd, ids):
        return -((-self.Lls - self.Ld) * self.rkd) / self.Llkd

    @staticmethod
    def jfkd_fkd(self, fkd):
        return (-((self.Lmd / (self.Llkd * (self.Lmd / self.Llkd + self.Lmd
                / self.Llfd + 1)) + 1) * self.rkd) / self.Llkd)
    
    @staticmethod
    def jfkd_ffd(self, fkd, ffd):
        return -(self.Lmd * self.rkd) / (self.Llfd * self.Llkd
               * (self.Lmd / self.Llkd + self.Lmd / self.Llfd + 1))

    @staticmethod
    def jffd_ids(self, ffd, ids):
        return -((-self.Lls - self.Ld) * self.rfd) / self.Llfd
    
    @staticmethod
    def jffd_fkd(self, ffd, fkd):
        return (-(self.Lmd * self.rfd) / (self.Llfd * self.Llkd
               * (self.Lmd / self.Llkd + self.Lmd/self.Llfd + 1)))
    
    @staticmethod
    def jffd_ffd(self, ffd):
        return (-((self.Lmd / (self.Llfd * (self.Lmd / self.Llkd + self.Lmd
                / self.Llfd + 1)) + 1) * self.rfd) / self.Llfd)

    @staticmethod
    def jwr_ffd(self, ffd):
        return (-((self.Lmd / (self.Llfd * (self.Lmd / self.Llkd + self.Lmd
                / self.Llfd + 1)) + 1) * self.rfd) / self.Llfd)

    @staticmethod
    def jwr_wr(self, wr):
        return -self.Kp / self.J

    @staticmethod
    def jwr_th(self, wr, th):
        return self.Ki / self.J

    @staticmethod
    def jth_wr(self, th, wr):
        return -1.0

    @staticmethod
    def dids(self, ids):
        return ((-self.wr.q * self.Lq * self.iqs.q + self.wr.q
                * (self.Lmq / (self.Lmq + self.Llkq) * self.fkq.q)
                - self.vtermd() - self.rs * ids) / self.Lls)

    @staticmethod
    def diqs(self, iqs):
        return ((self.wr.q * self.Ld * self.ids.q + self.wr.q
                * (self.Lmd * (self.fkd.q / self.Llkd + self.ffd.q / self.Llfd)
                /  (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))
                - self.vtermq() - self.rs * iqs) / self.Lls)
                             
    @staticmethod      
    def dfkq(self, fkq):
        return (-self.rkq / self.Llkq * (fkq - self.Lq * self.iqs.q
                - (self.Lmq / (self.Lmq + self.Llkq) * fkq) + self.Lls * self.iqs.q))

    @staticmethod
    def dfkd(self, fkd):
        return (-self.rkd / self.Llkd * (fkd - self.Ld * self.ids.q
                + (self.Lmd * (fkd / self.Llkd + self.ffd.q / self.Llfd)
                / (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))
                - self.Lls * self.ids.q))

    @staticmethod
    def dffd(self, ffd):
        return (self.vfd() - self.rfd / self.Llfd
                * (ffd - self.Ld * self.ids.q
                + (self.Lmd * (self.fkd.q / self.Llkd + ffd / self.Llfd)
                /  (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))
                - self.Lls * self.ids.q))

    @staticmethod
    def dwr(self, wr):
        return ((3.0 * self.P / 4.0 * ((self.Ld * self.ids.q
                + ((self.Lmd * (self.fkd.q / self.Llkd + self.ffd.q / self.Llfd)
                /  (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd)))) * self.iqs.q
                - (self.Lq * self.iqs.q + (self.Lmq / (self.Lmq + self.Llkq) * self.fkq.q)) * self.ids.q))
                + (self.Kp * (self.ws - wr) + self.Ki * self.th.q)) / self.J

    @staticmethod
    def dth(self, th):
        return (self.ws - self.wr.q)


class AC8B(Device):

    """IEEE AC8B Exciter
                    
                .--------.                                               
    vref     .->|  Kpr   |----.                Vrmax                    
      |      |  '--------'    |                ,---               (vfd)   
    + v      |              + v       .-------'-.               .------.   
     ,-.  e1 |  .--------. + ,-.  pid |   Ka    |    + ,-.  e2  |  1   |   
    ( Σ )----+->| Kir/s  |->( Σ )---->| ------- |---->( Σ )---->| ---- |---+--> vfd
     `-'     |  '--------'   `-'      | 1+s*Ta  |  vr  `-'      | s*Te |   |
    - ^      |  .--------.  + ^       '-,-------'     - ^       '------'   |
      |      |  | s*Kdr  |    |     ---'  (x3)          | vse              |
     vt      '->| ------ |----'    Vrmin               ,'.    .--------.   | 
                | 1+s*Tr |                            ( Σ )<--| Se(Ve) |<--+
                '--------'                             `-' +  '--------'   |
                                                      + ^      .----.      |
                 (x1, x2)                               '------| Ke |<-----'
                                                               '----'     
    """

    def __init__(self, name, vref=1.0, Kpr=200.0, Kir=0.8,
                 Kdr=1e-3, Tdr=1e-3, Ka=1.0, Ta=1e-4, Vrmin=0.0, Vrmax=5.0,
                 Te=1.0, Ke=1.0, Sea=1.0119, Seb=0.0875, x10=0.0, x20=0.0,
                 x30=0.0, vfd0=None, dq=1e-3):

        Device.__init__(self, name)

        self.vref = vref
        self.Kpr = Kpr
        self.Kir = Kir
        self.Kdr = Kdr
        self.Tdr = Tdr
        self.Ka = Ka
        self.Ta = Ta
        self.Vrmin = Vrmin
        self.Vrmax = Vrmax
        self.Te = Te
        self.Ke = Ke
        self.Sea = Sea
        self.Seb = Seb

        self.x10 = x10
        self.x20 = x20
        self.x30 = x30

        if not vfd0:
            vfd0 = self.vref

        self.vfd0 = vfd0

        self.dq = dq

        self.x1 = StateAtom("x1", x0=x10, derfunc=self.dx1, dq=dq*0.01)
        self.x2 = StateAtom("x2", x0=x20, derfunc=self.dx2, dq=dq*0.0001)
        self.x3 = StateAtom("x3", x0=x30, derfunc=self.dx3, dq=dq*0.1)
        self.vfd = StateAtom("vfd", x0=vfd0, derfunc=self.dvfd, dq=dq*10.0)

        self.add_atoms(self.x1, self.x2, self.x3, self.vfd)

        self.x2.add_connection(self.x1)
        self.x3.add_connection(self.x1)
        self.x3.add_connection(self.x2)
        self.vfd.add_connection(self.x3)

        self.x1.add_jacfunc(self.x1,   self.jx1_x1  )
        self.x2.add_jacfunc(self.x1,   self.jx2_x1  )
        self.x3.add_jacfunc(self.x1,   self.jx3_x1  )
        self.x3.add_jacfunc(self.x2,   self.jx3_x2  )
        self.vfd.add_jacfunc(self.x3,  self.jvfd_x3 )
        self.vfd.add_jacfunc(self.vfd, self.jvfd_vfd)

        self.vd = Connection()
        self.vq = Connection()

    @staticmethod
    def jx1_x1(self, x1):
        return -1 / self.Tdr

    @staticmethod
    def jx2_x1(self, x2, x1):
        return 1

    @staticmethod
    def jx3_x1(self, x3, x1):
        return self.Kir * self.Tdr - self.Kdr / self.Tdr

    @staticmethod
    def jx3_x2(self, x3, x2):
        return self.Kir

    @staticmethod
    def jvfd_x3(self, vfd, x3):
        return self.Ka / (self.Ta * self.Te)

    @staticmethod
    def jvfd_vfd(self, vfd):
        return -self.Ke / self.Te

    @staticmethod
    def dx1(self, x1):  # x1 depends on: vt
        return (-1.0 / self.Tdr * x1 + (self.vref
               - sqrt(self.vd.value()**2 + self.vq.value()**2)))

    @staticmethod
    def dx2(self, x2):  # x2 depends on: x1
        return self.x1.q

    @staticmethod
    def dx3(self, x3):  # x3 depends on: x1, x2
        return -1.0 / self.Ta + ((self.Kir * self.Tdr - self.Kdr / self.Tdr)
                * self.x1.q + self.Kir * self.x2.q + (self.Kdr + self.Kpr
                * self.Tdr) * (self.vref - sqrt(self.vd.value()**2
                + self.vq.value()**2)))

    @staticmethod
    def dvfd(self, vfd):  # ve depends on: vt
        return (self.Ka / self.Ta * self.x3.q - vfd * self.Ke) / self.Te


class InductionMachineDQ(Device):

    """

    fqs = Lls * iqs + Lm * (iqs + iqr)
    fds = Lls * ids + Lm * (ids + idr)
    fqr = Llr * iqr + Lm * (iqr + iqs)
    fdr = Llr * idr + Lm * (idr + ids)

    vqs = Rs * iqs + wr * fds + (Lls + Lm) * diqs + Lm * diqr
    vds = Rs * ids - wr * fqs + (Lls + Lm) * dids + Lm * didr
    vqr = Rr * iqr + (ws - wr) * fdr + (Llr + Lm) * diqr + Lm * diqs
    vdr = Rr * idr - (ws - wr) * fqr + (Llr + Lm) * didr + Lm * dids

    """

    def __init__(self, name, ws=30*pi, P=4, Tb=26.53e3, Rs=31.8e-3,
                 Lls=0.653e-3, Lm=38e-3, Rr=24.1e-3, Llr=0.658e-3, J=250.0,
                 iqs0=0.0, ids0=0.0, iqr0=0.0, idr0=0.0, wr0=0.0, dq_i=1e-2,
                 dq_wr=1e-1):

        Device.__init__(self, name)

        self.name = name
        self.ws = ws
        self.P = P
        self.Tb = Tb
        self.Rs = Rs
        self.Lls = Lls
        self.Lm = Lm
        self.Rr = Rr
        self.Llr = Llr
        self.J = J

        self.iqs0 = iqs0
        self.ids0 = ids0
        self.iqr0 = iqr0
        self.idr0 = idr0
        self.wr0 = wr0
        
        self.dq_i = dq_i
        self.dq_wr = dq_wr

        # derived:

        self.dinv = 1.0 / (Lls * Lm + Llr * Lm + Llr * Lls)

        # atoms:

        self.iqs = StateAtom("iqs", x0=iqs0, derfunc=self.diqs, units="A",     dq=dq_i)
        self.ids = StateAtom("ids", x0=ids0, derfunc=self.dids, units="A",     dq=dq_i)
        self.iqr = StateAtom("iqr", x0=iqr0, derfunc=self.diqr, units="A",     dq=dq_i)
        self.idr = StateAtom("idr", x0=idr0, derfunc=self.didr, units="A",     dq=dq_i)
        self.wr  = StateAtom("wr",  x0=wr0,  derfunc=self.dwr,  units="rad/s", dq=dq_wr)
        
        self.add_atoms(self.iqs, self.ids, self.iqr, self.idr, self.wr)

        # atom connections:

        self.iqs.add_connection(self.ids)
        self.iqs.add_connection(self.iqr)
        self.iqs.add_connection(self.idr)
        self.iqs.add_connection(self.wr)
        self.ids.add_connection(self.iqs)
        self.ids.add_connection(self.iqr)
        self.ids.add_connection(self.idr)
        self.ids.add_connection(self.wr)
        self.iqr.add_connection(self.iqs)
        self.iqr.add_connection(self.ids)
        self.iqr.add_connection(self.idr)
        self.iqr.add_connection(self.wr)
        self.idr.add_connection(self.iqs)
        self.idr.add_connection(self.ids)
        self.idr.add_connection(self.iqr)
        self.idr.add_connection(self.wr)
        self.wr.add_connection(self.iqs)
        self.wr.add_connection(self.ids)
        self.wr.add_connection(self.iqr)
        self.wr.add_connection(self.idr)

        # ports:

        self.atomd = self.iqs
        self.atomq = self.ids

        self.termd = None  # terminal voltage d connection
        self.termq = None  # terminal voltage q connection

    def connect(self, bus):

        self.termd = self.iqs.add_connection(bus.atomq)
        self.termq = self.ids.add_connection(bus.atomd)

    def vds(self):
        return self.termd.value()
        
    def vqs(self):
        return self.termq.value()

    def vdr(self):
        return 0.0
        
    def vqr(self):
        return 0.0

    def Te(self, iqs, ids, iqr, idr):

        fqs = self.Lls * iqs + self.Lm * (iqs + iqr)
        fds = self.Lls * ids + self.Lm * (ids + idr)
        return (3*self.P/4) * (fds * iqs - fqs * ids)

    def Tm(self, wr):
        return self.Tb * (wr / self.ws)**3

    @staticmethod
    def diqs(self, iqs):
        return ((self.Lm**2*self.ids.q+(self.Lm**2+self.Llr*self.Lm)*self.idr.q)*self.ws
	        +((-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.ids.q
            +(-2*self.Lm**2-2*self.Llr*self.Lm)*self.idr.q)*self.wr.q+(self.Lm+self.Llr)*self.vqs()-self.Lm*self.vqr()
            +(-self.Lm-self.Llr)*self.Rs*self.iqs.q+self.Lm*self.Rr*self.iqr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def dids(self, ids):
        return -((self.Lm**2*self.iqs.q+(self.Lm**2+self.Llr*self.Lm)*self.iqr.q)*self.ws
	        +((-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.iqs.q
		    +(-2*self.Lm**2-2*self.Llr*self.Lm)*self.iqr.q)*self.wr.q+(-self.Lm-self.Llr)*self.vds()+self.Lm*self.vdr()
		    +(self.Lm+self.Llr)*self.Rs*self.ids.q-self.Lm*self.Rr*self.idr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def diqr(self, iqr):
        return -(((self.Lm**2+self.Lls*self.Lm)*self.ids.q+(self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)*self.idr.q)*self.ws
	        +((-2*self.Lm**2-2*self.Lls*self.Lm)*self.ids.q+(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.idr.q)*self.wr.q
	        +self.Lm*self.vqs()+(-self.Lm-self.Lls)*self.vqr()-self.Lm*self.Rs*self.iqs.q+(self.Lm+self.Lls)*self.Rr*self.iqr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def didr(self, idr):
        return (((self.Lm**2+self.Lls*self.Lm)*self.iqs.q+(self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)*self.iqr.q)*self.ws
	        +((-2*self.Lm**2-2*self.Lls*self.Lm)*self.iqs.q+(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.iqr.q)*self.wr.q
	        -self.Lm*self.vds()+(self.Lm+self.Lls)*self.vdr()+self.Lm*self.Rs*self.ids.q+(-self.Lm-self.Lls)*self.Rr*self.idr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def dwr(self, wr):
        return (self.P / (2.0 * self.J)) * (
            self.Te(self.iqs.q, self.ids.q, self.iqr.q, self.idr.q) - self.Tm(wr)) 


class TRLoadDQ(Device):

    """Transformer rectifier load average model.


              .-----.-----.------VVVV---UUUU----.------.
             _|_   _|_   _|_     Rdc    Ldc     |      |
             /_\   /_\   /_\                    |      |
       Lc     |     |     |       ----->        |      |          +
    o--UUUU---o     |     |        idc          |      |
       Lc     |     |     |                    _|_    <.
    o--UUUU---------o     |                 C  ___    <.  R     vdc
              |     |     |                     |     <.
    o--UUUU---------------o                     |      |
       Lc    _|_   _|_   _|_                    |      |
             /_\   /_\   /_\                    |      |          -
              |     |     |                     |      |
              '-----'-----'---------------------'------'

              Average DQ model:

                     .----------------------------------------------------.
                     |            TRLoadDQ       Rdc  Ldc                 |
    .----. vtermd    |      ,^.               .--VVV--UUU--.----.         |
    | DQ |---o-------|-----<   >--.           |   idc -->  |    |    +    |
    |Node|    id ->  |  igd `.'   |  .     + ,^.          _|_  <.         |
    |    |           |            +--||-    <   >     C   ___  <. R  vdc  |
    |    |    iq ->  |  igq ,^.   |  '    vg `.'           |   <.         |
    |    |---o-------|-----<   >--'           |            |    |    -    |
    '----' vtermq    |      `.'               '------------'----'         |
                     '----------------------------------------------------'

    """

    def __init__(self, name, w=2*pi*60.0, alpha_cmd=0.0, Lc=76.53e-6, Rdc=0.0,
                 Ldc=0.383e-3, R=3.16, C=0.384e-3, Nps=1.0, idc0=0.0, vdc0=0.0,
                 dq_i=1e0, dq_v=1e0):

        self.name = name

        # params:

        self.w = w
        self.alpha_cmd = alpha_cmd
        self.Lc = Lc 
        self.Rdc = Rdc
        self.Ldc = Ldc
        self.R = R
        self.C = C
        self.Nps = Nps

        # intial conditions:

        self.idc0 = idc0
        self.vdc0 = vdc0

        # delta q:
                
        self.dq_i = dq_i
        self.dq_v = dq_v

        # call super:

        Device.__init__(self, name)

        # derived:

        self.ke = 1.0 / (sqrt(2.0) * self.Nps)

        # state atoms:

        self.id = SourceAtom("id", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_id, units="A", dq=dq_i)

        self.iq = SourceAtom("iq", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_iq, units="A", dq=dq_i)

        self.idc = StateAtom("idc", x0=idc0, derfunc=self.didc, units="A", dq=dq_i)
        self.vdc = StateAtom("vdc", x0=vdc0, derfunc=self.dvdc, units="V", dq=dq_v)

        self.add_atoms(self.id, self.iq, self.idc, self.vdc)

        # atom connections:

        self.idc.add_connection(self.vdc)
        self.vdc.add_connection(self.idc)

        self.id.add_connection(self.idc)
        self.iq.add_connection(self.idc)
        
        # ports:

        self.atomd = self.id
        self.atomq = self.iq

        self.vdg = None  # terminal voltage d atom
        self.vqg = None  # terminal voltage q atom

    def connect(self, bus):

        self.vdg = bus.atomd
        self.vqg = bus.atomq

        self.id.add_connection(bus.atomd)
        self.id.add_connection(bus.atomq)

        self.iq.add_connection(bus.atomd)
        self.iq.add_connection(bus.atomq)

        self.idc.add_connection(bus.atomd)
        self.idc.add_connection(bus.atomq)

    def E(self, vdg, vqg):
        return self.ke * sqrt(vdg**2 + vqg**2)

    def phi(self, vdg, vqg):
        return atan2(vdg, vqg)

    def get_angles(self, idc, vdg, vqg):

        e = sqrt(6) * self.E(vdg, vqg)

        if e == 0.0:
           return 0.0, 0.0

        k = cos(self.alpha_cmd) - 2 * self.Lc * self.w * idc / e
        mu = -self.alpha_cmd + acos(k)
        alpha = self.alpha_cmd

        if mu >= PI_3 or mu + alpha >= pi:
            mu = PI_3
            alpha = PI_3 - acos((2 * self.Lc * self.w * idc) / e)

        return alpha, mu

    def iqg_com(self, vdg, vqg, idc, alpha, mu):

        E = self.E(vdg, vqg)  
        k1 = 2 * sqrt(3) / pi
        k2 = 3 * sqrt(3) * E / (pi * self.Lc * self.w)
        k3 = 3 * sqrt(2) * E / (4 * pi * self.Lc * self.w)

        return (k1 * idc * (sin(mu + alpha - PI5_6) - sin(alpha - PI5_6))
              * k2 * cos(alpha) * (cos(mu + alpha) - cos(alpha))
              + k3 * (cos(2*mu) - cos(2*alpha + 2*mu)))

    def idg_com(self, vdg, vqg, idc, alpha, mu):

        E = self.E(vdg, vqg)
        k1 = 2 * sqrt(3) / pi
        k2 = 3 * sqrt(2) * E / (pi * self.Lc * self.w)
        k3 = 3 * sqrt(2) * E / (4 * pi * self.Lc * self.w)
        k4 = 3 * sqrt(2) * E / (2 * pi * self.Lc * self.w)

        return (k1 * idc * (-cos(mu + alpha - PI5_6) + cos(alpha - PI5_6))
              * k2 * cos(alpha) * (sin(mu + alpha) - sin(alpha))
              + k3 * (sin(2*mu) - sin(2*alpha + 2*mu))
              - k4 * mu)

    def iqg_cond(self, idc, alpha, mu):
        return 2 * sqrt(3) / pi * idc * (sin(alpha + PI7_6) - sin(alpha + mu + PI5_6))

    def idg_cond(self, idc, alpha, mu):
        return 2 * sqrt(3) / pi * idc * (-cos(alpha + PI7_6) + cos(alpha + mu + PI5_6))

    def iqg(self, idc, vdg, vqg, alpha, mu):
        return (self.iqg_com(vdg, vqg, idc, alpha, mu) + self.iqg_cond(idc, alpha, mu))

    def idg(self, idc, vdg, vqg, alpha, mu):
        return (self.idg_com(vdg, vqg, idc, alpha, mu) + self.idg_cond(idc, alpha, mu))

    @staticmethod
    def get_iq(self, t):  # iq depends on: vd, vq, idc

        vdg = self.vdg.q
        vqg = self.vqg.q
        idc = self.idc.q
        phi = self.phi(vdg, vqg)
        alpha, mu = self.get_angles(idc, vdg, vqg)

        return (self.iqg(idc, vdg, vqg, alpha, mu) * cos(phi)
              - self.idg(idc, vdg, vqg, alpha, mu) * sin(phi))

    @staticmethod
    def get_id(self, t):  # id depends on: vd, vq, idc

        vdg = self.vdg.q
        vqg = self.vqg.q
        idc = self.idc.q
        phi = self.phi(vdg, vqg)
        alpha, mu = self.get_angles(idc, vdg, vqg)

        return (self.iqg(idc, vdg, vqg, alpha, mu) * sin(phi)
              + self.idg(idc, vdg, vqg, alpha, mu) * cos(phi))

    @staticmethod
    def didc(self, idc):  # idc depends on: vd, vq, vdc

        vdg = self.vdg.q
        vqg = self.vqg.q
        alpha, mu = self.get_angles(idc, vdg, vqg)
        E = self.E(vdg, vqg)
        k1 = 3 * sqrt(3) * sqrt(2) / pi

        e = k1 * E * cos(alpha)
        req = self.Rdc + 3/pi * self.Lc * self.w

        return (e - req * idc - self.vdc.q) / (self.Ldc + 2*self.Lc)

    @staticmethod
    def dvdc(self, vdc):  # vdc depends on: idc

        return (self.idc.q - vdc / self.R) / self.C


class TRLoadDQ2(Device):

    """Transformer rectifier load average model simplified (alpha=0).

    Sd = sqrt(3/2) * 2 * sqrt(3)/pi * cos(phi)
    Sq = -sqrt(3/2) * 2 * sqrt(3)/pi * sin(phi)
    
    id = idc/Sd
    iq = idc/Sq

    idc = Sd*id + Sq*iq 

    edc = Sd*vd + Sq*vq

    edc - vdc = idc*Rdc + didc*Ldc

    didc*Ldc = (edc - vdc - idc*Rdc) / Ldc

    idc = dvdc*C + vdc/R

    did = (pi/(3*sqrt(2)*cos(phi))) * didc
    diq = (-pi/(3*sqrt(2)*sin(phi))) * didc
    dedc = 

    """

    def __init__(self, name, w=2*pi*60.0, Lc=76.53e-6, Rdc=0.0,
                 Ldc=0.383e-3, R=3.16, C=0.384e-3, id0=0.0,  
                 vdc0=0.0, dq_i=1e0, dq_v=1e0):

        self.name = name

        # params:

        self.w = w
        self.Lc = Lc 
        self.Rdc = Rdc
        self.Ldc = Ldc
        self.R = R
        self.C = C

        # intial conditions:

        self.id0 = id0
        self.vdc0 = vdc0

        # delta q:
                
        self.dq_i = dq_i
        self.dq_v = dq_v

        # cached:

        self.S = sqrt(3/2) * 2 * sqrt(3) / pi
        self.S2 = self.S**2
        self.Req = Rdc + 3 / pi * Lc * w
        self.Leq = Ldc + 2 * Lc

        # call super:

        Device.__init__(self, name)

        # atoms:

        self.iq = SourceAtom("iq", source_type=SourceType.CONSTANT, u0=0.0,
                              units="A", dq=dq_i)

        self.id = StateAtom("id", x0=id0, derfunc=self.did, units="A", dq=dq_i)
        self.vdc = StateAtom("vdc", x0=vdc0, derfunc=self.dvdc, units="V", dq=dq_v)

        self.add_atoms(self.id, self.iq, self.vdc)

        # atom connections:

        self.vdc.add_connection(self.id)
        self.id.add_connection(self.vdc)
        
        # ports:

        self.atomd = self.id
        self.atomq = self.iq

        self.vd = None  # terminal voltage d atom
        self.vq = None  # terminal voltage q atom

    def connect(self, bus):

        self.vd = bus.atomd
        self.vq = bus.atomq

        self.id.add_connection(bus.atomd)
        self.id.add_connection(bus.atomq)

    @staticmethod
    def did(self, id):  # id depends on: vd, vdc

        return (self.vd.q * self.S2 - self.Req * id
                - self.vdc.q * self.S) / self.Leq

    @staticmethod
    def dvdc(self, vdc):  # vdc depends on: id 

        return (self.id.q / self.S - vdc / self.R) / self.C


class Pendulum(Device):

    """  Simple non-linear example.
    """

    def __init__(self, name, r=1.0, l=1.0, theta0=0.0, omega0=0.0,
                 dq_omega=1e-3, dq_theta=1e-3):

        Device.__init__(self, name)

        self.l = l
        self.r = r
        self.g = 9.81

        self.omega = StateAtom("omega", x0=omega0, derfunc=self.domega, units="rad/s", dq=dq_omega)
        self.theta = StateAtom("theta", x0=theta0, derfunc=self.dtheta, units="rad",   dq=dq_theta)
        
        self.add_atoms(self.omega, self.theta)

        self.omega.add_connection(self.theta)
        self.theta.add_connection(self.omega)

        # jacobian:
        self.omega.add_jacfunc(self.theta, self.j21)
        self.omega.add_jacfunc(self.omega, self.j22)
        self.theta.add_jacfunc(self.omega, self.j12)

    @staticmethod
    def domega(self, omega):
        return -(self.r*omega + self.g / self.l * sin(self.theta.q))

    @staticmethod
    def dtheta(self, theta):
        return self.omega.q

    @staticmethod
    def j12(self):
        return 1.0

    @staticmethod
    def j21(self):
        return -self.g / self.l * cos(self.theta.x)

    @staticmethod
    def j22(self):
        return -self.r


class CoupledPendulums(Device):

    """  
    dw1 = -g/l1 * sin(th1)/cos(th1)
          + w1*w1 * sin(th1)/cos(th1)
          + k*l2/(m1*l1) * sin(th2) / cos(th1)
          + k*l1/(m1*l1) * sin(th1) / cos(th1) 

    dw2 = -g/l2 * sin(th2)/cos(th2)
          + w2*w2 * sin(th2)/cos(th2)
          + k*l1/(m2*l2) * sin(th1) / cos(th2)
          + k*l2/(m2*l2) * sin(th2) / cos(th2) 

    dth1 = w1
    dth2 = w2

    """

    def __init__(self, name, k=1.0, r1=1.0, r2=1.0, l1=1.0, l2=1.0, m1=1.0,
                 m2=1.0, th10=0.0, w10=0.0, th20=0.0, w20=0.0, dq_w=1e-3, dq_th=1e-3):

        Device.__init__(self, name)

        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        self.g = 9.81

        self.w1  = StateAtom("w1",  x0=w10,  derfunc=self.dw1,  units="rad/s", dq=dq_w)
        self.th1 = StateAtom("th1", x0=th10, derfunc=self.dth1, units="rad",   dq=dq_th)
        self.w2  = StateAtom("w2",  x0=w20,  derfunc=self.dw2,  units="rad/s", dq=dq_w)
        self.th2 = StateAtom("th2", x0=th20, derfunc=self.dth2, units="rad",   dq=dq_th)

        self.add_atoms(self.w1, self.th1, self.w2, self.th2)

        self.w1.add_connection(self.th1)
        self.w1.add_connection(self.w2)
        self.th1.add_connection(self.w1)
        self.w2.add_connection(self.th2)
        self.w2.add_connection(self.w1)
        self.th2.add_connection(self.w2)

        # jacobian:

        self.w1.add_jacfunc(self.th1, self.jw1_th1)
        self.w1.add_jacfunc(self.th2, self.jw1_th2)
        self.w1.add_jacfunc(self.w1, self.jw1_w1)
        self.w2.add_jacfunc(self.th1, self.jw2_th1)
        self.w2.add_jacfunc(self.th2, self.jw2_th2)
        self.w2.add_jacfunc(self.w2, self.jw2_w2)
        self.th1.add_jacfunc(self.w1, self.jth1_w1)
        self.th2.add_jacfunc(self.w2, self.jth2_w2)

    @staticmethod
    def dth1(self, th1):
        return self.w1.q

    @staticmethod
    def dw1(self, w1):
        return (-self.r1 * w1 - self.g / self.l1
                * sin(self.th1.q) / cos(self.th1.q)
                + w1**2 * sin(self.th1.q) / cos(self.th1.q)
                + self.k * self.l2/(self.m1 * self.l1)
                * sin(self.th2.q) / cos(self.th1.q)
                + self.k * self.l1 / (self.m1 * self.l1)
                * sin(self.th1.q) / cos(self.th1.q)) 

    @staticmethod
    def dth2(self, th2):
        return self.w2.q

    @staticmethod
    def dw2(self, w2):
        return (-self.r2 * w2 - self.g / self.l2
                * sin(self.th2.q) / cos(self.th2.q)
                + w2**2 * sin(self.th2.q) / cos(self.th2.q)
                + self.k * self.l1 / (self.m2 * self.l2)
                * sin(self.th1.q) / cos(self.th2.q)
                + self.k * self.l2 / (self.m2 * self.l2)
                * sin(self.th2.q) / cos(self.th2.q)) 

    @staticmethod
    def jw1_th1(self, w1, th1):
        return (sin(th1)**2 * w1**2
                / cos(th1)**2 + w1**2
                + self.k * self.l2 * sin(th1) * sin(self.th2.q) 
                / (self.l1 * self.m1 * cos(th1)**2)
                + (self.k * sin(th1)**2)
                / (self.m1 * cos(th1)**2)
                - (self.g * sin(th1)**2)
                / (self.l1 * cos(th1)**2)
                + self.k / self.m1 - self.g / self.l1)
	 
    @staticmethod
    def jw1_th2(self, w1, th2):
        return (self.k * self.l2 * cos(th2)
               / (self.l1 * self.m1 * cos(self.th1.q)))
	
    @staticmethod
    def jw1_w1(self, w1):
        return 2.0 * sin(self.th1.q) * w1 / cos(self.th1.q) 
	
    @staticmethod
    def jw2_th1(self, w2, th1):
        return (self.k * self.l1 * cos(th1)
                / (self.l2 * self.m2 * cos(self.th2.q)))
	
    @staticmethod
    def jw2_th2(self, w2, th2):
        return (sin(th2)**2 * w2**2
                / cos(th2)**2 + w2**2
                + (self.k *  sin(th2)**2)
                / (self.m2 * cos(th2)**2)
                - (self.g *  sin(th2)**2)
                / (self.l2 * cos(th2)**2)
                + (self.k * self.l1 * sin(self.th1.q) * sin(th2))
                / (self.l2 * self.m2 * cos(th2)**2)
                + self.k / self.m2 - self.g / self.l2)
	
    @staticmethod
    def jw2_w2(self, w2):
        return 2.0 * sin(self.th2.q) * w2 / cos(self.th2.q)	

    @staticmethod
    def jth1_w1(self, th, w1):
        return 1.0	

    @staticmethod
    def jth2_w2(self, th2, w2):
        return 1.0	
                      

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

    rload = vbase**2/pload
    lload  = vbase**2/(qload/w)

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
    branch2 = LimBranch("branch2", l=l, r=r,          dq=dq_v)
    branch3 = LimBranch("branch3", l=l, r=r,          dq=dq_v)
    branch4 = LimBranch("branch4", l=l, r=r,          dq=dq_v)
    branch5 = LimBranch("branch5", l=l, r=r,          dq=dq_v)
    branch6 = LimBranch("branch6", l=l*10, r=rload,   dq=dq_v)

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

    sys = System(dq=None)

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

    tstop = 0.1
    dc = True

    sys.init_ode(dt=1.0e-3, dc=dc)
    sys.init_qss(dc=dc)

    sys.run_ode(tstop*0.1)
    sys.run_qss(tstop*0.1)

    branch1.sourced.x0 = 12.0
    branch1.sourceq.x0 = 3.0

    sys.run_ode(tstop)
    sys.run_qss(tstop)

    sys.plot((node1.atomd, node1.atomq), plot_ode=True)
    sys.plot((branch1.atomd, branch1.atomq), plot_ode=True)


def test4():

    ws = 60*pi

    sys = System(dq=1e-3)

    sm = SyncMachineDQ("sm", vfdb=3e4, dq_v=1e0, dq_th=1e-3, dq_wr=1e-3, dq_flux=1e-3)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=1e0)
    gnd = GroundNodeDQ("gnd")   

    sys.add_devices(sm, load, gnd)

    sm.connect(load, terminal="i")
    load.connect(sm, gnd)

    tstop = 6.0
    dt = None
    dc = True
    ode = True 
    qss = True
    upds = True

    chk_ss = False
    ndq = 100

    sys.initialize(dt=1.0e-3, dc=dc)

    if ode: sys.run_ode(tstop*0.2, reset_after=True)
    if qss: sys.run_qss(tstop*0.2, reset_after=False)
    
    load.r = 1.0
    
    if ode: sys.run_ode(tstop, reset_after=True)
    if qss: sys.run_qss(tstop, reset_after=False, chk_ss_period=chk_ss, chk_ss_ndq=ndq)

    sys.plot(sm.atomd, sm.atomq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(sm.wr, sm.th, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(sm.fkq, sm.fkd, sm.ffd, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(load.atomd, load.atomq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)


def test5():

    ws = 60*pi

    sys = System(dq=1e-3)

    sm = SyncMachineDQ2("sm", vfdb=2700,
                       dq_current=1e0,
                       dq_th=1e-3,
                       dq_wr=1e-3,
                       dq_flux=1e-3)

    bus = LimNodeDQ("bus", c=0.001, g=1.0/2.8, w=ws, dq=1e0)
    #load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=1e0)
    #gnd = GroundNodeDQ("gnd")  

    sys.add_devices(sm, bus)#, load, gnd)

    sm.connect(bus)
    bus.connect(sm, terminal="j")
    #bus.connect(load, terminal="i")
    #load.connect(bus, gnd)

    tstop = 20.0
    dt = None
    dc = True
    ode = True 
    qss = False
    upds = False

    chk_ss = False
    ndq = 100

    sys.initialize(dt=1.0e-3, dc=dc)

    #print(sys.x)
    #print(sys.u)

    #if qss: sys.run_qss(tstop*0.2, reset_after=True)
    if ode: sys.run_ode(tstop*0.2, reset_after=False)

    #print(sys.x)
    #print(sys.u)
    
    bus.g = bus.g*1.2
    
    #if qss: sys.run_qss(tstop*0.21, reset_after=True, chk_ss_period=chk_ss, chk_ss_ndq=ndq)
    if ode: sys.run_ode(tstop*0.21, reset_after=False)
    
    sys.plot(bus.vd, bus.vq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(sm.wr, sm.th, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(sm.fkq, sm.fkd, sm.ffd, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
    sys.plot(load.atomd, sm.ids, load.atomq, sm.iqs, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)


def test6():

    ws = 60*pi    # system radian frequency
    vfdb = 90.1   # sm rated field voltage
    VLL = 4160.0  # bus voltage
    vref = 1.0    # pu voltage setpoint

    dq_i = 1e-1
    dq_v = 1e-1

    sys = System(dq=1e-3)

    sm = SyncMachineDQ4("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v,
                        dq_th=1e-3, dq_wr=1e-3, dq_f=1e-3)

    bus = LimNodeDQ("bus", c=1e-4, g=0.0, w=ws, dq=dq_v)

    avr = AC8B("avr", vref=vref, dq=1e-4)

    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, iq0=1000, dq=dq_i)

    gnd = GroundNodeDQ("gnd")

    sys.add_devices(sm, bus, load, gnd, avr)

    sm.connect(bus, avr)

    bus.connect(sm, terminal="j")
    bus.connect(load, terminal="i")

    load.connect(bus, gnd)

    tstop = 10.0

    dt = 1.0e-3
    dc = True

    ode = True 
    qss = False
    upds = False
    upd_bins = 200

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
        load.r *= 0.8

    sys.schedule(event, tstop*0.2)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss)

    sys.plot(avr.x1, avr.x2, avr.x3, avr.vfd, **plotargs)
    sys.plot(bus.vd, bus.vq, **plotargs)
    sys.plot(sm.wr, sm.th, **plotargs)
    sys.plot(sm.fkq, sm.fkd, sm.ffd, **plotargs)
    sys.plot(load.id, load.iq, **plotargs)


def test7():

    ws = 2*60*pi    # system radian frequency
    VLL = 4160.0  # bus nominal line-to-line rms voltage
    theta = pi/8

    dq_i = 1e-2
    dq_v = 1e-2

    sys = System(dq=1e-3)

    Vd = VLL * cos(theta)
    Vq = VLL * sin(theta)

    source = LimBranchDQ("source", l=1e-5, r=0.0, vd0=Vd, vq0=Vq,  w=ws, dq=dq_i)

    bus1 = LimNodeDQ("bus1", c=1e-5, g=0.0, w=ws, vd0=Vd, vq0=Vq, dq=dq_v)

    bus2 = LimNodeDQ("bus2", c=1e-5, g=0.0, w=ws, vd0=Vd, vq0=Vq, dq=dq_v)

    cable = LimBranchDQ("cable", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)

    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=dq_i)

    trload = TRLoadDQ2("trload", w=ws, dq_i=dq_i, dq_v=dq_v)

    gnd = GroundNodeDQ("gnd")

    sys.add_devices(source, bus1, bus2, cable, load, gnd)
    sys.add_devices(trload)

    source.connect(gnd, bus1)
    bus1.connect(source, terminal="j")
    
    bus1.connect(cable, terminal="i")
    cable.connect(bus1, bus2)
    bus2.connect(cable, terminal="j")
    
    load.connect(bus2, gnd)
    bus2.connect(load, terminal="i")

    trload.connect(bus2)
    bus2.connect(trload, terminal="i")

    tstop = 0.1

    dt = 1.0e-5
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
        load.r *= 0.8

    sys.schedule(event, tstop*0.2)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True)

    sys.plot(trload.id, trload.iq, trload.vdc, **plotargs)
    sys.plot(bus1.vd, bus1.vq, bus2.vd, bus2.vq, **plotargs)
    sys.plot(cable.id, cable.iq, load.id, load.iq, **plotargs)
    

def test8():

    ws = 2*60*pi  # system radian frequency
    vfdb = 90.1   # sm rated field voltage
    VLL = 4160.0  # bus nominal line-to-line rms voltage
    vref = 1.0    # pu voltage setpoint

    dq_i = 1e-1
    dq_v = 1e-1

    sys = System(dq=1e-3)

    sm = SyncMachineDQ4("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v, dq_th=1e-5, dq_wr=1e-4, dq_f=1e-3)
    avr = AC8B("avr", vref=vref, dq=1e-5)
    bus1 = LimNodeDQ("bus1", c=1e-3, g=0.0, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus2 = LimNodeDQ("bus2", c=1e-3, g=0.0, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    cable = LimBranchDQ("cable", l=2.3e-5, r=0.865e-2, w=ws, dq=dq_i)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=dq_i)
    trload = TRLoadDQ2("trload", w=ws, vdc0=VLL, dq_i=dq_i, dq_v=dq_v)
    gnd = GroundNodeDQ("gnd")

    sys.add_devices(sm, avr, bus1, bus2, cable, load, trload, gnd)

    sm.connect(bus1, avr)
    bus1.connect(sm, terminal="j")
    
    bus1.connect(cable, terminal="i")
    cable.connect(bus1, bus2)
    bus2.connect(cable, terminal="j")
    
    bus2.connect(load, terminal="i")
    load.connect(bus2, gnd)
    
    trload.connect(bus2)
    bus2.connect(trload, terminal="i")

    tstop = 30.0

    dt = 1.0e-4
    dc = True

    ode = True 
    qss = True
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
        load.r *= 0.90

    sys.schedule(event, tstop*0.2)

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA")

    sys.plot(sm.wr, sm.th, **plotargs)
    sys.plot(avr.x1, avr.x2, avr.x3, avr.vfd, **plotargs)
    sys.plot(bus1.vd, bus1.vq, bus2.vd, bus2.vq, **plotargs)
    sys.plot(cable.id, cable.iq, load.id, load.iq, **plotargs)
    sys.plot(trload.id, trload.vdc, **plotargs)


def test9():

    ws = 2*60*pi 
    vfdb = 90.1  
    VLL = 4160.0 
    vref = 1.0   

    dq_i = 1e-2
    dq_v = 1e-2
    dq_wr = 1e-4
    dq_th = 1e-4
    dq_f = 1e-4
    dq_avr = 1e-5

    sys = System(dq=1e-3)

    sm = SyncMachineDQ4("sm", vfdb=vfdb, VLL=VLL, dq_i=dq_i, dq_v=dq_v,
                        dq_th=dq_th, dq_wr=dq_wr, dq_f=dq_f)

    avr = AC8B("avr", vref=vref, dq=dq_avr)
    bus1 = LimNodeDQ("bus1", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    bus2 = LimNodeDQ("bus2", c=1e-3, g=1e-4, w=ws, vd0=VLL, vq0=VLL, dq=dq_v)
    cable = LimBranchDQ("cable", l=2.3e-4, r=0.865e-2, w=ws, dq=dq_i)
    #im = InductionMachineDQ("im", ws=ws/2, P=4, wr0=ws/2, dq_i=dq_i, dq_wr=dq_wr, Tb=0.0)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=dq_i)
    gnd = GroundNodeDQ("gnd")

    #sys.add_devices(sm, avr, bus1, cable, bus2, im, gnd)
    sys.add_devices(sm, avr, bus1, cable, bus2, load, gnd)

    sm.connect(bus1, avr)
    #sm.connect(bus1)
    bus1.connect(sm, terminal="j")

    bus1.connect(cable, terminal="i")
    cable.connect(bus1, bus2)
    bus2.connect(cable, terminal="j")

    load.connect(bus2, gnd)
    bus2.connect(load, terminal="i")

    #im.connect(bus2)
    #bus2.connect(im, terminal="i")

    tstop = 60.0
    dt = 1.0e-4
    dc = 1
    optimize_dq = 1
    ode = 1 
    qss = 1
    chk_ss_delay = 1.0
    #chk_ss_delay = None

    def event(sys):
        load.r *= 0.8

    sys.schedule(event, tstop*0.2)

    sys.Km = 2.0

    sys.initialize(dt=dt, dc=dc)

    sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA")

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":False, "errorband":True}

    sys.plot(sm.wr, sm.th, avr.vfd, **plotargs)
    sys.plot(avr.x1, avr.x2, avr.x3, **plotargs)
    #sys.plot(im.iqs, im.ids, im.iqr, im.idr, **plotargs)
    sys.plot(cable.id, cable.iq,  **plotargs)
    sys.plot(bus1.vd, bus1.vq, bus2.vd, bus2.vq, **plotargs)
    sys.plot(load.id, load.iq, **plotargs)


def test10():

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


def test11():

    sys = System(dq=1e-3)

    pendulum = Pendulum("pendulum", r=0.4, l=8.0, theta0=1.0, omega0=1.0,
                        dq_omega=2e-2, dq_theta=2e-2)

    sys.add_devices(pendulum)

    tstop = 60.0
    dt = 1.0e-3
    dc = 0
    optimize_dq = 0
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


def test12():

    sys = System(dq=1e-3)

    pendulums = CoupledPendulums("pendulums", k=1.0, r1=1.0, r2=1.0,
                                 l1=1.0, l2=1.0, m1=1.0, m2=1.0,
                                 th10=pi/4, w10=0.0, th20=-pi/4, w20=0.0,
                                 dq_w=1e-1, dq_th=1e-2)

    sys.add_devices(pendulums)

    tstop = 20.0
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

    #sys.schedule(event, tstop*0.1)

    sys.run(tstop, ode=ode, qss=qss, verbose=True,
            ode_method="LSODA", optimize_dq=optimize_dq,
            chk_ss_delay=chk_ss_delay)

    plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
                "plot_qss_updates":qss, "errorband":True}

    sys.plot(pendulums.w1, pendulums.w2, **plotargs)
    sys.plot(pendulums.th1, pendulums.th2, **plotargs)

    #sys.plotxy(pendulums.w1, pendulums.th1, arrows=False, ss_region=True)
    #sys.plotxy(pendulums.w2, pendulums.th2, arrows=False, ss_region=True)

    #sys.plotxyt(pendulums.w1, pendulums.th1, ss_region=True)
    #sys.plotxyt(pendulums.w2, pendulums.th2, ss_region=True)

if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    #test4()
    #test5()
    #test6()
    #test7()
    #test8()
    test9()
    #test10()
    #test11()
    #test12()


     