"""Quantized DEVS-LIM modeling and simulation framework.
"""

from math import pi, sin, cos, sqrt, floor
from collections import OrderedDict as odict

import numpy as np
import numpy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes.formatter', useoffset=False)

from scipy.integrate import odeint
from scipy.optimize import fsolve

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
                 dqerr=None, dtmin=None, dmax=1e5, units=""):

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
        self.d = max(self.d, -self.dmax*self.dq)
        self.d = min(self.d, self.dmax*self.dq)

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
                 dtmin=None, dmax=1e5, units=""):

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

        self.omega = 2.0 * pi * self.freq

        if self.freq:
            self.T = 1.0 / self.freq

        if self.source_type == SourceType.RAMP:
            self.u0 = self.u1

        self.ramp_slope = 0.0
        if (self.t2 - self.t1) > 0:
            self.ramp_slope = (self.u2 - self.u1) / (self.t2 - self.t1)

    def dint(self):

        if self.source_type == SourceType.FUNCTION:

            self.u = self.srcfunc(self.device, self.time)

        elif self.source_type == SourceType.CONSTANT:

            self.u = self.u0

        elif self.source_type == SourceType.STEP:

            if self.time < self.t1:
                self.u = self.u0
            else:
                self.u = self.u1

        elif self.source_type == SourceType.SINE:

            if self.time >= self.t1:
                self.u = self.u0 + self.ua*sin(self.omega*self.time + self.phi)
            else:
                self.u = self.u0

        elif self.source_type == SourceType.PWM:

            pass # todo

        elif self.source_type == SourceType.RAMP:

            if self.time <= self.t1:
                self.u = self.u1
            elif self.time <= self.t2:
                self.u = self.u1 + (self.time - self.t1) * self.d 
            else:
                self.u = self.u2

        elif self.source_type == SourceType.FUNCTION:

            self.u = self.srcfunc()

        self.tlast = self.time

        self.x = self.u

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
                 dtmin=None, dmax=1e5, units=""):

        Atom.__init__(self, name=name, x0=x0, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.derfunc = derfunc

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

    @staticmethod
    def compute_odes(x, t, sys):

        """Returns array of derivatives from state atoms. This function must be
        a static method in order to be passed as a delgate to the
        scipy ode integrator function.
        """

        dx_dt = [0.0]*sys.n

        for atom in sys.state_atoms:
            atom.q = x[atom.index]

        for atom in sys.state_atoms:
            dx_dt[atom.index] = atom.f()

        return dx_dt

    def solve_dc(self):

        xi = [0.0]*self.n
        for atom in self.state_atoms:
            xi[atom.index] = atom.x0

        xdc = fsolve(self.compute_odes, xi, args=(0, sys))

        for atom in self.state_atoms:
            atom.x0 = xdc[atom.index]

    def initialize(self, t0=0.0, dt=1e-4, dc=False):

        self.time = t0
        self.dt = dt

        if dc: self.solve_dc()

        for atom in self.state_atoms:   
            atom.initialize(self.time)

        for atom in self.source_atoms:   
            atom.initialize(self.time)

    def run_ode(self, tstop, reset_after=True):

        self.tstop = tstop

        if reset_after:
            self.save_state()

        print("Non-linear ODE Simulation started...")

        xi = [0.0]*self.n
        for atom in self.state_atoms:
            xi[atom.index] = atom.x

        t = np.arange(self.time, self.tstop, self.dt)
        x = odeint(self.compute_odes, xi, t, args=(sys,))

        for i, tval in enumerate(t):

            for atom in self.state_atoms:
                atom.q = x[i, atom.index]
                atom.save_ode(tval, atom.q)

            for atom in self.source_atoms:
                atom.save_ode(tval, atom.dint())

        for atom in self.state_atoms:
            xf = x[-1, atom.index]
            atom.x = xf
            atom.q = xf

        for atom in self.source_atoms:
            atom.dint()
            atom.q = atom.q

        print("Non-linear ODE Simulation complete.")

        if reset_after:
            self.restore_state()

    def run_qss(self, tstop, fixed_dt=None, verbose=False, reset_after=True,
                chk_ss_period=None, chk_ss_ndq=None):

        print("QSS Simulation started...")

        if reset_after:
            self.save_state()

        self.tstop = tstop

        ndq = 2
        if chk_ss_ndq: ndq = chk_ss_ndq
        check_ss_clock = 0.0
        check_ss_tlast = self.tstop = tstop

        # start by updating all atoms:

        for i in range(1):
            for atom in self.atoms:
                atom.update(self.time)
                atom.save()
                atom.save()

        if fixed_dt:

            while(self.time <= self.tstop):

                for atom in self.source_atoms:
                    atom.step(self.time)

                for atom in self.state_atoms:
                    atom.step(self.time)

                self.time += fixed_dt

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

            while self.time < self.tstop:

                self.advance()

                if chk_ss_period:
                    if check_ss_clock > chk_ss_period:
                        check_ss_clock = 0.0
                        self.check_steadystate(ndq=ndq)

                if verbose and self.time-last_print_time > 0.1:
                    print("t = {0:5.2f} s".format(self.time))
                    last_print_time = self.time

                check_ss_clock += self.time - tlast 
                tlast = self.time
                
            # force time to tstop and do one update at time = tstop:

            self.time = self.tstop

            for atom in self.atoms:
                atom.update(self.time)
                atom.save()

        if reset_after:
            self.restore_state()

        print("QSS Simulation complete.")

    def check_steadystate(self, ndq=2, apply_if_true=True):

        is_ss = True

        x = [0.0]*self.n
        for atom in self.state_atoms:
            x[atom.index] = atom.x

        self.save_state()

        xss = fsolve(self.compute_odes, x, args=(0, sys))

        # debug:
        for i in range(self.n):
            print(x[i]-xss[i])

        self.restore_state()

        for atom in self.state_atoms:

            if abs(atom.x - xss[atom.index]) > atom.dq*ndq:
                is_ss = False
                break

        if is_ss and apply_if_true:

            for atom in self.state_atoms:
                atom.x = xss[atom.index]
                atom.q = atom.x

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

    def plot(self, *atoms, plot_qss=False, plot_ss=False,
                    plot_ode=False, plot_qss_updates=False, 
                    plot_ss_updates=False, plot_ode_updates=False,
                    legend=True):

        c, j = 1, 1
        r = len(atoms)/c

        if r % c > 0.0: r += 1

        nbins = 1000

        for i, atom in enumerate(atoms):

            plt.subplot(r, c, j)

            ax1 = plt.gca()
            ax1.set_ylabel("{} ({})".format(atom.full_name(),
                            atom.units), color='r')
            ax1.grid()

            ax2 = None

            if plot_qss_updates or plot_ss_updates:

                ax2 = ax1.twinx()
                ylabel = "updates"
                ax2.set_ylabel(ylabel, color='b')

            if plot_qss_updates:

                label = "upds per {} s".format(atom.tout[-1]/nbins)
                ax2.hist(atom.tout, nbins, alpha=0.5,
                         color='b', label=label, density=False)

            if plot_ss_updates:
                ax2.plot(self.tout_ss, self.nupd_ss, 'b', label="ss_upds")

            if plot_qss:

                lbl = "{} qss ({})".format(atom.full_name(), atom.units)

                ax1.plot(atom.tout, atom.qout,
                         marker='.',
                         markersize=4,
                         markerfacecolor='none',
                         markeredgecolor='r',
                         markeredgewidth=0.5,
                         linestyle='none',
                         label=lbl)

            if plot_ss:

                lbl = "{} ss ({})".format(atom.full_name(), atom.units)

                ax1.plot(atom.tout_ss, atom.xout_ss, 
                         color='r',
                         linewidth=1.0,
                         linestyle='dashed',
                         label=lbl)

            if plot_ode:

                lbl = "{} ode ({})".format(atom.full_name(), atom.units)

                ax1.plot(atom.tout_ode, atom.xout_ode, 
                         color='k',
                         alpha=0.6,
                         linewidth=1.0,
                         linestyle='dashed',
                         label=lbl)

            lines1, labels1 = ax1.get_legend_handles_labels()

            if ax2:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1+lines2, labels1+labels2)
            else:
                ax1.legend(lines1, labels1)

            plt.xlabel("t (s)")
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

    def __init__(self, atom, other, coefficient=1.0, coeffunc=None):

        self.atom = atom
        self.other = other
        self.coefficient = coefficient
        self.coeffunc = coeffunc
        self.device = None

        self.other.broadcast_to.append(self.atom)

    def compute_coefficient(self):

        if self.coeffunc:
            return self.coeffunc(self.device)
        else:
            return self.coefficient                                                  

    def value(self):

        if isinstance(self.other, StateAtom):
            return self.compute_coefficient() * self.other.q

        elif isinstance(self.other, SourceAtom):
            return self.compute_coefficient() * self.other.dint()


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

        self.atom = StateAtom("state", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="V")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii)

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.atom.add_connection(branch.atom, coeffunc=self.aij)

        elif terminal == "j":
            self.atom.add_connection(branch.atom, coeffunc=self.aji)
        
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

        self.atom = StateAtom("state", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="A")

        self.add_atoms(self.source, self.atom)

        self.atom.add_connection(self.source, coeffunc=self.bii)

    def connect(self, inode, jnode):

        self.atom.add_connection(inode.atom, coeffunc=self.aij)
        self.atom.add_connection(jnode.atom, coeffunc=self.aji)
        
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


    def __init__(self, name, v, th0=0.0):

        Device.__init__(self, name)

        self.v = v

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

        self.sourced = SourceAtom("id", source_type=source_type, u0=id0, u1=id1,
                                  u2=id2, ua=ida, freq=freqd, phi=phid,
                                  duty=dutyd, t1=td1, t2=td2, dq=dq, units="A")

        self.sourceq = SourceAtom("iq", source_type=source_type, u0=iq0, u1=iq1,
                                  u2=iq2, ua=iqa, freq=freqq, phi=phiq,
                                  duty=dutyq, t1=tq1, t2=tq2, dq=dq, units="A")

        self.atomd = StateAtom("stated", x0=vd0, coeffunc=self.aii, dq=dq,
                               units="V")

        self.atomq = StateAtom("stateq", x0=vq0, coeffunc=self.aii, dq=dq,
                               units="V")

        self.add_atoms(self.sourced, self.sourceq, self.atomd, self.atomq)

        self.atomd.add_connection(self.sourced, coeffunc=self.bii)
        self.atomq.add_connection(self.sourceq, coeffunc=self.bii)

        self.vd = self.atomd
        self.vq = self.atomq

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.atomd.add_connection(branch.atomd, coeffunc=self.aij)
            self.atomq.add_connection(branch.atomq, coeffunc=self.aij)
        elif terminal == "j":
            self.atomd.add_connection(branch.atomd, coeffunc=self.aji)
            self.atomq.add_connection(branch.atomq, coeffunc=self.aji)

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

    vqs = rs*iq + wr*Ld*id + wr*fd

                               wr*Ld*ids  (j)
                  Rs     Lls      ,^.
            .----VVVV----UUUU----<- +>----o
            |         --->        `.'     +
          +,-.        iqs                vqs
    wr*fq (   )                           -
          -`-'                            o
     (i)   _|_                           _|_
            -                             -

                               wr*Lq*iqs
                  Rs     Lls      ,^.
            .----VVVV----UUUU----<+ ->----o
            |         --->        `.'     +
          +,-.        ids                vds
    wr*fd (   )                           -
          -`-'                            o
           _|_                           _|_
            -                             -

                      
     diqs*Lls = (1/Lls) * (wr*fq - iqs*Rs + wr*Ld*ids - vqs)
     dids*Lls = (1/Lls) * (wr*fd - ids*Rs - wr*Lq*iqs - vds)

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


        # diqs = (1/Lls) * (vqs - rs*iqs - wr*Ld*ids - wr*fd)
        # dids = (1/Lls) * (vds - rs*ids - wr*Lq*iqs - wr*fq)

        return (1.0 / self.Lls) * (self.vqs()
                                   - self.rs * iqs
                                   - self.wr.q * self.Ld * self.ids.q
                                   - self.wr.q * self.fq(self.fkq.q))
    @staticmethod
    def dids(self, ids):  # ids relies on ext_d, wr, iqs, fkd

        return (1.0 / self.Lls) * (self.vds()
                                   - self.rs * ids
                                   + self.wr.q * self.Lq * self.iqs.q
                                   + self.wr.q * self.fq(self.fkd.q))
    
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

    sm = SyncMachineDQ("sm", vfdb=3e4, dq_v=1e-1, dq_th=1e-4, dq_wr=1e-4, dq_flux=1e-4)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=1e-1)
    gnd = GroundNodeDQ("gnd")   

    sys.add_devices(sm, load, gnd)

    sm.connect(load, terminal="i")
    load.connect(sm, gnd)

    tstop = 10.0
    dt = None
    dc = True
    ode = True 
    qss = True
    upds = True

    chk_ss = True
    ndq = 100

    sys.initialize(dt=1.0e-3, dc=dc)

    if ode: sys.run_ode(tstop*0.2, reset_after=True)
    if qss: sys.run_qss(tstop*0.2, reset_after=False)
    
    load.r = 1.0
    
    if ode: sys.run_ode(tstop, reset_after=True)
    if qss: sys.run_qss(tstop, reset_after=False, chk_ss_period=chk_ss, chk_ss_ndq=ndq)

    if 1:
        sys.plot(sm.atomd, sm.atomq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(sm.wr, sm.th, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(sm.fkq, sm.fkd, sm.ffd, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(load.atomd, load.atomq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)

    if 0:
        with open(r"c:\temp\initcond.txt", "w") as f:
            for atom in sys.atoms:
                f.write("    {}0 = {}\n".format(atom.name, atom.x))


def test5():

    ws = 60*pi

    sys = System(dq=1e-3)

    sm = SyncMachineDQ2("sm", vfdb=2700,
                       dq_current=1e0,
                       dq_th=1e-3,
                       dq_wr=1e-3,
                       dq_flux=1e-3)

    bus = LimNodeDQ("bus", c=0.001, w=ws, dq=1e0)
    load = LimBranchDQ("load", l=3.6e-3, r=2.8, w=ws, dq=1e0)
    gnd = GroundNodeDQ("gnd")  

    sys.add_devices(sm, bus, load, gnd)

    sm.connect(bus)
    bus.connect(sm, terminal="j")
    bus.connect(load, terminal="i")
    load.connect(bus, gnd)

    tstop = 20.0
    dt = None
    dc = True
    ode = True 
    qss = False
    upds = False

    chk_ss = False
    ndq = 100

    if qss: sys.init_qss(dc=dc)
    if ode: sys.init_ode(dt=1.0e-3, dc=dc)

    print(sys.x)
    print(sys.u)

    if qss: sys.run_qss(tstop*0.2, reset_after=False)
    if ode: sys.run_ode(tstop*0.2, reset_after=False)

    print(sys.x)
    print(sys.u)
    
    #load.r = load.r*0.5
    
    if qss: sys.run_qss(tstop*0.2001, reset_after=True, chk_ss_period=chk_ss, chk_ss_ndq=ndq)
    if ode: sys.run_ode(tstop*0.2001, reset_after=False)
    
    if 1:
        sys.plot(bus.vd, bus.vq, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(sm.wr, sm.th, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(sm.fkq, sm.fkd, sm.ffd, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        sys.plot(load.atomd, sm.ids, load.atomq, sm.iqs, plot_ode=ode, plot_qss=qss, plot_qss_updates=upds)
        

if __name__ == "__main__":

    #test1()
    #test2()
    #test3()
    test4()
    #test5()

     