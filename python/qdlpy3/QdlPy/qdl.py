"""Quantized DEVS-LIM modeling and simulation framework.
"""

from math import pi, sin, cos, sqrt, floor
from collections import OrderedDict as odict

import numpy as np
import numpy.linalg as la

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    CONST_NODE = "CONST_NODE"
    CONST_BRANCH = "CONST_BRANCH"


class PortDirection(object):

    IN = "IN"
    OUT = "OUT"
    INOUT = "INOUT"


# ============================= Qdl Model ======================================


class Atom(object):

    """ Qdl Atom.
    """

    def __init__(self, name, is_latent=True, a=0.0, b=0.0, c=0.0,
                 source_type=SourceType.NONE, lim_type=LimAtomType.NONE,
                 x0=0.0, x1=0.0, x2=0.0, xa=0.0, freq=0.0, phi=0.0, duty=0.0,
                 t1=0.0, t2=0.0, derivative=None, dq=None, dqmin=None,
                 dqmax=None, dqerr=None, dtmin=None, dmax=1e5, units="",
                 srcfunc=None, fixed_dt=None, output_scale=1.0):

        # params:

        self.name = name
        self.is_latent = is_latent
        self.a = a
        self.b = b
        self.c = c
        self.source_type = source_type
        self.lim_type = lim_type
        self.x0 = x0
        self.x1 = x1 
        self.x2 = x2
        self.xa = xa
        self.freq = freq
        self.phi = phi
        self.duty = duty
        self.t1 = t1
        self.t2 = t2
        self.derivative = derivative
        self.dq = dq
        self.dqmin = dqmin
        self.dqmax = dqmax
        self.dqerr = dqerr
        self.dtmin = dtmin
        self.dmax = dmax
        self.units = units
        self.srcfunc = srcfunc
        self.fixed_dt = fixed_dt
        self.output_scale = output_scale

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

        self.updates = 0
        self.tout = None  # output times quantized output
        self.qout = None  # quantized output 
        self.tzoh = None  # zero-order hold output times quantized output
        self.qzoh = None  # zero-order hold quantized output 
        self.tout2 = None  # output times quantized output
        self.qout2 = None  # quantized output 
        self.nupd2 = None  # cummulative update count

        # atom connections:

        self.receive_from = {}
        self.broadcast_to = []

        # parent object references:

        self.sys = None
        self.device = None

        # derived:

        self.ainv = 0.0
        if self.a > 0:
            self.ainv = 1.0 / a

        self.omega = 2.0 * pi * self.freq

        if self.freq:
            self.T = 1.0 / self.freq

        if self.source_type == SourceType.RAMP:
            self.x0 = self.x1

        self.ramp_slope = 0.0
        if (self.t2 - self.t1) > 0:
            self.ramp_slope = (self.x2 - self.x1) / (self.t2 - self.t1)

        self.implicit = True

    def full_name(self):

        return self.device.name + "." + self.name

    def connect(self, atom, gain=1.0):

        self.receive_from[atom] = gain
        atom.broadcast_to.append(self)

    def connects(self, *atoms):

        for atom in atoms:
            self.receive_from[atom] = 1.0
            atom.broadcast_to.append(self)

    def update_gain(self, atom, gain):

        if atom in self.receive_from:
            self.receive_from[atom] = gain

    def initialize(self, t0):

        self.tlast = t0
        self.time = t0
        self.tnext = _INF

        # init state:

        if self.source_type == SourceType.FUNCTION:

            self.x = self.srcfunc()
            self.q = self.x
            self.q0 = self.x

        else:
            self.x = self.x0
            self.q = self.x0
            self.q0 = self.x0

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

    def update(self, time):

        self.time = time
        self.updates += 1
        self.triggered = False   # reset triggered flag

        self.d = self.f(self.q)
        self.d = max(self.d, -self.dmax)
        self.d = min(self.d, self.dmax)

        self.dint()
        self.quantize()
        self.ta()

        # trigger external update if quantized output changed:
        
        if self.q != self.q0:
            self.save()
            self.q0 = self.q
            self.trigger()
            self.update_dq()

    def step(self, time):

        self.time = time
        self.updates += 1
        self.d = self.f(self.x)
        self.dint()
        self.q = self.x
        self.save()
        self.q0 = self.q

    def dint(self):

        if self.source_type == SourceType.NONE:

            self.x += self.d * (self.time - self.tlast)
        
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

    def quantize(self):
        
        interp = False
        change = False

        self.d0 = self.d

        if self.source_type in (SourceType.FUNCTION, SourceType.STEP, SourceType.SINE):

            # non-derivative based:

            self.q = self.x

        elif self.source_type in (SourceType.NONE, SourceType.RAMP):

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

                self.d = self.f(self.q)

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

        if self.source_type == SourceType.NONE:

            if self.d > 0.0:
                self.tnext = self.time + (self.qhi - self.x) / self.d
            elif self.d < 0.0:
                self.tnext = self.time + (self.qlo - self.x) / self.d
            else:
                self.tnext = _INF
        
        elif self.source_type == SourceType.RAMP:

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
            
                if theta < pi/2.0:  # quadrant I
                    self.tnext = t0 + (asin(min(1.0, (x + self.dq)/self.xa))) / self.omega

                elif theta < pi:  # quadrant II
                    self.tnext = t0 + self.T/2.0 - (asin(max(0.0, (x - self.dq)/self.xa))) / self.omega

                elif theta < 3.0*pi/2:  # quadrant III
                    self.tnext = t0 + self.T/2.0 - (asin(max(-1.0, (x - self.dq)/self.xa))) / self.omega

                else:  # quadrant IV
                    self.tnext = t0 + self.T + (asin(min(0.0, (x + self.dq)/self.xa))) / self.omega


        elif self.source_type == SourceType.FUNCTION:

            self.tnext = self.time + self.srcdt

        else:

            self.tnext = _INF

        self.tnext = max(self.tnext, self.tlast + self.dtmin)

    def f(self, qval):

        d = 0.0

        if self.source_type == SourceType.NONE:

            if self.derivative:  # delegate  

                if self.device:
                    d = self.device.derivative()
                else:
                    d = self.derivative()

            else:  # auto ode derivative: 
                xsum = 0.0
                for atom, gain in self.receive_from.items():
                    xsum += atom.q * gain
                d = self.ainv * (self.c - qval * self.b + xsum)

        elif self.source_type == SourceType.RAMP:

            d = self.ramp_slope

        return d

    def trigger(self):

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

    def get_error(self, typ="l2"):

        # interpolate qss to ss time vector:
        # this function can only be called after state space and qdl simualtions
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

    def __repr__(self):

        return self.full_name()

    def __str__(self):

        return __repr__(self)


class ComplexAtom(Atom):

    def __init__(self, *args, freq1=1.0, **kwargs):

        Atom.__init__(self, *args, **kwargs)

        self.x0 = complex(self.x0, 0.0)

        # other member variables:
        self.qlo = complex(0.0, 0.0)   
        self.qhi = complex(0.0, 0.0)     
        self.x = self.x0    
        self.d = complex(0.0, 0.0)       
        self.d0 = complex(0.0, 0.0)      
        self.q = self.x0      
        self.q0 = self.x0 
        
        self.freq1 = freq1
        
    def initialize(self, t0):

        self.tlast = t0
        self.time = t0
        self.tnext = _INF

        self.x = self.x0
        self.q = self.x0
        self.q0 = self.x0
        self.dq = self.dqmin
        self.qhi = complex(self.q.real + self.dq, self.q.imag + self.dq)
        self.qlo = complex(self.q.real - self.dq, self.q.imag - self.dq)
        
        self.tout = [self.time]
        self.qout = [self.q0]
        self.nupd = [0]

        self.tzoh = [self.time]
        self.qzoh = [self.q0]

        self.updates = 0

    def update(self, time):

        self.time = time
        self.updates += 1
        self.triggered = False

        self.d = self.f(self.q)

        self.dint()
        self.quantize()

        self.ta()

        # trigger external update if quantized output changed:
        
        if self.q != self.q0:
            self.save()
            self.q0 = self.q
            self.trigger()
            self.update_dq()

    def dint(self):
        
        self.x = complex(self.x.real + self.d.real * (self.time - self.tlast), 
                         self.x.imag + self.d.imag * (self.time - self.tlast))

        self.tlast = self.time

    def step(self, time):

        self.time = time
        self.updates += 1
        self.d = self.f(self.x)
        self.dint()
        self.q = self.x
        self.save()
        self.q0 = self.q

    def quantize(self):
        
        interp = False
        change = False

        self.d0 = self.d

        if self.x.real >= self.qhi.real:

            self.q = complex(self.qhi.real, self.q.imag)
            self.qlo = complex(self.qlo.real + self.dq, self.qlo.imag) 
            change = True

        elif self.x.real <= self.qlo.real:

            self.q = complex(self.qlo.real, self.q.imag)
            self.qlo = complex(self.qlo.real - self.dq, self.qlo.imag) 
            change = True

        if self.x.imag >= self.qhi.imag:

            self.q = complex(self.q.real, self.qhi.imag)
            self.qlo = complex(self.qlo.real, self.qlo.imag + self.dq) 
            change = True

        elif self.x.imag <= self.qlo.imag:

            self.q = complex(self.q.real, self.qlo.imag)
            self.qlo = complex(self.qlo.real, self.qlo.imag - self.dq)
            change = True

        self.qhi = complex(self.qlo.real + 2.0 * self.dq, self.qlo.imag + 2.0 * self.dq)

        if change and self.implicit:  # we've ventured out of (qlo, qhi) bounds

            self.d = self.f(self.q)

            # if the derivative has changed signs, then we know 
            # we are in a potential oscillating situation, so
            # we will set the q such that the derivative ~= 0:

            if (self.d.real * self.d0.real) < 0 or (self.d.imag * self.d0.imag) < 0: 
                flo = self.f(self.qlo) 
                fhi = self.f(self.qhi)
                if flo != fhi:
                    a = (2.0 * self.dq) / (fhi - flo)
                    self.q = self.qhi - a * fhi
                    interp = True

        return interp

    def ta(self):

        if self.d.real > 0.0:
            treal = self.time + (self.qhi.real - self.x.real) / self.d.real
        elif self.d.real < 0.0:
            treal = self.time + (self.qlo.real - self.x.real) / self.d.real
        else:
            treal = _INF

        if self.d.imag > 0.0:
            timag = self.time + (self.qhi.imag - self.x.imag) / self.d.imag
        elif self.d.imag < 0.0:
            timag = self.time + (self.qlo.imag - self.x.imag) / self.d.imag
        else:
            timag = _INF

        self.tnext = min(treal, timag)
        self.tnext = max(self.tnext, self.tlast + self.dtmin)

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
            
        self.dq = min(self.dqmax, max(self.dqmin, abs(self.dqerr * abs(self.q)))) 
            
        self.qlo = self.q - complex(self.dq, self.dq) 

        self.qhi = self.q + complex(self.dq, self.dq) 


class System(object):

    def __init__(self, name="sys", dq=None, dqmin=None, dqmax=None, dqerr=None, dtmin=None,
                 dmax=None, print_time=False):
        
        self.name = name

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

        self.print_time = print_time

        self.devices = []
        self.atoms = []
        self.ss = None

        # simulation variables:
        self.tstop = 0.0  # end simulation time
        self.time = 0.0   # current simulation time
        self.iprint = 0   # for runtime updates

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

        setattr(self, device.name, device)

    def add_devices(self, *devices):

        for device in devices:
            self.add_device(device)

    def connect(self, from_port, *to_ports):

        for to_port in to_ports:
            
            if to_port.direction == PortDirection.INOUT:
                
                from_port.connected_ports.append(to_port)
                to_port.connected_ports.append(from_port)

                to_port.atom.broadcast_to.append(from_port.atom)
                from_port.atom.broadcast_to.append(to_port.atom)

                to_port.atom.receive_from[from_port.atom] = to_port.gain
                from_port.atom.receive_from[to_port.atom] = from_port.gain

    def initialize(self, t0=0.0, ss=False, dc=False):

        self.time = t0

        if dc:
            self.solve_dc()

        for atom in self.atoms:
            atom.initialize(t0)

    def initialize_ss(self, dt, dc=False, reset_state=True):

        if dc:
            self.solve_dc()

        self.build_ss(dt)

        self.ss.initialize(dt, reset_state=reset_state)

    def get_nodes_and_branches(self):

        nodes = []
        branches = []
        index = 1

        for device in self.devices:

            for atom in device.atoms:

                if (atom.lim_type == LimAtomType.GROUND_NODE):

                    device.atom.index = 0
                    nodes.append(device.atom)

                elif (atom.lim_type in (LimAtomType.LATENCY_NODE,
                                        LimAtomType.CONST_NODE)):
                    atom.index = index
                    index += 1
                    nodes.append(atom)

        for device in self.devices:

            for atom in device.atoms:

                if (atom.lim_type in (LimAtomType.LATENCY_BRANCH,
                                      LimAtomType.CONST_BRANCH)):

                    atom.index = index
                    index += 1
                    branches.append(atom)

        return nodes, branches

    def build_ss(self, dt):

        nodes, branches = self.get_nodes_and_branches()

        n = len(nodes) + len(branches)

        # dimension state space matrices:

        a = np.zeros((n, n))
        b = np.zeros((n, n))
        u = np.zeros((n, 1))
        x0 = np.zeros((n, 1))

        for node in nodes:

            ii = node.index

            x0[ii] = node.x0

            if ii == 0:
                continue

            a[ii, ii] = -node.b / node.a
            b[ii, ii] = 1.0 / node.a
            u[ii, 0] = node.c

            for branch in branches:

                k = branch.index
                i = branch.device.negative.connected_ports[0].atom.index
                j = branch.device.positive.connected_ports[0].atom.index

                if ii == i:
                    a[i, k] = -1.0 / node.a

                elif ii == j:
                    a[j, k] = 1.0 / node.a

        for branch in branches:

            k = branch.index

            x0[k] = branch.x0

            i = branch.device.negative.connected_ports[0].atom.index
            j = branch.device.positive.connected_ports[0].atom.index

            a[k, k] = -branch.b / branch.a
            a[k, i] = 1.0 / branch.a
            a[k, j] = -1.0 / branch.a

            b[k, k] = 1.0 / branch.a
            u[k, 0] = branch.c

        if not self.ss:

            # make a new state space model (skip ground row/col):

            self.ss = lti.StateSpace(a[1:, 1:], b[1:, 1:], u0=u[1:,:], x0=x0[1:,:])
        
        else:

            # update existing model (skip ground row/col):

            self.ss.a = a[1:, 1:]
            self.ss.b = b[1:, 1:]
            self.ss.u0 = u[1:,:]
            self.ss.x0 = x0[1:,:]

    def solve_dc(self, rmin=1e-6):

        """solves for the dc steady-state of this systems by building
        and solving the conductance matrix of the system:
        
        G*x = b,  x = (G^-1)*b

        If branch R is zero, a small ficticous resistance (rmin) is added.
        """

        nodes, branches = self.get_nodes_and_branches()

        n = len(nodes)
        k = len(branches)

        g = np.zeros((n, n))
        b = np.zeros((n, 1))

        states = np.zeros((n+k, 1))

        for node in nodes:

            i = node.index

            if i != 0:
                g[i, i] += node.b
                b[i] += node.c

        for branch in branches:

            i = branch.device.positive.connected_ports[0].atom.index
            j = branch.device.negative.connected_ports[0].atom.index

            if branch.b:
                rinv = 1.0 / branch.b
            else:
                rinv = 1.0 / rmin

            g[i, i] += rinv
            g[i, j] -= rinv
            g[j, i] -= rinv
            g[j, j] += rinv

            b[i] += branch.c * rinv
            b[j] -= branch.c * rinv

        states[1:n] = la.solve(g[1:,1:], b[1:])

        for node in nodes:
            node.x0 = states[node.index, 0]

        for branch in branches:

            k = branch.index
            i = branch.device.positive.connected_ports[0].atom.index
            j = branch.device.negative.connected_ports[0].atom.index

            if branch.b:
                rinv = 1.0 / branch.b
            else:
                rinv = 1.0 / rmin

            branch.x0 = (-states[i, 0] + states[j, 0] + branch.c) * rinv

    def run_to(self, tstop, fixed_dt=None, verbose=False):

        self.tstop = tstop

        print("Simulation started...")

        if fixed_dt:
            dt = fixed_dt
            i = 0
            while(self.time < self.tstop):
                for atom in self.atoms:
                    atom.step(self.time)
                    atom.save()
                if i >= 100: 
                    print("t = {0:5.2f} s".format(self.time))
                    i = 0
                i += 1
                self.time += dt
            return

        #self.print_percentage(header=True)

        for i in range(1):
            for atom in self.atoms:
                atom.update(self.time)
                atom.save()

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

        #self.print_percentage()
        #self.print_percentage(footer=True)

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

    def run_ss_to(self, tstop):

        self.ss.run_to(tstop)

    def finalize_ss(self):

        # concatenate the data for number of "updates" (time steps):

        self.ss_tupd = [self.ss.t[0], self.ss.t[-1]]
        self.ss_nupd = [0, len(self.ss.t)]

    def save_data(self):

        # stores the tout, qout, and nupd arrays to secondary arrays:

        for atom in self.atoms:

            atom.tout2 = atom.tout[:]
            atom.qout2 = atom.qout[:]
            atom.nupd2 = atom.nupd[:]

    def print_percentage(self, header=False, footer=False):
        
        if header:
            print("\n\nPercentage Complete:") #\n+" + "-"*98 + "+")
            return

        if footer:
            print("\nDone.\n\n")
            return

        i = int(self.time / self.tstop * 100.0)

        while self.iprint < i:
            print(str(self.iprint) + "%")
            self.iprint += 1

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
            atoms = [device.atom for device in self.devices]

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
                ax1.plot(self.ss.t[1:], self.ss.y[atom.index-1][1:], 'c--', label="ss_x")

            if plot_qss_updates or plot_ss_updates:
                ax2 = ax1.twinx()
                ax2.set_ylabel('total updates', color='r')
                
            if plot_qss_updates:
                ax2.plot(atom.tout, atom.nupd, 'r-', label="qss_upds")

            if plot_ss_updates:
                ax2.plot(self.ss_tupd, self.ss_nupd, 'm--', label="ss_upds")

            if ax1 and legend:
                ax1.legend(loc="upper left")

            if ax2 and legend:
                ax2.legend(loc="upper right")

            plt.xlabel("t (s)")
 
            j += 1

        plt.tight_layout()
        plt.show()

    def plot_groups(self, *groups, plot_ss=False, legend=True):

        #mpl.style.use('seaborn')

        if len(groups) > 1:
            c, j = 2, 1
            r = floor(len(groups)/2) + 1

        for group in groups:

            if len(groups) > 1:
                plt.subplot(r, c, j)

            for i, atom in enumerate(group):

                lbl = "{} qss ({})".format(atom.full_name(), atom.units)

                color = 'C{}'.format(i)

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

                    plt.plot(self.ss.t[1:], self.ss.y[atom.index-1][1:], 
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

    """Collection of Atoms, Ports and Viewables that comprise a device.

    .--------------.
    |              |-------o inout_port_1
    |              |-------o inout_port_2
    |  Device      |  ...
    |              |-------o port_N  
    | (atoms)      |  ...
    |              |-------< in_port_1
    | (viewables)  |-------< in_port_2
    |              |  ...
    |              |-------< in_port_N 
    |              |  ...
    |              |-------> out_port_1
    |              |-------> out_port_2
    |              |  ...
    |              |-------> out_port_N 
    '--------------'

    """

    def __init__(self, name):

        self.name = name
        self.atoms = []
        self.ports = []
        self.viewables = []

    def add_atom(self, atom):
        
        self.atoms.append(atom)
        atom.device = self
        setattr(self, atom.name, atom)

    def add_atoms(self, *atoms):
        
        for atom in atoms:
            self.add_atom(atom)

    def add_port(self, port):
        
        port.device = self
        self.ports.append(port)

    def add_ports(self, *ports):
        
        for port in ports:
            self.add_port(port)

    def add_viewable(self, viewable):
        
        viewable.device = self
        self.viewables.append(viewable)

    def add_viewables(self, *viewables):
        
        for viewable in viewables:
            self.add_viewable(viewable)

    def __repr__(self):

        return self.name

    def __str__(self):

        return __repr__(self)


class Port(object):

    """Mediates the connections and data transfer between atoms.
    """

    def __init__(self, name, atom, direction=PortDirection.INOUT, gain=1.0):

        self.name = name
        self.direction = direction
        self.atom = atom
        self.gain = gain

        self.connected_ports = []
        self.device = None

    def __repr__(self):

        return self.device.name + "." + self.name

    def __str__(self):

        return __repr__(self)


class Viewable(object):

    def __init__(self, name, func, units=""):

        self.name = name
        self.func = func
        self.units = units
        self.device = None

    def __repr__(self):

        return self.device.name + "." + self.name

    def __str__(self):

        return __repr__(self)


# ============================ Basic Devices ===================================

class LimGround(Device):

    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.atom = Atom(name="voltage", is_latent=False,
                 source_type=SourceType.CONSTANT, lim_type=LimAtomType.GROUND_NODE,
                 x0=0.0, units="V")

        self.positive = Port("positive", self.atom, PortDirection.INOUT)

        self.add_atom(self.atom)
        self.add_port(self.positive)


class LimLatencyNode(Device):

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

    def __init__(self, name, c, g=0.0, h=0.0, v0=0.0,
                 source_type=SourceType.NONE, v1=0.0, v2=0.0, va=0.0, freq=0.0,
                 phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None):

        Device.__init__(self, name)

        self.atom = Atom(name="voltage", a=c, b=g, c=h, is_latent=True, derivative=self.derivative,
                 source_type=source_type, lim_type=LimAtomType.LATENCY_NODE,
                 x0=v0, x1=v1, x2=v2, xa=va, freq=freq, phi=phi, duty=duty,
                 t1=t1, t2=t2, units="V", dq=dq)

        self.positive = Port("positive", self.atom, PortDirection.INOUT, gain=-1.0)

        self.add_atom(self.atom)
        self.add_port(self.positive)

    def derivative(self):

        return 1.0 / self.atom.a * (-self.atom.q * self.atom.b + self.current()
                                   + self.atom.c)
    def current(self):

        return sum([port.atom.q * port.gain for port in self.positive.connected_ports])


class LimLatencyBranch(Device):

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

    def __init__(self, name, l, r=0.0, e=0.0, i0=0.0,
                 source_type=SourceType.NONE, i1=0.0, i2=0.0, ia=0.0, freq=0.0,
                 phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None):

        Device.__init__(self, name)

        self.atom = Atom(name="current", a=l, b=r, c=e, is_latent=True, derivative=self.derivative,
                 source_type=source_type, lim_type=LimAtomType.LATENCY_BRANCH,
                 x0=i0, x1=i1, x2=i2, xa=ia, freq=freq, phi=phi, duty=duty,
                 t1=t1, t2=t2, units="A", dq=dq)

        self.positive = Port("positive", self.atom, PortDirection.INOUT, gain=1.0)
        self.negative = Port("negative", self.atom, PortDirection.INOUT, gain=-1.0)

        self.add_atom(self.atom)
        self.add_ports(self.positive, self.negative)

    def derivative(self):

        return 1.0 / self.atom.a * (-self.atom.q * self.atom.b + self.voltage()
                                    + self.atom.c)
    
    def voltage(self):

        vi = self.negative.connected_ports[0].atom.q
        vj = self.positive.connected_ports[0].atom.q

        return vi - vj

# =============================== Tests ========================================


if __name__ == "__main__":

     sys = System(dq=1e-2)

     ground = LimGround("ground")
     node1 = LimLatencyNode("node1", 1.0, 1.0, 0.0)
     node2 = LimLatencyNode("node2", 1.0, 1.0, 0.0)
     node3 = LimLatencyNode("node3", 1.0, 1.0, 0.0)

     branch1 = LimLatencyBranch("branch1", 1.0, 0.1, 10.0)
     branch2 = LimLatencyBranch("branch2", 1.0, 0.1, 0.0)
     branch3 = LimLatencyBranch("branch2", 1.0, 0.1, 0.0)

     sys.add_devices(ground, node1, branch1, node2, node3, branch2, branch3)

     sys.connect(branch1.negative, ground.positive)
     sys.connect(branch1.positive, node1.positive)

     sys.connect(branch2.negative, node1.positive)
     sys.connect(branch2.positive, node2.positive)

     sys.connect(branch3.negative, node2.positive)
     sys.connect(branch3.positive, node3.positive)

     tstop = 60.0
     dc = True

     sys.initialize_ss(1e-3, dc=dc)
     sys.initialize(dc=dc)

     sys.run_ss_to(tstop*0.2)
     sys.run_to(tstop*0.2)

     node2.atom.b = 1000.0

     sys.initialize_ss(1e-3, reset_state=False)

     sys.run_ss_to(tstop*0.2+0.1)
     sys.run_to(tstop*0.2+0.1)

     node2.atom.b = 1.0

     sys.initialize_ss(1e-3, reset_state=False)

     sys.run_ss_to(tstop)
     sys.run_to(tstop)

     sys.finalize_ss()

     sys.plot_groups((node1.voltage, node2.voltage, node3.voltage), plot_ss=True)

     sys.plot_groups((branch1.current, branch2.current, branch3.current), plot_ss=True)