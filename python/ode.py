"""Test nl ode framework for qdevs-lim system.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import inspect


sys = None

def fode(x, t, sys):

    dx_dt = [0.0]*sys.n

    for atom in sys.state_atoms:

        dx_dt[atom.index] = atom.fode(x, t)

    return dx_dt


def build_source_function(atom):

    def u(t):

       pass

    return u


class System(object):

    def __init__(self):

        self.state_atoms = []
        self.source_atoms = []
        self.n = 0
        self.m = 0

    def add_states(self, *atoms):

        for atom in atoms:
            atom.index = self.n
            self.state_atoms.append(atom)
            self.n += 1

    def add_sources(self, *atoms):

        for atom in atoms:
            atom.index = self.m
            self.source_atoms.append(atom)
            self.m += 1

    def get_f(self):

        return f

    def get_y0(self):

        y0 = [0.0]*self.n
        for atom in self.state_atoms:
            y0[atom.index] = atom.x0
        return y0

    def run_to(self, tstop, dt):

        self.t = np.arange(0, tstop, dt)
        y0 = self.get_y0()
        self.y = odeint(fode, y0, self.t, args=(sys,))


class Atom(object):
    
    def __init__(self, x0=0.0):

        self.x0 = x0

    def fode(self, x, t):

        i = self.index
        js = [connected.index for connected in self.connected]
        xsum = sum([x[j] * coef for j, coef in zip(js, self.coefs)])

        xsrc = 0.0
        if self.source:
            xsrc = self.source.u(t)

        return -(self.b/self.a)*x[i] + (1/self.a)*xsrc - (1/self.a)*xsum


class SourceAtom(Atom):

    def __init__(self, x0):

        Atom.__init__(self, x0)

        self.x = 0.0
        self.index = -1

    def u(self, t):
        pass


class StateAtom(Atom):

    def __init__(self, a, b, x0):

        Atom.__init__(self, x0)

        self.a = a
        self.b = b
        self.index = -1
        self.source = None
        self.connected = []
        self.coefs = []

    def connect(self, other, coef=1.0):

        self.connected.append(other)
        self.coefs.append(coef)


class VoltageSource(SourceAtom):

    def __init__(self, v0):

        SourceAtom.__init__(self, v0)

        self.v0 = v0

    def u(self, t):

        return self.v0


class CurrentSource(SourceAtom):

    def __init__(self, i0):
        
        SourceAtom.__init__(self, i0)

        self.i0 = i0

    def u(self, t):

        return self.i0

class Node(StateAtom):

    def __init__(self, c, g=0.0, v0=0.0):

        StateAtom.__init__(self, g, c, v0)


class Branch(StateAtom):

    def __init__(self, l, r=0.0, i0=0.0):

        StateAtom.__init__(self, l, r, i0)


sys = System()

esource = VoltageSource(1.0)
hsource = CurrentSource(1.0)
branch = Branch(1.0, 1.0)
node = Node(1.0, 1.0)

sys.add_sources(esource, hsource)
sys.add_states(branch, node)

branch.source = esource
node.source = hsource

branch.connect(node, 1.0)
node.connect(branch, -1.0)

sys.run_to(10.0, 1e-2)

plt.plot(sys.t, sys.y[:,0])
plt.plot(sys.t, sys.y[:,1])

plt.show()

"""
# initialize system's parameters
R0 = 2 # Ohm
L = 400*10e-3 # Henri
k = 100 # Nm/A
r = 0.5 # m
m = 500 # kg
g = 9.81 # m/s^2

Tr = 3

def Rnl(t):
    return R0 + 8*(1 - np.exp(-t/Tr))

t = np.arange(0, 60, 1e-2)

# this one could be any other function of time
def u(t):
    return 0

def f(x, t):
    dx_dt = [0, 0, 0]
    dx_dt[0] = x[1]
    dx_dt[1] = k/(r*m)*x[2]-g
    dx_dt[2] = -Rnl(t)/L*x[2] - k/r*x[1] + 1/L*u(t)
    return dx_dt

# y0 is our initial state
s = odeint(f, y0=[0, 0, 0], t=t)

# s is a Nx3 matrix with N timesteps

plt.plot(t, s[:,0])
plt.plot(t, s[:,1])
plt.plot(t, s[:,2])

plt.show()
"""
