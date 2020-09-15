""" Quantized DEVS LIM (QDL) Framework
"""

from math import pi, sin, cos, sqrt, floor
import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from collections import OrderedDict as odict

import qss
from qss import SourceType

import lti


# =========================== Base Atom Definitions ============================


class Atom(object):

    """Generic LIM Atom. Base LIM object of the QDL framework.
    """

    def __init__(self, name, source_type=SourceType.NONE, x0=0.0, x1=0.0,
                 x2=0.0, xa=0.0, freq=0.0, phi=0.0, func=None, duty=0.0,
                 t1=0.0, t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units="",
                 srcfunc=None, srcdt=None, output_scale=1.0):

        self.name = name

        # source params:
        self.source_type = source_type
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.xa = xa
        self.freq = freq
        self.phi = phi
        self.func = func
        self.duty = duty
        self.t1 = t1
        self.t2 = t2
        self.srcfunc = srcfunc
        self.srcdt = srcdt

        # quantization params:
        self.dq = dq
        self.dqmin = dqmin
        self.dqmax = dqmax
        self.dqerr = dqerr
        self.dtmin = dtmin
        self.dmax = dmax

        # additional params:
        self.units = units
        self.output_scale = output_scale

        # QSS model atom (interface with QSS solver):
        self.qss_atom = None  

        # Atoms to which this qss model needs to broadcast/recieve updates:
        self.connected_atoms = []

    def x(self):

        return self.qss_atom.x

    def q(self):

        return self.qss_atom.q

    def connect(self, atom):

        # cross connect:
        self.connected_atoms.append(atom)
        atom.connected_atoms.append(self)

    def derivative(self):

        # should be implemented by latent atoms:
        raise NotImplementedError

    def __repr__(self):

        return self.name

    def __str__(self):

        return self.name


class LimBranch(Atom):

    """Generic LIM Branch.

                  .-----------.    
    nodei  o------| LimBranch |------o  nodej
                  '-----------'
                      --->
                       i

                      OR

                  +    v    -    (for ideal voltage source mode)

    """

    def __init__(self, name, nodei, nodej, source_type=SourceType.NONE, i0=0.0,
                 i1=0.0, i2=0.0, ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units=""):

        Atom.__init__(self, name=name, source_type=SourceType.NONE, x0=i0,
                      x1=i1, x2=i2, xa=ia, freq=freq, phi=phi, duty=duty,
                      t1=t1, t2=t2, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        # LimBranch specific attributes:

        self.nodei = nodei
        self.nodej = nodej


class LimNode(Atom):

    """Generic LIM Node.

              o
              |      
         .---------.   +
         |         |             ^
         | LimNode |   v    OR   | i   (for ideal current source mode)
         |         |             |
         '---------'   -
              |
             _|_
              -
    """

    def __init__(self, name, source_type=SourceType.NONE, v0=0.0,
                 v1=0.0, v2=0.0, va=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units=""):

        Atom.__init__(self, name=name, source_type=SourceType.NONE, x0=v0,
                      x1=v1, x2=v2, xa=va, freq=freq, phi=phi, duty=duty,
                      t1=t1, t2=t2, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.connected_branches = []

    def connect_branch(self, branch, polarity=1.0):

        branch.connect(self)

        self.connected_branches.append((branch, polarity))


class LimPhasorNode(Atom):

    """Generic LIM Phasor Node.

              o
              |                          (for ideal current source mode):
      .---------------.      +           
      |               |                        ^
      | LimPhasorNode |   V*exp(j*delta)  OR   | I*exp(j*delta)   
      |               |                        |
      '---------------'      -
              |
             _|_
              -
    """

    def __init__(self, name, source_type=SourceType.NONE, v0=0.0,
                 v1=0.0, v2=0.0, va=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units=""):

        Atom.__init__(self, name=name, source_type=SourceType.NONE, x0=v0,
                      x1=v1, x2=v2, xa=va, freq=freq, phi=phi, duty=duty,
                      t1=t1, t2=t2, dq=dq, dqmin=dqmin, dqmax=dqmax,
                      dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

        self.connected_branches = []

    def connect_branch(self, branch, polarity=1.0):

        branch.connect(self)

        self.connected_branches.append((branch, polarity))


class LimLatencyBranch(LimBranch):

    """Generic LIM Lantency Branch with R, L, E, T and Z components.

                                 t[k]*v[k]  z[p,q]*i[p,q]
              e
    nodei    ,-.     r      l       ,^.          ,^.       nodej
      o-----(- +)---VVV---UUU-----< - + >------< - + >--------o
             `-'                    `.'          `.'  
                              ---->
                                i
    """

    def __init__(self, name, nodei, nodej, l, r=0.0, e=0.0, 
                 source_type=SourceType.NONE, i0=0.0, i1=0.0, i2=0.0, ia=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None, units="A"):

        LimBranch.__init__(self, name, nodei, nodej,
                           source_type=SourceType.NONE, i0=i0, i1=i1, i2=i2,
                           ia=ia, freq=freq, phi=phi, duty=duty, t1=t1, t2=t2,
                           dq=dq, dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                           dtmin=dtmin, dmax=dmax, units=units)
        
        # latency branch specific attributes:
        self.l = l
        self.r = r
        self.e = e

        # controlled source connection arrays:
        self.tnodes = []
        self.zbranches = [] 

        # cache inverted l:
        self.linv = None
        if self.l != 0.0:
            self.linv = 1.0 / l

        # create electrical connection to nodes:
        self.nodei.atom.connect_branch(self, 1.0)
        self.nodej.atom.connect_branch(self, -1.0)

    def add_tnode(self, node, gain):
        
        self.tnodes.append((node, gain))

    def add_zbranch(self, branch, gain):

        self.add_zbranch.append((branch, gain))

    def derivative(self):

        # get branch voltage drop:
        vi = self.nodei.atom.q()
        vj = self.nodej.atom.q()

        return self.linv * (self.e  - self.q() * self.r + (vi - vj))


class LimLatencyNode(LimNode):

    """Generic LIM Lantency Node with G, C, H, B and S components.
                             
                    \       
              i[i,2] ^    ... 
                      \   
              i[i,1]   \     i[i,k]
             ----<------o------>----
                        |
        .--------.------+-------------.----------------.
        |        |      |             |                |     +
       ,-.      <.     _|_   b[k]*   ,^.    s[p,q]*   ,^.  
    h ( ^ )   g <.   c ___    v[k] <  ^  >   i[p,q] <  ^  >  v
       `-'      <.      |            `.'              `.'  
        |        |      |             |                |     -
        '--------'------+-------------'----------------'
                       _|_                      
                        -                       
    """

    def __init__(self, name, c, g=0.0, h=0.0, 
                 source_type=SourceType.NONE, v0=0.0, v1=0.0, v2=0.0, va=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None, units="V"):

        LimNode.__init__(self, name=name, source_type=SourceType.NONE, v0=v0,
                         v1=v1, v2=v2, va=va, freq=freq, phi=phi, duty=duty,
                         t1=t1, t2=t2, dq=dq, dqmin=dqmin, dqmax=dqmax,
                         dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)
        
        # latency node specific attributes:
        self.c = c
        self.g = g
        self.h = h

        # controlled source connection arrays:
        self.bnodes = []
        self.sbranches = [] 

        # cache inverted c:
        self.cinv = None
        if self.c != 0.0:
            self.cinv = 1.0 / c

    def add_bnode(self, node, gain):
        
        self.bnodes.append((node, gain))

    def add_sbranch(self, branch, gain):

        self.add_sbranch.append((branch, gain))

    def derivative(self):

        # sum current contributions from all attached branches:

        isum = sum([branch.q() * polarity for (branch, polarity)
                    in self.connected_branches])

        return self.cinv * (self.h - self.q() * self.g - isum)


class LimSourceBranch(LimBranch):

    """Generic LIM Ideal Current Source Branch

             i(t)
    nodei    ,-.   nodej
      o-----(-->)-----o
             `-'  

    """

    def __init__(self, name, nodei, nodej, source_type=SourceType.NONE, i0=0.0,
                 i1=0.0, i2=0.0, ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units="A"):
        
        LimBranch.__init__(self, name, nodei, nodej,
                          source_type=SourceType.NONE, v0=v0, v1=v1, v2=v2,
                          va=va, freq=freq, phi=phi, duty=duty, t1=t1, t2=t2,
                          dq=dq, dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                          dtmin=dtmin, dmax=dmax, units=units)

        # create electrical connection to nodes:
        self.nodei.atom.connect_branch(self, 1.0)
        self.nodej.atom.connect_branch(self, -1.0)

    def derivative(self):

        return 0.0


class LimSourceNode(LimNode):

    """Generic LIM Ideal Voltage Source Node

            o
            |      
         + ,-.  
          (   )  v(t)
         - `-'  
            |
           _|_
            -
    """

    def __init__(self, name, source_type=SourceType.NONE, v0=0.0,
                 v1=0.0, v2=0.0, va=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None, units="V"):

        LimNode.__init__(self, name, source_type=SourceType.NONE, v0=v0,
                         v1=v1, v2=v2, va=va, freq=freq, phi=phi, duty=duty,
                         t1=t1, t2=t2, dq=dq, dqmin=dqmin, dqmax=dqmax,
                         dqerr=dqerr, dtmin=dtmin, dmax=dmax, units=units)

    def derivative(self):

        return 0.0


# ============================= Base Device ====================================


class LimDevice(object):

    """Collection of Lim Atoms that comprise a device.

    .----------.
    |          |
    |   LIM    |
    |  Device  |
    |          |
    | (atoms)  |
    |          |
    '----------'

    """

    def __init__(self, name):

        self.name = name
        self.atoms = []

    def add_atom(self, atom):
        
        self.atoms.append(atom)

    def add_atoms(self, *atoms):
        
        for atom in atoms:
            self.atoms.append(atom)


# ==============================================================================
#                              Device Models 
# ==============================================================================


class GenericBranch(LimDevice):
    
    """A device wrapper for a Generic LIM latency branch.
    """

    def __init__(self, name, npos, nneg, l, r=0.0, e=0.0, 
                 source_type=SourceType.NONE, v0=0.0, v1=0.0, v2=0.0, va=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        self.npos = npos
        self.nneg = nneg

        atom_name = name + ".current"

        self.atom = LimLatencyBranch(name=atom_name, nodei=nneg, nodej=npos,
                                     l=l, r=r, e=e,
                                     source_type=source_type, v0=v0, v1=v1,
                                     v2=v2, va=va, freq=freq, phi=phi,
                                     duty=duty, t1=t1, t2=t2, dq=dq,
                                     dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                     dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)

    def add_vcvs(self, node, gain):

        self.atom.add_tnode(node, gain)

    def add_ccvs(self, branch, gain):

        self.atom.add_zbranch(branch, gain)


class GenericNode(LimDevice):
    
    """A device wrapper for a Generic LIM latency node.
    """

    def __init__(self, name, g, c, h, source_type=SourceType.NONE, i0=0.0,
                 i1=0.0, i2=0.0, ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".voltage"

        self.atom = LimLatencyNode(name=atom_name, g=g, c=c, h=h, 
                 source_type=source_type, i0=i0, i1=i1, i2=i2, ia=ia,
                 freq=freq, phi=phi, duty=duty, t1=t1, t2=t2, dq=dq,
                 dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                 dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)

    def add_cccs(self, branch, gain):

        self.atom.add_sbranch(branch, gain)

    def add_vccs(self, node, gain):

        self.atom.add_bnode(node, gain)

    def add_capacitance(self, c):

        self.atom.c += c

        if self.atom.c:
            self.atom.cinv = 1.0 / self.atom.c

    def add_conductance(self, g):

        self.atom.g += g


class Node(LimDevice):
    
    """A connection node for other devices to connect to.
    """

    def __init__(self, name, v0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".voltage"

        self.atom = LimLatencyNode(name=atom_name, c=0.0, v0=v0, dq=dq,
                 dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                 dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)

    def add_capacitance(self, c):

        self.atom.c += c

        if self.atom.c:
            self.atom.cinv = 1.0 / self.atom.c

    def add_conductance(self, g):

        self.atom.g += g


class RLLoad(LimDevice):
    
    """RL to ground model.

                  r     l
    npos  o------VVV---UUU------o nneg
                   ---->     
                     i       
                           
    """

    def __init__(self, name, npos, nneg, r, l, i0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = LimLatencyBranch(name=atom_name, nodei=npos, nodej=nneg,
                                     l=l, r=r, i0=i0, dq=dq,
                 dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                 dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)


# ================================= Ground =====================================


class Ground(LimDevice):

    """Ground device, which is just a LIM source node with voltage == 0.
    """

    def __init__(self, name="gnd", v0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".voltage"

        self.atom = LimSourceNode(name=atom_name,
                                  source_type=SourceType.CONSTANT, v0=v0, dq=dq,
                 dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                 dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)


# ============================= Voltage Sources ================================


class IdealVoltageSource(LimDevice):

    """Generic ideal (loseless) voltage source device.
    """

    def __init__(self, name, source_type, i0=0.0,
                 i1=0.0, i2=0.0, ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = LimSourceNode(name=atom_name, source_type=source_type,
                                  i0=i0, i1=i1, i2=i2, va=va, freq=freq,
                                  phi=phi, duty=duty, t1=t1, t2=t2,
                                  dq=dq, dqmin=dqmin, dqmax=dqmax,
                                  dqerr=dqerr, dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)


class VoltageSource(LimDevice):

    """Generic lossy voltage source device.
    """

    def __init__(self, name, node_pos, node_neg, e, r, l,
                 source_type=SourceType.CONSTANT, i0=0.0, i1=0.0, i2=0.0,
                 ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = LimLatencyBranch(name=atom_name, nodei=node_neg,
                                     nodej=node_pos, e=e, r=r, l=l,
                                     source_type=source_type, i0=i0, i1=i1,
                                     i2=i2, ia=ia, freq=freq, phi=phi,
                                     duty=duty, t1=t1, t2=t2, dq=dq,
                                     dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                     dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)


class ConstantVoltageSource(VoltageSource):

    """Constant lossy voltage source device.
    """

    def __init__(self, name, node_pos, node_neg, e, r, l, i0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        VoltageSource.__init__(self, name=name, node_pos=node_pos,
                               node_neg=node_neg, e=e, r=r, l=l,
                               source_type=SourceType.CONSTANT, i0=i0, dq=dq,
                               dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                               dtmin=dtmin, dmax=dmax)


# ============================= Voltage Sources ================================


class Cable(LimDevice):

    """Cable with lumped components (PI Model)

                          r      l           
    node_pos  o----+-----UUU----VVV-----+----o  node_neg
                   |                    |   
                  _|_                  _|_  
              c/2 ___              c/2 ___  
                   |                    |   
                  _|_                  _|_  
                   -                    -   

    """

    def __init__(self, name, node_pos, node_neg, r, l, c, g=0.0, i0=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.branch = LimLatencyBranch(name=atom_name, nodei=node_pos,
                                       nodej=node_neg, l=l, r=r, i0=i0, dq=dq,
                 dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                 dtmin=dtmin, dmax=dmax)

        node_pos.add_capacitance(c*0.5)
        node_pos.add_conductance(g*0.5)

        node_neg.add_capacitance(c*0.5)
        node_neg.add_conductance(g*0.5)

        self.add_atoms(self.branch)

# ============================== QDL System ====================================


class System(object):

    """QDL System model.
    """

    def __init__(self, name="sys", dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):
        
        self.name = name
        
        self.dq = qss.DEF_DQ
        if dq:
            self.dq = dq

        self.dqmin = qss.DEF_DQMIN
        if dqmin:
            self.dqmin = dqmin
        elif dq:
            self.dqmin = dq

        self.dqmax = qss.DEF_DQMAX
        if dqmax:
            self.dqmax = dqmax
        elif dq:
            self.dqmax = dq

        self.dqerr = qss.DEF_DQERR
        if dqerr:
            self.dqerr = dqerr

        self.dtmin = qss.DEF_DTMIN
        if dtmin:
            self.dtmin = dtmin

        self.dmax = qss.DEF_DMAX
        if dmax:
            self.dmax = dmax

        self.devices = []

        self.qss = None
        self.ss = None

    def add_device(self, device):

        self.devices.append(device)

    def add_devices(self, *devices):

        for device in devices:
            self.devices.append(device)

    def initialize(self, dc=True):

        if dc:
            self.solve_dc()

        self.build_qss()
        self.qss.initialize()

    def initialize_ss(self, dt, reset_state=True):

        self.build_ss(dt)
        self.ss.initialize(dt, reset_state=reset_state)

    def get_nodes_and_branches(self):
        
        nodes = []
        branches = []
        index = 1

        for device in self.devices:

            if (isinstance(device, Ground)):
                device.atom.index = 0
                nodes.append(device.atom)
                continue

            for atom in device.atoms:

                if (isinstance(atom, LimNode)):
                    atom.index = index
                    index += 1
                    nodes.append(atom)

        for device in self.devices:

            for atom in device.atoms:

                if (isinstance(atom, LimBranch)):
                    atom.index = index
                    index += 1
                    branches.append(atom)

        return nodes, branches

    def build_ss(self, dt):

        nodes, branches = self.get_nodes_and_branches()

        n = len(nodes)
        m = len(branches)

        a = np.zeros((n+m, n+m))
        b = np.zeros((n+m, n+m))
        u = np.zeros((n+m, 1))
        x0 = np.zeros((n+m, 1))

        for node in nodes:

            ii = node.index

            x0[ii] = node.x0

            if ii == 0:
                continue

            a[ii, ii] = -node.g / node.c
            b[ii, ii] = 1.0 / node.c
            u[ii, 0] = node.h

            for branch in branches:

                k = branch.index
                i = branch.nodei.atom.index
                j = branch.nodej.atom.index

                if ii == i:
                    a[i, k] = -1.0 / node.c

                elif ii == j:
                    a[j, k] = 1.0 / node.c

        for branch in branches:

            k = branch.index

            x0[k] = branch.x0

            i = branch.nodei.atom.index
            j = branch.nodej.atom.index

            a[k, k] = -branch.r / branch.l
            a[k, i] = 1.0 / branch.l
            a[k, j] = -1.0 / branch.l

            b[k, k] = 1.0 / branch.l
            u[k, 0] = branch.e

        if not self.ss:
            self.ss = lti.StateSpace(a[1:, 1:], b[1:, 1:], u0=u[1:,:], x0=x0[1:,:])
        else:
            self.ss.a = a[1:, 1:]
            self.ss.b = b[1:, 1:]
            self.ss.u0 = u[1:,:]
            self.ss.x0 = x0[1:,:]
        
    def build_qss(self):

        self.qss = qss.System(self.name, dqmin=self.dqmin, dqmax=self.dqmax,
                              dq=self.dqerr, print_time=True)

        for device in self.devices:

            # transfer relavent atom attributes to qss atoms:

            for atom in device.atoms:

                qss_atom = qss.LiqssAtom(name=atom.name,
                                    source_type=atom.source_type, x0=atom.x0,
                                    x1=atom.x1, x2=atom.x2, xa=atom.xa,
                                    freq=atom.freq, phi=atom.phi,
                                    derivative=atom.derivative, duty=atom.duty,
                                    t1=atom.t1, t2=atom.t2, dq=atom.dq,
                                    dqmin=atom.dqmin, dqmax=atom.dqmax,
                                    dqerr=atom.dqerr, dtmin=atom.dtmin,
                                    dmax=atom.dmax, units=atom.units,
                                    srcfunc=atom.srcfunc, srcdt=atom.srcdt,
                                    output_scale=atom.output_scale,
                                    parent=atom)

                atom.qss_atom = qss_atom
                self.qss.add_atom(qss_atom)

        # now make connections:

        for device in self.devices:

            for atom in device.atoms:

                for connected in atom.connected_atoms:

                    atom.qss_atom.recieve_from[connected.qss_atom] = 0.0
                    connected.qss_atom.broadcast_to.append(atom.qss_atom)

    def run_to(self, tstop, fixed_dt=None, verbose=True):

        self.qss.run_to(tstop, fixed_dt=fixed_dt, verbose=verbose)

    def run_ss_to(self, tstop):

        self.ss.run_to(tstop)

    def finalize_ss(self):

        self.ss_tupd = [self.ss.t[0], self.ss.t[-1]]
        self.ss_nupd = [0, len(self.ss.t)]

    def solve_dc(self):

        nodes, branches = self.get_nodes_and_branches()

        n = len(nodes)
        k = len(branches)

        g = np.zeros((n, n))
        b = np.zeros((n, 1))

        states = np.zeros((n+k, 1))

        for node in nodes:

            i = node.index

            if i != 0:
                g[i, i] += node.g
                b[i] += node.h

        for branch in branches:

            i = branch.nodei.atom.index
            j = branch.nodej.atom.index

            if branch.r:
                rinv = 1.0 / branch.r
            else:
                rinv = 1e6

            g[i, i] += rinv
            g[i, j] -= rinv
            g[j, i] -= rinv
            g[j, j] += rinv

            b[i] -= branch.e * rinv
            b[j] += branch.e * rinv

        states[1:n] = la.solve(g[1:,1:], b[1:])

        for node in nodes:
            node.x0 = states[node.index, 0]

        for branch in branches:

            k = branch.index
            i = branch.nodei.atom.index
            j = branch.nodej.atom.index

            if branch.r:
                rinv = 1.0 / branch.r
            else:
                rinv = 1e6

            branch.x0 = (states[i, 0] - states[j, 0] + branch.e) * rinv

    def plot(self, *devices, plot_qss=True, plot_ss=True, plot_qss_updates=False, plot_ss_updates=False):

        atoms = []

        for device in devices:
            for atom in device.atoms:
                atoms.append(atom)

        c, j = 2, 1
        r = floor(len(atoms)/2) + 1

        for atom in atoms:
    
            if plot_qss or plot_ss:
                plt.subplot(r, c, j)
                ax1 = plt.gca()
                ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')
                ax1.grid()
                ax1.legend()

            if plot_qss:
                ax1.plot(atom.qss_atom.tzoh, atom.qss_atom.qzoh, 'b-', label="qss_q")

            if plot_ss:
                ax1.plot(self.ss.t[1:], self.ss.y[atom.index-1][1:], 'c--', label="ss_x")

            if plot_qss_updates or plot_ss_updates:
                ax2 = ax1.twinx()
                ax2.set_ylabel('total updates', color='r')
                

            if plot_qss_updates:
                ax2.plot(atom.qss_atom.tout, atom.qss_atom.nupd, 'r-', label="qss_upds")

            if plot_ss_updates:
                ax2.plot(self.ss_tupd, self.ss_nupd, 'm--', label="ss_upds")

            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.xlabel("t (s)")
 
            j += 1

        plt.tight_layout()
        plt.show()


# ================================= Tests ======================================


def test1():

    """         
          n1      .-------.     n2
          o-------| Cable |------o
       +  |       '-------'      |
         ,-.                  .------.
    Vg  (   )                 | Load |
         `-'                  '------'
          |                      |
     gnd  o----------------------'
         _|_
          -

    """

    sys = System(dq=1e-3)

    vn10    = 0.0
    vn20    = 0.0
    icable0 = 0.0
    ivg0    = 0.0
    iload0  = 0.0

    gnd = Ground()
    n1 = Node("n1", v0=vn10)
    n2 = Node("n2", v0=vn20)
    vg = ConstantVoltageSource("vg", n1, gnd, e=1.0, r=0.01, l=0.001, i0=ivg0)
    cable = Cable("cable", n1, n2, r=0.01, l=0.001, c=0.01, i0=icable0)
    load = RLLoad("load", n2, gnd, r=2.0, l=0.001, i0=iload0)

    sys.add_devices(gnd, n1, n2, vg, cable, load)

    ssdt = 1e-4

    sys.initialize()
    sys.initialize_ss(ssdt)

    tstop = 2.0
    sys.run_to(tstop, verbose=True)
    sys.run_ss_to(tstop)

    load.atom.r = 1.0
    sys.initialize_ss(ssdt, reset_state=False)

    tstop = 4.0
    sys.run_to(tstop, verbose=True)
    sys.run_ss_to(tstop)

    sys.finalize_ss()

    sys.plot(n1, n2, vg, cable, load, plot_qss_updates=True, plot_ss_updates=True)


def test2():

    sys = qss.System("simple")

    n1 = qss.LiqssAtom("n1", a=0.005, dq=1e-3)  
    n2 = qss.LiqssAtom("n2", a=0.005, dq=1e-3)
    vg = qss.LiqssAtom("vg", a=0.001, c=1.0, dq=1e-3)
    cable = qss.LiqssAtom("cable", a=0.001, dq=1e-3) 
    load = qss.LiqssAtom("load", a=0.001, b=2.0, dq=1e-3)

    n1.connect(vg, -1.0)
    n1.connect(cable, 1.0)

    vg.connect(n1, 1.0)
    cable.connect(n1, -1.0)

    n2.connect(cable, 1.0)
    n2.connect(load, -1.0)

    cable.connect(n2, -1.0)
    load.connect(n2, 1.0)

    sys.add_atoms(n1, n2, vg, cable, load)

    sys.initialize()
    sys.run_to(0.1)

    sys.plot()


if __name__ == "__main__":

    test1()
    #test2()