"""Latency-Insersion Method Modeling Framework.
"""


class SourceType:

    CONSTANT = "CONSTANT"
    STEP = "STEP"
    SINE = "SINE"
    PWM = "PWM"
    RAMP = "RAMP"
    FUNCTION = "FUNCTION"


class PortDirection(object):

    NONE = "NONE"
    IN = "IN"
    OUT = "OUT"
    INOUT = "INOUT"


class Ode(object):

    """Generic First-order Ordinary Differential Equation (ODE) data.

         dx     
    a * ---- + b*x = c
         dt     

    """

    def __init__(self, parent, a=0.0, b=0.0, c=0.0, afunc=None, bfunc=None,
                 cfunc=None):

        self.parent = parent
        self.a = a
        self.b = b
        self.c = c
        self.afunc = afunc
        self.bfunc = bfunc
        self.cfunc = cfunc


class Source(object):

    """Generic stimulus source data.
    """

    def __init__(self, source_type=SourceType.CONSTANT, x0=0.0, x1=0.0, x2=0.0,
                 xa=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0,
                 srcfunc=None):

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


class Atom(object):

    def __init__(self, name, is_latent=True, ode=None, source=None, units="",
                 x0=0.0):

        self.name = name
        self.is_latent = is_latent
        self.ode = ode 
        self.source = source
        self.units = units
        self.x0 = x0

    def value(self):

        raise NotImplementedError()


class Device(object):

    """Collection of Atoms, Ports and Viewables that comprise a device.

    .--------------.
    |              |-------o inout_port_1
    |              |-------o inout_port_2
    |  Device      | ...
    |              |-------o port_N  
    | (atoms)      | ...
    |              |-------< in_port_1
    | (viewables)  |-------< in_port_2
    |              | ...
    |              |-------< in_port_N 
    |              | ...
    |              |-------> out_port_1
    |              |-------> out_port_2
    |              | ...
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

    def add_atoms(self, *atoms):
        
        for atom in atoms:
            self.atoms.append(atom)

    def add_port(self, port):
        
        self.ports.append(port)

    def add_ports(self, *ports):
        
        for port in ports:
            self.ports.append(port)

    def add_viewable(self, viewable):
        
        self.viewables.append(viewable)

    def add_viewables(self, *viewables):
        
        for viewable in viewables:
            self.viewables.append(viewable)


class Port(object):

    """Mediates the connections and data transfer between atoms.
    """

    def __init__(self, name, atom, direction=PortDirection.NONE):

        self.name = name
        self.direction = direction
        self.atom = atom

        self.connections = []  # [(atom, gain),...]
        self.is_connected = False

    def value(self):

        sum([atom.value()*gain for value, gain in self.connections])

    def connect(self, port, gain=1.0):

        self.connections.append((port.atom, gain))


class Viewable(object):

    def __init__(self, name, func, units=""):

        self.name = name
        self.func = func
        self.units = units


class Branch(Atom):

    """Generic LIM 2-port Branch.

                +     voltage    -
                   .-----------.    
    port_i  o------|    Branch |------o  port_j
                   '-----------'
                       --->
                     current
    """      

    def __init__(self, name, port_i, port_j, i0=0.0, is_latent=True, ode=None,
                 source=None):

        Atom.__init__(name=name, is_latent=is_latent, ode=ode, source=source,
                         units="A", x0=i0)

        self.port_i = port_i
        self.port_j = port_j

    def voltage(self):

        return self.port_i.value() - self.port_j.value()

    def current(self):

        return self.value()


class Node(Atom):

    """Generic LIM Node.

                +     voltage    -
                   .-----------.     .
    port_i  o------|     Node  |-----||-
                   '-----------'     '
                       --->
                     current
    """      

    def __init__(self, name, port_i, v0=0.0, is_latent=True, ode=None,
                 source=None,):

        Atom.__init__(name=name, is_latent=is_latent, ode=ode, source=source,
                         units="V", x0=v0)

        self.port_i = port_i


    def voltage(self):

        return self.value()

    def current(self):

        return self.port_i.value()


class LatencyBranch(Branch):

    """Generic LIM Lantency Branch with R, L, V, T and Z components.

                    +                v_ij(t)              -

                                     i(t) --> 

                                     v_t(t) =         v_z(t) =
             v(t)  +   -   +   -    T_ijk * v_k(t)   Z_ijpq * i_pq(t)
    port_i   ,-.     R       L          ,^.             ,^.             port_j
      o-----(- +)---VVV-----UUU-------< - + >---------< - + >-------------o
             `-'                        `.'             `.'  
                                         ^               ^
                                         |               |
                                      port_t           port_z

    vij = -(v + vt + vz) + i*R + i'*L

    i'= 1/L * (vij + v + vt + vz - i*R) 

    """

    def __init__(self, name, port_i, port_j, R=0.0, L=0.0,
                 V=0.0, i0=0.0, source=None, port_t=None, port_z=None):

        # ode params:
        self.R = R
        self.L = L
        self.V = V

        self.ode = Ode(L, R, V)

        Branch.__init__(name=name, port_i=port_i, port_j=port_j, i0=i0,
                           is_latent=True, ode=self.ode, source=source)

        # ports:
        self.port_i = port_i
        self.port_j = port_j
        self.port_t = port_t
        self.port_z = port_z


class LatencyNode(Node):

    """Generic LIM Lantency Node with G, C, I, B and S components.
                                
                       \       
               i_i2(t)  ^   ... 
                         \   
                 i_i1(t)  \     i_ik(t)
                ----<------o------>----
                           |
                           |   ^
                           |   | i_i(t)
           .--------.------+-------------.----------------.
           |        |      |    i_b =    |      i_s =     |         +
          ,-.      <.     _|_   b_k *   ,^.     s_pq *   ,^.     
    I(t) ( ^ )   G <.   C ___  v_k(t) <  ^  >  i_pq(t) <  ^  >     v(t)
          `-'      <.      |            `.' ^            `.' ^   
           |        |      |             |   \            |   \     -
           '--------'------+-------------'----\-----------'    \
                          _|_                  \                '---- port_s
                           -                    '---- port_b   
    """

    def __init__(self, name, port_i, G=0.0, L=0.0,
                 I=0.0, v0=0.0, source=None, port_b=None, port_s=None):

        # ode params:
        self.G = G
        self.C = C
        self.I = I

        self.ode = Ode(C, G, I)

        Branch.__init__(name=name, port_i=port_i, v0=v0, is_latent=True,
                           ode=self.ode, source=source)

        # ports:
        self.port_i = port_i
        self.port_b = port_b
        self.port_s = port_s
