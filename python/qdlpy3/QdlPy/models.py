"""QDL Device Model Library.
"""


import qss
import lim
import qdl
from math import pi, sin, cos


class Ground(qdl.LimDevice):

    """Ground device, which is just a LIM source node with voltage == 0.
    """

    def __init__(self, name="gnd"):

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".voltage"

        self.atom = qdl.LimSourceNode(name=atom_name, port_i=self.positive,
                                      source_type=qdl.SourceType.CONSTANT,
                                      v0=0.0)

        self.positive = qdl.LimPort("positive", qdl.PortDirection.INOUT)

        self.add_port(self.positive)
        self.add_atom(self.atom)
        self.positive.add_atom(self.atom)

        self.is_ground = True

class Branch(qdl.LimDevice, qss.Atom):
    
    """A device wrapper for a Generic LIM latency branch.
    """

    def __init__(self, l, r=0.0, e=0.0, source_type=qdl.SourceType.NONE,
                 i0=0.0, i1=0.0, i2=0.0, ia=0.0, freq=0.0, phi=0.0, duty=0.0,
                 t1=0.0, t2=0.0, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        # define ports:

        self.positive = qdl.LimPort("positive", qdl.PortDirection.INOUT)
        self.negative = qdl.LimPort("negative", qdl.PortDirection.INOUT)
        self.vcvs = qdl.LimPort("vcvs", qdl.PortDirection.IN)
        self.ccvs = qdl.LimPort("ccvs", qdl.PortDirection.IN)

        self.add_ports(self.positive_port, self.negative_port, self.vcvs_port,
                       self.ccvs_port)

        # define atom:

        atom_name = name + ".current"

        self.atom = qdl.LimLatencyBranch(name=atom_name,
                                         port_i=self.positive,
                                         port_j=self.negative,
                                         port_t=self.vcvs,
                                         port_z=self.ccvs,
                                         l=l, r=r, e=e,
                                         source_type=source_type, i0=i0, i1=i1,
                                         i2=i2, ia=ia, freq=freq, phi=phi,
                                         duty=duty, t1=t1, t2=t2, dq=dq,
                                         dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                         dtmin=dtmin, dmax=dmax
        self.add_atom(self.atom)

        # associate atom and ports:
        
        self.positive.add_atom(self.atom)
        self.negative.add_atom(self.atom)
        self.vcvs.add_atom(self.atom)
        self.ccvs.add_atom(self.atom)


class Node(qdl.LimDevice):
    
    """A device wrapper for a Generic LIM latency node.
    """

    def __init__(self, name, g, c, h, source_type=qdl.SourceType.NONE, v0=0.0,
                 v1=0.0, v2=0.0, va=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0,
                 t2=0.0, dq=None, dqmin=None, dqmax=None, dqerr=None,
                 dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        # define ports:

        self.positive = qdl.LimPort("positive", qdl.PortDirection.INOUT)
        self.vccs = qdl.LimPort("vccs", qdl.PortDirection.IN)
        self.cccs = qdl.LimPort("cccs", qdl.PortDirection.IN)
        
        self.add_ports(self.positive, self.vccs, self.cccs)

        # define atom:

        atom_name = name + ".voltage"

        self.atom = qdl.LimLatencyNode(name=atom_name,
                                       port_i=self.positive,
                                       port_s=self.vccs,
                                       port_b=self.cccs,
                                       g=g, c=c, h=h, 
                                       source_type=source_type,
                                       v0=v0, v1=v1, v2=v2, va=va,
                                       freq=freq, phi=phi, duty=duty, t1=t1, t2=t2, dq=dq,
                                       dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                       dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)

        # associate atom and ports:

        self.positive.add_atom(self.atom)
        self.vccs.add_atom(self.atom)
        self.cccs.add_atom(self.atom)

    def add_capacitance(self, c):

        self.atom.c += c

        if self.atom.c:
            self.atom.cinv = 1.0 / self.atom.c

    def add_conductance(self, g):

        self.atom.g += g


class ConnectionNode(qdl.LimDevice):
    
    """A connection node for other devices to connect to.
    """

    def __init__(self, name, v0=0.0, dq=None, dqmin=None, dqmax=None,
                 dqerr=None, dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".voltage"

        self.atom = qdl.LimLatencyNode(name=atom_name, c=0.0, v0=v0, dq=dq,
                                       dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                       dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)

    def add_capacitance(self, c):

        self.atom.c += c

        if self.atom.c:
            self.atom.cinv = 1.0 / self.atom.c

    def add_conductance(self, g):

        self.atom.g += g


# ================================= Ground =====================================





# ============================= Voltage Sources ================================


class IdealVoltageSource(qdl.LimDevice):

    """Generic ideal (loseless) voltage source device.
    """

    def __init__(self, name, source_type, i0=0.0, i1=0.0, i2=0.0, ia=0.0,
                 freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None, dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = qdl.LimSourceNode(name=atom_name, source_type=source_type,
                                      i0=i0, i1=i1, i2=i2, va=va, freq=freq,
                                      phi=phi, duty=duty, t1=t1, t2=t2, dq=dq,
                                      dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                                      dtmin=dtmin, dmax=dmax)

        self.add_atom(self.atom)


class VoltageSource(qdl.LimDevice):

    """Generic lossy voltage source device.
    """

    def __init__(self, name, node_pos, node_neg, e, r, l,
                 source_type=qdl.SourceType.CONSTANT, i0=0.0, i1=0.0, i2=0.0,
                 ia=0.0, freq=0.0, phi=0.0, duty=0.0, t1=0.0, t2=0.0, dq=None,
                 dqmin=None, dqmax=None, dqerr=None, dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = qdl.LimLatencyBranch(name=atom_name, nodei=node_neg,
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
                               source_type=qdl.SourceType.CONSTANT, i0=i0, dq=dq,
                               dqmin=dqmin, dqmax=dqmax, dqerr=dqerr,
                               dtmin=dtmin, dmax=dmax)


# ============================= Passive Devices ================================


class Cable(qdl.LimDevice):

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

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.branch = qdl.LimLatencyBranch(name=atom_name, nodei=node_pos,
                                           nodej=node_neg, l=l, r=r, i0=i0,
                                           dq=dq, dqmin=dqmin, dqmax=dqmax,
                                           dqerr=dqerr, dtmin=dtmin, dmax=dmax)

        node_pos.add_capacitance(c*0.5)
        node_pos.add_conductance(g*0.5)

        node_neg.add_capacitance(c*0.5)
        node_neg.add_conductance(g*0.5)
                                                      
        self.add_atoms(self.branch)


class RLLoad(qdl.LimDevice):
    
    """RL to ground model.

                  r     l
    npos  o------VVV---UUU------o nneg
                   ---->     
                     i       
                           
    """

    def __init__(self, name, npos, nneg, r, l, i0=0.0, dq=None, dqmin=None,
                 dqmax=None, dqerr=None, dtmin=None, dmax=None):

        qdl.LimDevice.__init__(self, name)

        atom_name = name + ".current"

        self.atom = qdl.LimLatencyBranch(name=atom_name, nodei=npos, nodej=nneg,
                                         l=l, r=r, i0=i0, dq=dq, dqmin=dqmin,
                                         dqmax=dqmax, dqerr=dqerr, dtmin=dtmin,
                                         dmax=dmax)

        self.add_atom(self.atom)


class RLVLoadDQ(qdl.LimDevice):
    
    """RLV Load DQ Model.

                       id*(r + w*L)      id'*L         vd
       .--------.    +             -    +      -    +  ,-. -    .
       |     o--|----------VVVV-----------UUUU--------(   )-----||-
       |        |                id -->                `-'      '
       | nodedq |   
       |        |      iq*(r + w*L)      iq'*L         vq
       |        |    +             -    +      -    +  ,-. -    .
       |     o--|----------VVVV-----------UUUU--------(   )-----||-
       '--------'                iq -->                `-'      '

    """

    def __init__(self, name, nodedq, r, l, v=0.0, w=60.0*pi, id0=0.0, iq0=0.0, dq=None):

        qdl.LimDevice.__init__(self, name)

        self.nodedq = nodedq

        self.r = r
        self.l = l
        self.v = v
        self.w = w

        self.linv = 1.0 / self.l

        atom_named = name + ".id"
        atom_nameq = name + ".iq"

        self.atomd = qdl.Atom(name=atom_named, derivative=self.derivative_daxis, x0=id0, dq=dq)
        self.atomq = qdl.Atom(name=atom_nameq, derivative=self.derivative_qaxis, x0=iq0, dq=dq)

        self.add_atoms(self.atomd, self.atomq)

        nodedq.connected_branches.append(self)

        for atom in nodedq.atoms:    # todo: optimize, only connect needed

            self.atomd.connect(atom)
            self.atomq.connect(atom)

        self.atomd.derivative = self.derivative_daxis
        self.atomq.derivative = self.derivative_qaxis

    def q_daxis(self):

        return self.atomd.q()

    def q_qaxis(self):

        return self.atomq.q()

    def derivative_daxis(self):

        return (self.linv * (-self.q_daxis() * (self.r + self.w * self.l) + 
                self.nodedq.q_daxis()))
    
    def derivative_qaxis(self):

        return (self.linv * (-self.q_qaxis() * (self.r + self.w * self.l) + 
                self.nodedq.q_qaxis()))


class RCILoadDQ(qdl.LimDevice):
    
    """RCI Load DQ Model.

    """

    def __init__(self, name, nodedq, r, c, i, w, vd0=0.0, vq0=0.0, dq=None):

        qdl.LimDevice.__init__(self, name)

        self.nodedq = nodedq

        self.r = r
        self.c = c
        self.i = i
        self.w = w

        self.cinv = 1.0 / self.c

        atom_named = name + ".vd"
        atom_nameq = name + ".vq"

        self.atomd = qdl.Atom(name=atom_named, x0=vd0, dq=dq, parent=self)
        self.atomq = qdl.Atom(name=atom_nameq, x0=vq0, dq=dq, parent=self)

        self.add_atoms(self.atomd, self.atomq)

        # branch connections:

        self.connected_branches = []

    def q_daxis(self):

        return self.atomd.q()

    def q_qaxis(self):

        return self.atomq.q()

    def derivative_daxis(self):

        return self.cinv * (-self.q_daxis() * self.w * self.c / self.r + self.nodedq.q_daxis())
    
    def derivative_qaxis(self):

        return self.cinv * (-self.q_qaxis() * self.w * self.c / self.r + self.nodedq.q_qaxis())


# ================================ Machines ====================================


class SyncMachineReducedDQ(qdl.LimDevice):

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

        qdl.LimDevice.__init__(self, name)

        # init connected brnaches array:

        self.connected_branches = []

        # derived:

        self.Lq = Lls + (Lmq * Llkq) / (Llkq + Lmq)
        self.Ld = Lls + (Lmd * Llfd * Llkd) / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd)

        # atoms:

        self.fkq = qdl.Atom("sm.fkq", derivative=self.dfkq, x0=self.fkq0, units="Wb",    dq=self.dq*1e-1)
        self.fkd = qdl.Atom("sm.fkd", derivative=self.dfkd, x0=self.fkd0, units="Wb",    dq=self.dq)
        self.ffd = qdl.Atom("sm.ffd", derivative=self.dffd, x0=self.ffd0, units="Wb",    dq=self.dq)
        self.wrm = qdl.Atom("sm.wrm", derivative=self.dwrm, x0=self.wrm0, units="rad/s", dq=self.dq)
        self.th  = qdl.Atom("sm.th",  derivative=self.dth,  x0=self.th0,  units="rad",   dq=self.dq) 

        self.add_atoms(self.fkq, self.fkd, self.ffd, self.wrm, self.th)

        self.fkd.connects(self.ffd)
        self.ffd.connects(self.fkd)
        self.wrm.connects(self.fkq, self.fkd, self.ffd, self.th)
        self.th.connects(self.wrm) 

        # viewables:

        self.vd = qdl.Atom("sm.vd", source_type=qdl.SourceType.FUNCTION, srcfunc=self.vds, srcdt=1e-1, units="V")
        self.vq = qdl.Atom("sm.vq", source_type=qdl.SourceType.FUNCTION, srcfunc=self.vqs, srcdt=1e-1, units="V")
        self.te = qdl.Atom("sm.te", source_type=qdl.SourceType.FUNCTION, srcfunc=self.Te,  srcdt=1e-1, units="N.m")
        self.tm = qdl.Atom("sm.tm", source_type=qdl.SourceType.FUNCTION, srcfunc=self.Tm,  srcdt=1e-1, units="N.m")
        
        self.add_atoms(self.vd, self.vq, self.te, self.tm)

        # branch connections:

        self.q_daxis = self.vds
        self.q_qaxis = self.vqs

    def vqs(self):
        return (self.rs * self.iqs() + self.wrm.q() * self.Ld * self.ids() +
                self.wrm.q() * self.fd())

    def vds(self):
        return (self.rs * self.ids() - self.wrm.q() * self.Lq * self.iqs() -
                self.wrm.q() * self.fq())

    def iqs(self):

        isum = 0.0
        for atom in self.connected_branches:
            isum -= atom.q_qaxis()
        return isum
        
    def ids(self):

        isum = 0.0
        for atom in self.connected_branches:
            isum -= atom.q_daxis()
        return isum

    def vfd(self):
        return self.vfdb

    def fq(self):
        return self.Lmq / (self.Lmq + self.Llkq) * self.fkq.q()

    def fd(self):
        return (self.Lmd * (self.fkd.q() / self.Llkd + self.ffd.q() / self.Llfd) /
                (1.0 + self.Lmd / self.Llfd + self.Lmd / self.Llkd))

    def fqs(self):
        return self.Lq * self.iqs() + self.fq()

    def fds(self):
        return self.Ld * self.ids() + self.fd()

    def Te(self):
        return 3.0 * self.P / 4.0 * (self.fds() * self.iqs() - self.fqs() * self.ids())

    def Tm(self):
        return self.Kp * (self.wmb - self.wrm.q()) + self.Ki * self.th.q()

    def dfkq(self):
        return (-self.rkq / self.Llkq * (self.fkq.q() - self.Lq * self.iqs() -
                self.fq() + self.Lls * self.iqs()))

    def dfkd(self):
        return (-self.rkd / self.Llkd * (self.fkd.q() - self.Ld * self.ids() + 
                self.fd() - self.Lls * self.ids()))

    def dffd(self):
        return (self.vfd() - self.rfd / self.Llfd * (self.ffd.q() - self.Ld *
                self.ids() + self.fd() - self.Lls * self.ids()))

    def dwrm(self):
        return (self.Te() - self.Tm()) / self.J

    def dth(self):
        return self.wmb - self.wrm.q()


class SyncMachineDQ(qdl.LimDevice):

    """7th order Synchronous Machine DQ Model

    """

    def __init__(self, name, f=50.0, wb=2*pi*50.0, Ld=7.0e-3, Ll=2.5067e-3,
                 Lm=6.6659e-3, LD=8.7419e-3, LF=7.3835e-3, Lq=5.61e-3,
                 MQ=4.7704e-3, Ra=0.001, Rs=1.6e-3, RF=9.845e-4, RD=0.11558,
                 RQ=0.0204, n=1.0, J=2.812e4, dq=1e-3):

        self.name = name

        # sm params:

        self.f  = f 
        self.wb = wb
        self.Ld = Ld
        self.Ll = Ll
        self.Lm = Lm
        self.LD = LD
        self.LF = LF
        self.Lq = Lq
        self.MQ = MQ
        self.Ra = Ra
        self.Rs = Rs
        self.RF = RF
        self.RD = RD
        self.RQ = RQ
        self.n  = n 
        self.J  = J 
        self.dq = dq

        # intial conditions:

        self.tm0  = 0.0
        self.fdr0 = 63.661977153494156
        self.fqr0 = -9.330288611488223e-11
        self.fF0  = 72.98578663103797
        self.fD0  = 65.47813807847828
        self.fQ0  = -5.483354402732449e-11
        self.wr0  = 314.1592653589793
        self.th0  = -5.1145054174461215e-05
        self.vdc0 = -1.4784789895393902
        self.vt0  = 1405.8929299980925
        self.idc0 = -0.010300081646320524
        self.vf0  = -1.4784641022274498
        self.ip0  = -0.010300632920201253 

        # qdl params:

        self.dq = dq

        # call super:

        qdl.LimDevice.__init__(self, name)

        # derived:

        det1 = Lm*(Lm**2-LF*Lm)+Lm*(Lm**2-LD*Lm)+(Ll+Ld)*(LD*LF-Lm**2)
        det2 = (Lq+Ll)*MQ-MQ**2

        self.a11 = (LD*LF-Lm**2) / det1
        self.a12 = (Lm**2-LD*Lm) / det1
        self.a13 = (Lm**2-LF*Lm) / det1
        self.a21 = (Lm**2-LD*Lm) / det1
        self.a22 = (LD*(Ll+Ld)-Lm**2) / det1
        self.a23 = (Lm**2-(Ll+Ld)*Lm) / det1
        self.a31 = (Lm**2-LF*Lm) / det1
        self.a32 = (Lm**2-(Ll+Ld)*Lm) / det1
        self.a33 = (LF*(Ll+Ld)-Lm**2) / det1
        self.b11 = MQ / det2
        self.b12 = -MQ / det2
        self.b21 = -MQ / det2
        self.b22 = (Lq+Ll) / det2

        # addl params:

        self.Tm_max = -265000.0  # 25% rated
        self.vb     =  20.0e3
        self.efd0   =  10.3

        # atoms:

        self.tm  = qdl.Atom("sm.tm", source_type=qdl.SourceType.RAMP, x1=0.0, x2=self.Tm_max, t1=5.0, t2=20.0, dq=1e-1, units="N.m")
        
        self.fdr = qdl.Atom("sm.fdr", x0=self.fdr0, derivative=self.dfdr, units="Wb",    dq=self.dq     )
        self.fqr = qdl.Atom("sm.fqr", x0=self.fqr0, derivative=self.dfqr, units="Wb",    dq=self.dq     )
        self.fF  = qdl.Atom("sm.fF",  x0=self.fF0,  derivative=self.dfF,  units="Wb",    dq=self.dq     )
        self.fD  = qdl.Atom("sm.fD",  x0=self.fD0,  derivative=self.dfD,  units="Wb",    dq=self.dq     )
        self.fQ  = qdl.Atom("sm.fQ",  x0=self.fQ0,  derivative=self.dfQ,  units="Wb",    dq=self.dq     )
        self.wr  = qdl.Atom("sm.wr",  x0=self.wr0,  derivative=self.dwr,  units="rad/s", dq=self.dq*0.1 ) 
        self.th  = qdl.Atom("sm.th",  x0=self.th0,  derivative=self.dth,  units="rad",   dq=self.dq     )

        self.add_atoms(self.wr, self.tm, self.fdr, self.fqr, self.fF, self.fD, self.fQ, self.th)

        self.fdr.connects(self.fqr, self.fF, self.fD, self.wr, self.th)
        self.fqr.connects(self.fdr, self.fQ, self.wr, self.th)
        self.fF.connects(self.fdr, self.fD)
        self.fD.connects(self.fdr, self.fF)
        self.fQ.connects(self.fqr)
        self.wr.connects(self.fqr, self.fdr, self.fF, self.fD, self.fQ, self.tm)
        self.th.connects(self.wr)

    def efd(self):
        return self.efd0

    def id(self):
        return self.a11 * self.fdr.q() + self.a12 * self.fF.q() + self.a13 * self.fD.q() 

    def iq(self):
        return self.b11 * self.fqr.q() + self.b12 * self.fQ.q()

    def iF(self):
        return self.a21 * self.fdr.q() + self.a22 * self.fF.q() + self.a23 * self.fD.q()

    def iD(self):
        return self.a31 * self.fdr.q() + self.a32 * self.fF.q() + self.a33 * self.fD.q()

    def iQ(self):
        return self.b21 * self.fqr.q() + self.b22 * self.fQ.q()

    def ed(self):
        return self.vb * sin(self.th.q())

    def eq(self):
        return self.vb * cos(self.th.q())

    # derivative functions:

    def didc(self):
        return 1/self.Lf * (self.Vdc.q() - self.idc.q() * self.Rf - self.vf.q())

    def dvf(self):
        return 1/self.Cf * (idc.q() - ip.q())

    def dip(self):
        return 1/self.Lp * (self.vf.q() - self.ip.q() * self.Rp)

    def dfdr(self):
       return self.ed() - self.Rs * self.id() + self.wr.q() * self.fqr.q()

    def dfqr(self):
        return self.eq() - self.Rs * self.iq() - self.wr.q() * self.fdr.q()

    def dfF(self):
        return self.efd() - self.iF() * self.RF

    def dfD(self):
        return -self.iD() * self.RD

    def dfQ(self):
        return -self.iQ() * self.RQ

    def dwr(self):
        return (self.n/self.J) * (self.iq() * self.fdr.q() - self.id() * self.fqr.q() - self.tm.q())

    def dth(self):
        return self.wr.q() - self.wb
