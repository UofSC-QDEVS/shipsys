"""QDL Model Library

Device architecture:

  .------------------.                  .------------------.
  |    Device        |                  |    Device        |
  |                  |   input/output   |                  |
  | .--------------. |                  | .--------------. |
  | | Input Port   | |                  | | Output Port  | |
  | |              | |    Connection    | |              | |
  | | (Connector)<--------------------------(Connector)  | |
  | '--------------' |                  | '--------------' |
  |                  |                  |                  |
  | .--------------. |                  | .--------------. |
  | |  Inout Port  | |                  | |  Inout Port  | |
  | |              | |    Connection1   | |              | |
  | | (Connector1)<-------------------------(Connector1) | |
  | | (Connector2)------------------------->(Connector2) | |
  | '--------------' |    Connection2   | '--------------' |
  |                  |                  |                  |
  '------------------'                  '------------------'


"""


from qdl import *


# ============================ Basic Devices ===================================


class GroundNode(Device):

    def __init__(self, name="ground"):

        Device.__init__(self, name)

        self.voltage = SourceAtom(name="voltage", source_type=SourceType.CONSTANT,
                                  x0=0.0, units="V", dq=1.0)

        self.add_atom(self.voltage)


class ConstantSourceNode(Device):

    def __init__(self, name="source", v0=0.0):

        Device.__init__(self, name)

        self.voltage = SourceAtom(name="voltage", source_type=SourceType.CONSTANT,
                                  x0=v0, units="V", dq=1.0)

        self.add_atom(self.voltage)


class PwmSourceNode(Device):

    def __init__(self, name="source", vlo=0.0, vhi=1.0, freq=1e3, duty=0.5):

        Device.__init__(self, name)

        self.voltage = SourceAtom(name="voltage", source_type=SourceType.PWM, x0=vlo, x1=vhi,
                                  x2=vlo, freq=freq, duty=duty, dq=1.0)

        self.add_atom(self.voltage)


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

        self.h = SourceAtom("h", source_type=source_type, x0=h0,
                                 x1=h1, x2=h2, xa=ha, freq=freq, phi=phi,
                                 duty=duty, t1=t1, t2=t2, dq=dq, units="A")

        self.voltage = StateAtom("voltage", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="V")

        self.add_atoms(self.h, self.voltage)

        self.voltage.add_connection(self.h, coeffunc=self.bii)

        self.voltage.add_jacfunc(self.voltage, self.aii)

    def connect(self, branch, terminal="i"):

        if terminal == "i":
            self.voltage.add_connection(branch.current, coeffunc=self.aij)
            self.voltage.add_jacfunc(branch.current, self.aij)

        elif terminal == "j":
            self.voltage.add_connection(branch.current, coeffunc=self.aji)
            self.voltage.add_jacfunc(branch.current, self.aji)

    @staticmethod
    def aii(self, v=0):
        return -self.g / self.c

    @staticmethod
    def bii(self):
        return 1.0 / self.c

    @staticmethod
    def aij(self, v=0, i=0):
        return -1.0 / self.c

    @staticmethod
    def aji(self, i=0, v=0):
        return 1.0 / self.c


class LimBranch(Device):

    """Generic LIM Lantency Branch with R, L, V, T and Z components.

              +                       v_ij(t)                    -

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

        self.e = SourceAtom("e", source_type=source_type, x0=e0,
                                 x1=e1, x2=e2, xa=ea, freq=freq, phi=phi,
                                 duty=duty, t1=t1, t2=t2, dq=dq, units="V")

        self.current = StateAtom("current", x0=0.0, coeffunc=self.aii, dq=dq,
                              units="A")

        self.add_atoms(self.e, self.current)

        self.current.add_connection(self.e, coeffunc=self.bii)

        self.current.add_jacfunc(self.current, self.aii)

    def connect(self, inode, jnode):

        self.current.add_connection(inode.voltage, coeffunc=self.aij)
        self.current.add_connection(jnode.voltage, coeffunc=self.aji)

        self.current.add_jacfunc(inode.voltage, self.aij)
        self.current.add_jacfunc(jnode.voltage, self.aji)

    @staticmethod
    def aii(self, i=0):
        return -self.r / self.l

    @staticmethod
    def bii(self):
        return 1.0 / self.l

    @staticmethod
    def aij(self, i=0, v=0):
        return 1.0 / self.l

    @staticmethod
    def aji(self, v=0, i=0):
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

        self.vd = SourceAtom(name="vd", source_type=SourceType.CONSTANT,
                                x0=0.0, units="V", dq=1.0)

        self.vq = SourceAtom(name="vq", source_type=SourceType.CONSTANT,
                                x0=0.0, units="V", dq=1.0)

        self.add_atoms(self.vd, self.vq)


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

        self.voltage = voltage

        self.vd0 = voltage * sin(th0)
        self.vq0 = voltage * cos(th0)

        self.vd = SourceAtom(name="vd", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_vd, x0=self.vd0, units="V", dq=1.0)

        self.vq = SourceAtom(name="vq", source_type=SourceType.FUNCTION,
                             srcfunc=self.get_vq, x0=self.vq0, units="V", dq=1.0)

        self.add_atoms(self.vd, self.vq)

        self.theta = None

    def connect_theta(self, theta):

        self.theta = theta

        self.vd.add_connection(self.theta)
        self.vq.add_connection(self.theta)

    @staticmethod
    def get_vd(self, t):

        if self.th_atom:
            return self.voltage * cos(self.theta.q)
        else:
            return self.vd0

    @staticmethod
    def get_vq(self, t):

        if self.th_atom:
            return self.voltage * sin(self.th_atom.q)
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

    def __init__(self, name, l, r=0.0, vd0=0.0, vq0=0.0, w=60.0*PI, id0=0.0,
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

        self.ed = SourceAtom("ed", source_type=source_type, x0=vd0, x1=vd1,
                             x2=vd2, xa=vda, freq=freqd, phi=phid, dmax=dmax,
                             duty=dutyd, t1=td1, t2=td2, dq=dq, units="V")

        self.eq = SourceAtom("eq", source_type=source_type, x0=vq0, x1=vq1,
                             x2=vq2, xa=vqa, freq=freqq, phi=phiq, dmax=dmax,
                             duty=dutyq, t1=tq1, t2=tq2, dq=dq, units="V")

        self.id = StateAtom("id", x0=id0, coeffunc=self.aii, dq=dq, dmax=dmax,
                            units="A")

        self.iq = StateAtom("iq", x0=iq0, coeffunc=self.aii, dq=dq, dmax=dmax,
                            units="A")

        self.add_atoms(self.ed, self.eq, self.id, self.iq)

        self.id.add_connection(self.ed, coeffunc=self.bii)
        self.iq.add_connection(self.eq, coeffunc=self.bii)

        self.id.add_jacfunc(self.id, self.aii)
        self.iq.add_jacfunc(self.iq, self.aii)

    def connect(self, inodedq, jnodedq):

        self.id.add_connection(inodedq.vd, coeffunc=self.aij)
        self.id.add_connection(jnodedq.vd, coeffunc=self.aji)

        self.iq.add_connection(inodedq.vq, coeffunc=self.aij)
        self.iq.add_connection(jnodedq.vq, coeffunc=self.aji)

        self.id.add_jacfunc(inodedq.vd, self.aij)
        self.id.add_jacfunc(jnodedq.vd, self.aji)

        self.iq.add_jacfunc(inodedq.vq, self.aij)
        self.iq.add_jacfunc(jnodedq.vq, self.aji)

    @staticmethod
    def aii(self, idq=0):
        return -(self.r + self.w * self.l) / self.l

    @staticmethod
    def bii(self):
        return 1.0 / self.l

    @staticmethod
    def aij(self, idq=0, vdq=0):
        return 1.0 / self.l

    @staticmethod
    def aji(self, idq=0, vdq=0):
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

        self.hd = SourceAtom("hd", source_type=source_type, x0=id0, x1=id1,
                                  x2=id2, xa=ida, freq=freqd, phi=phid,
                                  duty=dutyd, t1=td1, t2=td2, dq=dq, units="A")

        self.hq = SourceAtom("hq", source_type=source_type, x0=iq0, x1=iq1,
                                  x2=iq2, xa=iqa, freq=freqq, phi=phiq,
                                  duty=dutyq, t1=tq1, t2=tq2, dq=dq, units="A")

        self.vd = StateAtom("vd", x0=vd0, coeffunc=self.aii, dq=dq, units="V")

        self.vq = StateAtom("vq", x0=vq0, coeffunc=self.aii, dq=dq, units="V")

        self.add_atoms(self.hd, self.hq, self.vd, self.vq)

        self.vd.add_connection(self.hd, coeffunc=self.bii)
        self.vq.add_connection(self.hq, coeffunc=self.bii)

        self.vd.add_jacfunc(self.vd, self.aii)
        self.vq.add_jacfunc(self.vq, self.aii)

    def connect(self, device, terminal="i"):

        if terminal == "i":

            self.vd.add_connection(device.id, coeffunc=self.aij)
            self.vq.add_connection(device.iq, coeffunc=self.aij)

            self.vd.add_jacfunc(device.id, self.aij)
            self.vq.add_jacfunc(device.iq, self.aij)

        elif terminal == "j":

            self.vd.add_connection(device.id, coeffunc=self.aji)
            self.vq.add_connection(device.iq, coeffunc=self.aji)

            self.vd.add_jacfunc(device.id, self.aji)
            self.vq.add_jacfunc(device.iq, self.aji)

    @staticmethod
    def aii(self, vdq=0):
        return -(self.g + self.w * self.c) / self.c

    @staticmethod
    def bii(self):
        return 1.0 / self.c

    @staticmethod
    def aij(self, vdq=0, idq=0):
        return -1.0 / self.c

    @staticmethod
    def aji(self, vdq=0, idq=0):
        return 1.0 / self.c


class SyncMachineDQ(Device):

    """Synchronous Machine Reduced DQ Model as Voltage source behind a
    Lim Latency Branch.

    """

    def __init__(self, name, Psm=25.0e6, VLL=4160.0, ws=60.0*PI,
                 P=4, pf=0.80, rs=3.00e-3, Lls=0.20e-3, Lmq=2.00e-3,
                 Lmd=2.00e-3, rkq=5.00e-3, Llkq=0.04e-3, rkd=5.00e-3,
                 Llkd=0.04e-3, rfd=20.0e-3, Llfd=0.15e-3, vfdb=90.1, Kp=10.0e4,
                 Ki=10.0e4, J=4221.7, fkq0=0.0, fkd0=0.0, ffd0=0.0, wr0=60.0*PI,
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

        self.wr.add_jacfunc(self.wr  , self.jwr_wr )
        self.wr.add_jacfunc(self.ids , self.jwr_ids )
        self.wr.add_jacfunc(self.iqs , self.jwr_iqs )
        self.wr.add_jacfunc(self.fkq , self.jwr_fkq )
        self.wr.add_jacfunc(self.fkd , self.jwr_fkd )
        self.wr.add_jacfunc(self.ffd , self.jwr_ffd )
        self.wr.add_jacfunc(self.th  , self.jwr_th  )

        self.th.add_jacfunc(self.wr  ,  self.jth_wr )

        # ports:

        self.id = self.iqs
        self.iq = self.ids

        self.vd = None  # terminal voltage d connection
        self.vq = None  # terminal voltage q connection

        self.input = None  # avr

    def connect(self, bus, avr=None):

        self.vd = self.iqs.add_connection(bus.vq)
        self.vq = self.ids.add_connection(bus.vd)

        self.iqs.add_jacfunc(bus.vq, self.jiqs_vq)
        self.ids.add_jacfunc(bus.vd, self.jids_vd)

        if avr:
            self.input = self.ffd.add_connection(avr.vfd, self.vfdb)
            self.ffd.add_jacfunc(avr.vfd, self.jffd_vfd)

    def vtermd(self):
        return self.vd.value()

    def vtermq(self):
        return self.vq.value()

    def vfd(self):
        if self.input:
            return self.input.value()
        else:
           return self.vfdb # no feedback control

    @staticmethod
    def jffd_vfd(self, ffd, vfd):
        return self.vfdb

    @staticmethod
    def jids_vd(self, ids, vd):
        return -1 / self.Lls

    @staticmethod
    def jiqs_vq(self, iqs, vq):
        return -1 / self.Lls

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
    def jwr_ids(self, wr, ids):
        return (0.75*self.P*(-self.Lq*self.iqs.q+self.Ld*self.iqs.q-(self.Lmq*self.fkq.q)/(self.Lmq+self.Llkq)))/self.J

    @staticmethod
    def jwr_iqs(self, wr, iqs):
        return (0.75*self.P*(-self.Lq*self.ids.q+self.Ld*self.ids.q+(self.Lmd*(self.fkd.q/self.Llkd+self.ffd.q/self.Llfd))/(self.Lmd/self.Llkd+self.Lmd/self.Llfd+1.0)))/self.J

    @staticmethod
    def jwr_fkq(self, wr, fkq):
        return (-(0.75 * self.Lmq * self.P * self.ids.q)
                / (self.J * (self.Lmq + self.Llkq)))

    @staticmethod
    def jwr_fkd(self, wr, fkd):
        return ((0.75 * self.Lmd * self.P * self.iqs.q) / (self.J * self.Llkd
               * (self.Lmd / self.Llkd + self.Lmd / self.Llfd + 1)))

    @staticmethod
    def jwr_ffd(self, wr, ffd):
        return ((0.75 * self.Lmd * self.P * self.iqs.q) / (self.J * self.Llfd
                * (self.Lmd / self.Llkd + self.Lmd / self.Llfd + 1)))

    @staticmethod
    def jwr_wr(self, wr):
        return -self.Kp / self.J

    @staticmethod
    def jwr_th(self, wr, fthkq):
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
     vt      '->| ------ |----'    Vrmin               ,-.    .--------.   |
                | 1+s*Tr |                            ( Σ )<--| Se(Ve) |<--+
                '--------'                             `-' +  '--------'   |
                                                      + ^      .----.      |
                 (x1, x2)                               '------| Ke |<-----'
                                                               '----'
    """

    def __init__(self, name, VLL=4160.0, vref=1.0, Kpr=200.0, Kir=0.8,
                 Kdr=1e-3, Tdr=1e-3, Ka=1.0, Ta=1e-4, Vrmin=0.0,
                 Vrmax=5.0, Te=1.0, Ke=1.0, Sea=1.0119, Seb=0.0875,
                 x10=0.0, x20=0.0, x30=0.0, vfd0=None,
                 dq_x1=1e-8, dq_x2=1e-8, dq_x3=1e-5, dq_vfd=1e-2):

        Device.__init__(self, name)

        self.VLL = VLL
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

        x10 = x10
        x20 = x20
        x30 = x30
        if not vfd0:
            vfd0 = self.vref

        dq_x1 = dq_x1
        dq_x2 = dq_x2
        dq_x3 = dq_x3
        dq_vfd = dq_vfd

        self.x1 = StateAtom("x1", x0=x10, derfunc=self.dx1, dq=dq_x1)
        self.x2 = StateAtom("x2", x0=x20, derfunc=self.dx2, dq=dq_x2)
        self.x3 = StateAtom("x3", x0=x30, derfunc=self.dx3, dq=dq_x3)
        self.vfd = StateAtom("vfd", x0=vfd0, derfunc=self.dvfd, dq=dq_vfd)

        self.add_atoms(self.x1, self.x2, self.x3, self.vfd)

        self.x2.add_connection(self.x1)
        self.x3.add_connection(self.x1)
        self.x3.add_connection(self.x2)
        self.vfd.add_connection(self.x3)

        self.x1.add_jacfunc(self.x1, self.jx1_x1)

        self.x2.add_jacfunc(self.x1, self.jx2_x1)

        self.x3.add_jacfunc(self.x1, self.jx3_x1)
        self.x3.add_jacfunc(self.x2, self.jx3_x2)

        self.vfd.add_jacfunc(self.x3, self.jvfd_x3)
        self.vfd.add_jacfunc(self.vfd, self.jvfd_vfd)

        self.vd = None
        self.vq = None

    def connect(self, bus, sm):

        self.vd = self.x1.add_connection(bus.vd, 1.0 / self.VLL)
        self.vq = self.x1.add_connection(bus.vq, 1.0 / self.VLL)

        self.x3.add_connection(bus.vd)
        self.x3.add_connection(bus.vq)

        self.x1.add_jacfunc(bus.vd, self.jx1_vd)
        self.x1.add_jacfunc(bus.vq, self.jx1_vq)

        self.x3.add_jacfunc(bus.vd, self.jx3_vd)
        self.x3.add_jacfunc(bus.vq, self.jx3_vq)

    @staticmethod
    def jx1_x1(self, x1):
        return -1.0/self.Tdr

    @staticmethod
    def jx1_vd(self, x1, vd):
        return -self.vd.value()/sqrt(self.vq.value()**2+self.vd.value()**2)

    @staticmethod
    def jx1_vq(self, x1, vq):
        return -self.vq.value()/sqrt(self.vq.value()**2+self.vd.value()**2)

    @staticmethod
    def jx2_x1(self, x2, x1):
        return 1

    @staticmethod
    def jx3_x1(self, x3, x1):
        return self.Kir*self.Tdr-self.Kdr/self.Tdr

    @staticmethod
    def jx3_x2(self, x3, x2):
        return self.Kir

    @staticmethod
    def jx3_vd(self, x3, vd):
        return (-((self.Kpr*self.Tdr+self.Kdr)*self.vd.value())
                /sqrt(self.vq.value()**2+self.vd.value()**2))

    @staticmethod
    def jx3_vq(self, x3, vq):
        return (-((self.Kpr*self.Tdr+self.Kdr)*self.vq.value())
                /sqrt(self.vq.value()**2+self.vd.value()**2))

    @staticmethod
    def jvfd_x3(self, vfd, x3):
        return self.Ka/(self.Ta*self.Te)

    @staticmethod
    def jvfd_vfd(self, vfd):
        return -self.Ke/self.Te

    @staticmethod
    def dx1(self, x1):
        return (-1.0 / self.Tdr * x1 + (self.vref
               - sqrt(self.vd.value()**2 + self.vq.value()**2)))

    @staticmethod
    def dx2(self, x2):
        return self.x1.q

    @staticmethod
    def dx3(self, x3):
        return -1.0 / self.Ta + ((self.Kir * self.Tdr - self.Kdr / self.Tdr)
                * self.x1.q + self.Kir * self.x2.q + (self.Kdr + self.Kpr
                * self.Tdr) * (self.vref - sqrt(self.vd.value()**2
                + self.vq.value()**2)))

    @staticmethod
    def dvfd(self, vfd):
        return (self.Ka / self.Ta * self.x3.q - vfd * self.Ke) / self.Te


class DCMotor(Device):

    """

    Jm * dwr = Kt * ia - Bm * wr
    La * dia = -ra * ia + va - Ke * wr

    dwr = (Kt * ia - Bm * wr) / Jm
    dia = (-ra * ia + va - Ke * wr) / La

              ia(t) -->

             Ra    La                         (shaft)
        o----VVV---UUU----.              .-------o------.
     (inode)              |              |       |      |   +
       +              +  ,^.    Kt * ia ,^.     <.     _|_
       va               <   >          < ^ >  B <.   J ___  wr
       -              -  `.' Ke * wr    `.'     <.      |
     (jnode)              |              |       |      |   -
        o-----------------'              '-------+------'
                                                _|_
                                                 -
    """

    def __init__(self, name, ra=0.1, La=0.01, Jm=0.1, Bm=0.001, Kt=0.1, Ke=0.1,
                 ia0=0.0, wr0=0.0, dq_ia=1e-2, dq_wr=1e-1):

        Device.__init__(self, name)

        self.name = name

        self.ra = ra
        self.La = La
        self.Jm = Jm
        self.Bm = Bm
        self.Kt = Kt
        self.Ke = Ke

        self.ia0 = ia0
        self.wr0 = wr0

        self.dq_ia = dq_ia
        self.dq_wr = dq_wr

        self.ia = StateAtom("ia", x0=ia0, derfunc=self.dia, der2func=self.d2ia,
                            units="A", dq=dq_ia)

        self.wr = StateAtom("wr", x0=wr0, derfunc=self.dwr, der2func=self.d2wr,
                            units="rad/s", dq=dq_wr)

        self.add_atoms(self.ia, self.wr)

        #self.ia.add_connection(self.ia)
        self.ia.add_connection(self.wr)

        #self.wr.add_connection(self.wr)
        self.wr.add_connection(self.ia)

        self.ia.add_jacfunc(self.ia, self.jia_ia)
        self.ia.add_jacfunc(self.wr, self.jia_wr)

        self.wr.add_jacfunc(self.wr, self.jwr_wr)
        self.wr.add_jacfunc(self.ia, self.jwr_ia)

        self.vi = None
        self.vj = None

        self.current = self.ia

    def connect(self, inode, jnode):

        self.ia.add_connection(inode.voltage)
        self.ia.add_connection(jnode.voltage)

        self.ia.add_jacfunc(inode.voltage, self.jia_vi)
        self.ia.add_jacfunc(jnode.voltage, self.jia_vj)

        self.vi = inode.voltage
        self.vj = jnode.voltage

    @staticmethod
    def dia(self, ia):
        return (-self.Ke * self.wr.q - self.vj.q + self.vi.q - ia * self.ra) / self.La

    @staticmethod
    def dwr(self, wr):
        return (self.Kt * self.ia.q - self.Bm * wr) / self.Jm

    @staticmethod
    def d2ia(self, ia, dia):
        return (-self.Ke * self.wr.d - self.vi.d + self.vj.d - dia * self.ra) / self.La

    @staticmethod
    def d2wr(self, wr, dwr):
        return (self.Kt * self.ia.d - self.Bm * dwr) / self.Jm

    @staticmethod
    def jia_ia(self, ia):
        return -self.ra / self.La

    @staticmethod
    def jia_wr(self, ia, wr):
        return -self.Ke / self.La

    @staticmethod
    def jia_vi(self, ia, vi):
        return 1 / self.La

    @staticmethod
    def jia_vj(self, ia, vj):
        return -1 / self.La

    @staticmethod
    def jwr_ia(self, wr, ia):
        return self.Kt / self.Jm

    @staticmethod
    def jwr_wr(self, wr):
        return -self.Bm / self.Jm


class Converter(Device):

    """                                              io
                  ii (isum, branch)                 ---->
                 ---->                              r     l
                o-----+------.------.         .---VVV---UUU---o  (jnode, vpos)
                      |      |      |  +      |               +
                  ^  ,^.    <. g   _|_       ,^. +
                h | <   >   <.     ___ vi   <   >  e          vo  (vpos - vneg)
                     `.'    <.    c |        `.'
                      |      |      |  -      |               -
                      '------+------'         '---------------o  (inode, vneg)
                            _|_
                             -

    ii = vi * g + dvi * c - h
    e  = io * r + dio * l + vo

    dvi = (-g * vi + i + h) / c
    dio = (-v - io * r + e) / l


    """

    def __init__(self, name, r=0.0, l=0.01, c=0.01, g=0.0, freq=100.0, duty=1.0,
                 io0=0.0, vi0=0.0, dq_i=None, dq_v=None):

        Device.__init__(self, name)

        self.name = name

        self.r = r
        self.l = l
        self.c = c
        self.g = g

        self.freq = freq
        self.duty = duty

        self.io0 = io0
        self.vi0 = vi0

        self.dq_i = dq_i
        self.dq_v = dq_v

        self.vi = StateAtom("vi", x0=vi0, coeffunc=self.jvi_vi, units="V", dq=dq_v)

        self.io = StateAtom("io", x0=io0, coeffunc=self.jio_io, units="A", dq=dq_i)

        self.e = SourceAtom(name="e", source_type=SourceType.PWM, x1=0.0, x2=1.0,
                            gainfunc=self.ke, freq=freq, duty=duty, dq=1.0)

        self.h = SourceAtom(name="h", source_type=SourceType.PWM, x1=0.0, x2=-1.0,
                            gainfunc=self.kh, freq=freq, duty=duty, dq=1.0)

        self.add_atoms(self.io, self.vi, self.e, self.h)

        self.vi.add_connection(self.h, coeffunc=self.jvi_h)
        self.vi.add_jacfunc(self.vi, self.jvi_vi)

        self.io.add_connection(self.e, coeffunc=self.jio_e)
        self.io.add_jacfunc(self.io, self.jio_io)

        # output connection aliases:

        self.voltage = self.vi
        self.current = self.io

    def connect(self, branch, inode, jnode, terminal="j"):

        self.io.add_connection(inode.voltage, coeffunc=self.jio_vneg)
        self.io.add_connection(jnode.voltage, coeffunc=self.jio_vpos)

        self.io.add_jacfunc(inode.voltage, self.jio_vneg)
        self.io.add_jacfunc(jnode.voltage, self.jio_vpos)

        if terminal == "i":
            self.vi.add_connection(branch.current, coeffunc=self.jvi_ii)
            self.vi.add_jacfunc(branch.current, self.jvi_ii)

        elif terminal == "j":
            self.voltage.add_connection(branch.current, coeffunc=self.jvi_ij)
            self.voltage.add_jacfunc(branch.current, self.jvi_ij)

    @staticmethod
    def ke(self):
        return self.vi.q

    @staticmethod
    def kh(self):
        return self.io.q

    @staticmethod
    def jio_io(self, *args):
        return -self.r / self.l

    @staticmethod
    def jvi_vi(self, *args):
        return -self.g / self.c

    @staticmethod
    def jio_e(self, *args):
        return 1.0 / self.l

    @staticmethod
    def jvi_h(self, *args):
        return 1.0 / self.c

    @staticmethod
    def jio_vpos(self, *args):
        return -1.0 / self.l

    @staticmethod
    def jio_vneg(self, *args):
        return 1.0 / self.l

    @staticmethod
    def jvi_ii(self, *args):
        return -1.0 / self.c

    @staticmethod
    def jvi_ij(self, *args):
        return 1.0 / self.c


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

    def __init__(self, name, ws=30*PI, P=4, Tb=26.53e3, Rs=31.8e-3,
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

        self.iqs.add_jacfunc(self.iqs, self.jiqs_iqs)
        self.iqs.add_jacfunc(self.ids, self.jiqs_ids)
        self.iqs.add_jacfunc(self.iqr, self.jiqs_iqr)
        self.iqs.add_jacfunc(self.idr, self.jiqs_idr)
        self.iqs.add_jacfunc( self.wr, self.jiqs_wr )

        self.ids.add_jacfunc(self.iqs, self.jids_iqs)
        self.ids.add_jacfunc(self.ids, self.jids_ids)
        self.ids.add_jacfunc(self.iqr, self.jids_iqr)
        self.ids.add_jacfunc(self.idr, self.jids_idr)
        self.ids.add_jacfunc( self.wr, self.jids_wr )

        self.iqr.add_jacfunc(self.iqs, self.jiqr_iqs)
        self.iqr.add_jacfunc(self.ids, self.jiqr_ids)
        self.iqr.add_jacfunc(self.iqr, self.jiqr_iqr)
        self.iqr.add_jacfunc(self.idr, self.jiqr_idr)
        self.iqr.add_jacfunc( self.wr, self.jiqr_wr )

        self.idr.add_jacfunc(self.iqs, self.jidr_iqs)
        self.idr.add_jacfunc(self.ids, self.jidr_ids)
        self.idr.add_jacfunc(self.iqr, self.jidr_iqr)
        self.idr.add_jacfunc(self.idr, self.jidr_idr)
        self.idr.add_jacfunc( self.wr, self.jidr_wr )

        self.wr.add_jacfunc(self.iqs, self.jwr_iqs)
        self.wr.add_jacfunc(self.ids, self.jwr_ids)
        self.wr.add_jacfunc(self.iqr, self.jwr_iqr)
        self.wr.add_jacfunc(self.idr, self.jwr_idr)
        self.wr.add_jacfunc( self.wr, self.jwr_wr )

        # ports:

        self.id = self.iqs
        self.iq = self.ids

        self.vds = None  # terminal voltage d atom
        self.vqs = None  # terminal voltage q atom

    def connect(self, bus):

        self.vds = bus.vq
        self.vqs = bus.vd

        self.iqs.add_connection(self.vqs)
        self.iqr.add_connection(self.vqs)

        self.ids.add_connection(self.vds)
        self.idr.add_connection(self.vds)

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
            +(-2*self.Lm**2-2*self.Llr*self.Lm)*self.idr.q)*self.wr.q+(self.Lm+self.Llr)*self.vqs.q-self.Lm*self.vqr()
            +(-self.Lm-self.Llr)*self.Rs*self.iqs.q+self.Lm*self.Rr*self.iqr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def dids(self, ids):
        return -((self.Lm**2*self.iqs.q+(self.Lm**2+self.Llr*self.Lm)*self.iqr.q)*self.ws
	        +((-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.iqs.q
		    +(-2*self.Lm**2-2*self.Llr*self.Lm)*self.iqr.q)*self.wr.q+(-self.Lm-self.Llr)*self.vds.q+self.Lm*self.vdr()
		    +(self.Lm+self.Llr)*self.Rs*self.ids.q-self.Lm*self.Rr*self.idr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def diqr(self, iqr):
        return -(((self.Lm**2+self.Lls*self.Lm)*self.ids.q+(self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)*self.idr.q)*self.ws
	        +((-2*self.Lm**2-2*self.Lls*self.Lm)*self.ids.q+(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.idr.q)*self.wr.q
	        +self.Lm*self.vqs.q+(-self.Lm-self.Lls)*self.vqr()-self.Lm*self.Rs*self.iqs.q+(self.Lm+self.Lls)*self.Rr*self.iqr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def didr(self, idr):
        return (((self.Lm**2+self.Lls*self.Lm)*self.iqs.q+(self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)*self.iqr.q)*self.ws
	        +((-2*self.Lm**2-2*self.Lls*self.Lm)*self.iqs.q+(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.iqr.q)*self.wr.q
	        -self.Lm*self.vds.q+(self.Lm+self.Lls)*self.vdr()+self.Lm*self.Rs*self.ids.q+(-self.Lm-self.Lls)*self.Rr*self.idr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)

    @staticmethod
    def dwr(self, wr):
        return (self.P / (2.0 * self.J)) * (0.75 * self.P * ((self.Lls
                * self.ids.q + self.Lm * (self.ids.q + self.idr.q)) * self.iqs.q
                - (self.Lls * self.iqs.q + self.Lm * (self.iqs.q + self.iqr.q))
                * self.ids.q) - self.Tb * (wr / self.ws)**3)

    @staticmethod
    def jiqs_iqs(self, iqs):
        return (((-self.Lm-self.Llr)*self.Rs)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jiqs_ids(self, iqs, ids):
        return ((self.Lm**2*self.ws+(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm
                -self.Llr*self.Lls)*self.wr.q)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jiqs_iqr(self, iqs, iqr):
        return 	((self.Lm*self.Rr)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jiqs_idr(self, iqs, idr):
        return (((self.Lm**2+self.Llr*self.Lm)*self.ws+(-2*self.Lm**2-2*self.Llr*self.Lm)
                 *self.wr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jiqs_wr(self, iqs, wr):
        return (((-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)
                 *self.ids.q+(-2*self.Lm**2-2*self.Llr*self.Lm)*self.idr.q)
                /((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jids_iqs(self, ids, iqs):
        return ((-self.Lm**2*self.ws-(-2*self.Lm**2+(-self.Lls-self.Llr)
                *self.Lm-self.Llr*self.Lls)*self.wr.q)/((self.Lls+self.Llr)
                *self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jids_ids(self, ids):
        return (-((self.Lm+self.Llr)*self.Rs)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jids_iqr(self, ids, iqr):
        return ((-(self.Lm**2+self.Llr*self.Lm)*self.ws-(-2*self.Lm**2-2
                *self.Llr*self.Lm)*self.wr.q)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jids_idr(self, ids, idr):
        return ((self.Lm*self.Rr)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jids_wr(self, ids, wr):
        return ((-(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)
                 *self.iqs.q-(-2*self.Lm**2-2*self.Llr*self.Lm)*self.iqr.q)
                /((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jiqr_iqs(self, iqr, iqs):
        return ((self.Lm*self.Rs)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jiqr_ids(self, iqr, ids):
        return ((-(self.Lm**2+self.Lls*self.Lm)*self.ws-(-2*self.Lm**2
                -2*self.Lls*self.Lm)*self.wr.q)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jiqr_iqr(self, iqr):
        return (-((self.Lm+self.Lls)*self.Rr)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jiqr_idr(self, iqr, idr):
        return ((-(self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)
                *self.ws-(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr
                *self.Lls)*self.wr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jiqr_wr(self, iqr, wr):
        return ((-(-2*self.Lm**2-2*self.Lls*self.Lm)*self.ids.q-(-2*self.Lm**2
                +(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.idr.q)
                /((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jidr_iqs(self, idr, iqs):
        return (((self.Lm**2+self.Lls*self.Lm)*self.ws+(-2*self.Lm**2-2
                *self.Lls*self.Lm)*self.wr.q)/((self.Lls+self.Llr)*self.Lm
                +self.Llr*self.Lls))

    @staticmethod
    def jidr_ids(self, idr, ids):
        return ((self.Lm*self.Rs)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jidr_iqr(self, idr, iqr):
        return (((self.Lm**2+(self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls)*self.ws
                 +(-2*self.Lm**2+(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)
                 *self.wr.q)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jidr_idr(self, idr):
        return (((-self.Lm-self.Lls)*self.Rr)/((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jidr_wr(self, idr, wr):
        return (((-2*self.Lm**2-2*self.Lls*self.Lm)*self.iqs.q+(-2*self.Lm**2
                +(-self.Lls-self.Llr)*self.Lm-self.Llr*self.Lls)*self.iqr.q)
                /((self.Lls+self.Llr)*self.Lm+self.Llr*self.Lls))

    @staticmethod
    def jwr_iqs(self, wr, iqs):
        return ((0.375*self.P**2*(self.Lm*(self.ids.q+self.idr.q)-(self.Lm+self.Lls)
                *self.ids.q+self.Lls*self.ids.q))/self.J)

    @staticmethod
    def jwr_ids(self, wr, ids):
        return ((0.375*self.P**2*(-self.Lm*(self.iqs.q+self.iqr.q)+(self.Lm+self.Lls)
                *self.iqs.q-self.Lls*self.iqs.q))/self.J)

    @staticmethod
    def jwr_iqr(self, wr, iqr):
        return (-(0.375*self.Lm*self.P**2*self.ids.q)/self.J)

    @staticmethod
    def jwr_idr(self, wr, idr):
        return ((0.375*self.Lm*self.P**2*self.iqs.q)/self.J)

    @staticmethod
    def jwr_wr(self, wr):
        return (-(1.5*self.P*self.Tb*self.wr.q**2)/(self.J*self.ws**3))


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

    def __init__(self, name, w=2*PI*60.0, alpha_cmd=0.0, Lc=76.53e-6, Rdc=0.0,
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

        if mu >= PI_3 or mu + alpha >= PI:
            mu = PI_3
            alpha = PI_3 - acos((2 * self.Lc * self.w * idc) / e)

        return alpha, mu

    def iqg_com(self, vdg, vqg, idc, alpha, mu):

        E = self.E(vdg, vqg)
        k1 = 2 * sqrt(3) / PI
        k2 = 3 * sqrt(3) * E / (PI * self.Lc * self.w)
        k3 = 3 * sqrt(2) * E / (4 * PI * self.Lc * self.w)

        return (k1 * idc * (sin(mu + alpha - PI5_6) - sin(alpha - PI5_6))
              * k2 * cos(alpha) * (cos(mu + alpha) - cos(alpha))
              + k3 * (cos(2*mu) - cos(2*alpha + 2*mu)))

    def idg_com(self, vdg, vqg, idc, alpha, mu):

        E = self.E(vdg, vqg)
        k1 = 2 * sqrt(3) / PI
        k2 = 3 * sqrt(2) * E / (PI * self.Lc * self.w)
        k3 = 3 * sqrt(2) * E / (4 * PI * self.Lc * self.w)
        k4 = 3 * sqrt(2) * E / (2 * PI * self.Lc * self.w)

        return (k1 * idc * (-cos(mu + alpha - PI5_6) + cos(alpha - PI5_6))
              * k2 * cos(alpha) * (sin(mu + alpha) - sin(alpha))
              + k3 * (sin(2*mu) - sin(2*alpha + 2*mu))
              - k4 * mu)

    def iqg_cond(self, idc, alpha, mu):
        return 2 * sqrt(3) / PI * idc * (sin(alpha + PI7_6) - sin(alpha + mu + PI5_6))

    def idg_cond(self, idc, alpha, mu):
        return 2 * sqrt(3) / PI * idc * (-cos(alpha + PI7_6) + cos(alpha + mu + PI5_6))

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
        k1 = 3 * sqrt(3) * sqrt(2) / PI

        e = k1 * E * cos(alpha)
        req = self.Rdc + 3/PI * self.Lc * self.w

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

    def __init__(self, name, w=2*PI*60.0, Lc=76.53e-6, Rdc=0.0,
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

        self.S = sqrt(3/2) * 2 * sqrt(3) / PI
        self.S2 = self.S**2
        self.Req = Rdc + 3 / PI * Lc * w
        self.Leq = Ldc + 2 * Lc

        # call super:

        Device.__init__(self, name)

        # atoms:

        self.iq = SourceAtom("iq", source_type=SourceType.CONSTANT, x0=0.0,
                              units="A", dq=dq_i)

        self.id = StateAtom("id", x0=id0, derfunc=self.did, units="A", dq=dq_i)

        self.vdc = StateAtom("vdc", x0=vdc0, derfunc=self.dvdc, units="V", dq=dq_v)

        self.add_atoms(self.id, self.iq, self.vdc)

        # atom connections:

        self.id.add_connection(self.vdc)
        self.vdc.add_connection(self.id)

        self.id.add_jacfunc(self.id, self.jid_id)
        self.id.add_jacfunc(self.vdc, self.jid_vdc)
        self.vdc.add_jacfunc(self.id, self.jvdc_id)
        self.vdc.add_jacfunc(self.vdc, self.jvdc_vdc)

        # ports:

        self.vd = None  # terminal voltage d atom
        self.vq = None  # terminal voltage q atom

    def connect(self, bus):

        self.vd = bus.vd
        self.id.add_connection(bus.vd)

    @staticmethod
    def did(self, id):  # id depends on: vd, vdc
        return (self.vd.q * self.S2 - self.Req * id - self.vdc.q * self.S) / self.Leq

    @staticmethod
    def dvdc(self, vdc):  # vdc depends on: id
        return (self.id.q / self.S - vdc / self.R) / self.C

    @staticmethod
    def jid_id(self, id):
       return -self.Req / self.Leq

    @staticmethod
    def jid_vdc(self, id, vdc):
       return -self.S / self.Leq

    @staticmethod
    def jvdc_id(self, vdc, id):
       return 1 / (self.C * self.S)

    @staticmethod
    def jvdc_vdc(self, vdc):
       return -1 / (self.C * self.R)


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
    def j12(self, theta, omega):
        return 1.0

    @staticmethod
    def j21(self, omega, theta):
        return -self.g / self.l * cos(theta)

    @staticmethod
    def j22(self, omega):
        return -self.r


class LiqssTest(Device):

    """Original Liqss test/demo stiff system from the paper

    'Linearly Implicit Quantization–Based Integration
    Methods for Stiff Ordinary Differential Equations'

    dx1 = 0.01 * x2
    dx2 = -100 * x1 - 100 * x2 + 2020

    lambda1 ~= -0.01 and lambda2 ~= -100.0 (stiffness ratio of 1e4)

    """

    def __init__(self, name, x10=0.0, x20=20.0, dq1=1.0, dq2=1.0):

        Device.__init__(self, name)

        self.x1 = StateAtom("x1", x0=x10, derfunc=self.dx1, der2func=self.d2x1,
                            units="", dq=dq1)


        self.x2 = StateAtom("x2", x0=x20, derfunc=self.dx2, der2func=self.d2x2,
                            units="", dq=dq2)

        self.add_atoms(self.x1, self.x2)

        #self.x1.broadcast_to.append(self.x2)
        #self.x2.broadcast_to.append(self.x1)
        #self.x2.broadcast_to.append(self.x2)

        self.x1.add_connection(self.x2)
        self.x2.add_connection(self.x1)
        self.x2.add_connection(self.x2)

        self.x1.add_jacfunc(self.x2, self.j12)
        self.x2.add_jacfunc(self.x1, self.j21)
        self.x2.add_jacfunc(self.x2, self.j22)

    @staticmethod
    def dx1(self, x1):
        return 0.01 * self.x2.q

    @staticmethod
    def dx2(self, x2):
        return -100 * self.x1.q - 100 * x2 + 2020

    @staticmethod
    def d2x1(self, x1, d1):
        return 0.01 * self.x2.d

    @staticmethod
    def d2x2(self, x2, d2):
        return -100 * self.x1.d - 100 * d2

    @staticmethod
    def j12(self, x1, x2):
        return 0.01

    @staticmethod
    def j21(self, x2, x1):
        return -100

    @staticmethod
    def j22(self, x2):
        return -100


class MLiqssTest(Device):

    """mLiqss test/demo system from the paper:

    'Improving Linearly Implicit Quantized State System Methods'

    dx1 = -x1 - x2 + 0.2
    dx2 = x1 - x2 + 1.2

    """

    def __init__(self, name, x10=-4.0, x20=4.0, dq1=1.0, dq2=1.0):

        Device.__init__(self, name)

        self.x1 = StateAtom("x1", x0=x10, derfunc=self.dx1, der2func=self.ddx1,
                            units="", dq=dq1)


        self.x2 = StateAtom("x2", x0=x20, derfunc=self.dx2, der2func=self.ddx2,
                            units="", dq=dq2)

        self.add_atoms(self.x1, self.x2)

        #self.x1.add_connection(self.x1)
        self.x1.add_connection(self.x2)
        self.x2.add_connection(self.x1)
        #self.x2.add_connection(self.x2)

        self.x1.add_jacfunc(self.x1, self.j11)
        self.x1.add_jacfunc(self.x2, self.j12)
        self.x2.add_jacfunc(self.x1, self.j21)
        self.x2.add_jacfunc(self.x2, self.j22)

    @staticmethod
    def dx1(self, x1):
        return -x1 - self.x2.q + 0.2

    @staticmethod
    def dx2(self, x2):
        return self.x1.q - x2 + 1.2

    @staticmethod
    def ddx1(self, x1, dx1):
        return -dx1 - self.x2.dx

    @staticmethod
    def ddx2(self, x2, dx2):
        return self.x1.dx - dx2

    @staticmethod
    def j11(self, x1):
        return -1.0

    @staticmethod
    def j12(self, x1, x2):
        return -1.0

    @staticmethod
    def j21(self, x2, x1):
        return 1.0

    @staticmethod
    def j22(self, x2):
        return -1.0


class CoupledPendulums(Device):

    """
    dw1 = (-g/l1 * sin(th1)/cos(th1) + w1*w1 * sin(th1)/cos(th1)
          + k*l2/(m1*l1) * sin(th2) / cos(th1)
          + k*l1/(m1*l1) * sin(th1) / cos(th1))

    dw2 = (-g/l2 * sin(th2)/cos(th2) + w2*w2 * sin(th2)/cos(th2)
          + k*l1/(m2*l2) * sin(th1) / cos(th2)
          + k*l2/(m2*l2) * sin(th2) / cos(th2))

    dth1 = w1
    dth2 = w2

    """

    def __init__(self, name, k=1.0, r1=1.0, r2=1.0, l1=1.0, l2=1.0, m1=1.0,
                 m2=1.0, th10=0.0, w10=0.0, th20=0.0, w20=0.0, dq_w=1e-3,
                 dq_th=1e-3):

        Device.__init__(self, name)

        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2

        self.g = 9.81

        self.w1  = StateAtom("w1", x0=w10,  derfunc=self.dw1,
                             der2func=self.d2w1, units="rad/s", dq=dq_w)


        self.th1 = StateAtom("th1", x0=th10, derfunc=self.dth1,
                             der2func=self.d2w2, units="rad", dq=dq_th)


        self.w2  = StateAtom("w2", x0=w20, derfunc=self.dw2,
                             der2func=self.d2th1, units="rad/s", dq=dq_w)


        self.th2 = StateAtom("th2", x0=th20, derfunc=self.dth2,
                             der2func=self.d2th2, units="rad", dq=dq_th)


        self.add_atoms(self.w1, self.th1, self.w2, self.th2)

        """
        dw1 = (-g/l1 * sin(th1)/cos(th1) + w1*w1 * sin(th1)/cos(th1)
              + k*l2/(m1*l1) * sin(th2) / cos(th1)
              + k*l1/(m1*l1) * sin(th1) / cos(th1))

        dw2 = )-g/l2 * sin(th2)/cos(th2) + w2*w2 * sin(th2)/cos(th2)
              + k*l1/(m2*l2) * sin(th1) / cos(th2)
              + k*l2/(m2*l2) * sin(th2) / cos(th2))

        dth1 = w1
        dth2 = w2

        """

        #self.w1.add_connection(self.w1)
        self.w1.add_connection(self.th1)
        self.w1.add_connection(self.w2)

        #self.w2.add_connection(self.w2)
        self.w2.add_connection(self.th2)
        self.w2.add_connection(self.w1)

        self.th1.add_connection(self.w1)
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
    def d2w1(self, w1, dw1):
        return ((2 * w1 * sin(self.th1.q) * dw1) / cos(self.th1.q)
                + (self.k * self.l2 * cos(self.th2.q) * self.th2.d)
                / (self.l1 * self.m1 * cos(self.th1.q)) + (self.k * self.l2
                * sin(self.th1.q) * self.th1.d * sin(self.th2.q)) / (self.l1
                * self.m1 * cos(self.th1.q)**2) + (w1**2 * sin(self.th1.q)**2
                * self.th1.d) / cos(self.th1.q)**2 + (self.k
                * sin(self.th1.q)**2 * self.th1.d) / (self.m1
                * cos(self.th1.q)**2) - (self.g * sin(self.th1.q)**2
                * self.th1.d) / (self.l1 * cos(self.th1.q)**2) + w1**2
                * self.th1.d + (self.k * self.th1.d) / self.m1 - (self.g
                * self.th1.d) / self.l1)

    @staticmethod
    def d2w2(self, w2, dw2):
        return ((2 * w2 * sin(self.th2.q) * dw2) / cos(self.th2.q)
                + (w2**2 * sin(self.th2.q)**2 * self.th2.d) / cos(self.th2.q)**2
                + (self.k * sin(self.th2.q)**2 * self.th2.d) / (self.m2
                * cos(self.th2.q)**2) - (self.g * sin(self.th2.q)**2
                * self.th2.d) / (self.l2 * cos(self.th2.q)**2) + (self.k * self.l1
                * sin(self.th1.q) * sin(self.th2.q) * self.th2.d) / (self.l2
                * self.m2 * cos(self.th2.q)**2) + w2**2 *self.th2.d
                + (self.k * self.th2.d) / self.m2 - (self.g * self.th2.d)
                / self.l2 + (self.k * self.l1 *cos(self.th1.q) * self.th1.d)
                / (self.l2 * self.m2 * cos(self.th2.q)))

    @staticmethod
    def d2th1(self, th1, dth1):
        return self.w1.d

    @staticmethod
    def d2th2(self, th2, dth2):
        return self.w2.d

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


class Pendulum2(SymbolicDevice):

    def __init__(self, name, mu=1.0, l=1.0, g=9.81, w0=0.0, a0=0.0,
                 dq_w=1e-3, dq_a=1e-3):

        SymbolicDevice.__init__(self, name)

        self.add_constant("g", desc="Acceleration of gravity", units="m.s^-2", value=g)

        self.add_parameter("mu", desc="Viscous friction", value=mu)
        self.add_parameter("l", desc="Length", value=l)

        self.add_state("w", "dw_dt", desc="Angular velocity", units="rad/s", x0=w0, dq=dq_w)
        self.add_state("a", "da_dt", desc="Angle", units="rad", x0=a0, dq=dq_a)

        self.add_diffeq("dw_dt + (mu * w + g / l * sin(a))")
        self.add_diffeq("da_dt - w")


class LimNode2(SymbolicDevice):

    def __init__(self, name, c, g=0.0, h=0.0, v0=0.0, dq=None):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("c", desc="Capacitance", units="F", value=c)
        self.add_parameter("g", desc="Conductance", units="S", value=g)
        self.add_parameter("h", desc="Source Current", units="A", value=h)

        self.add_state("v", "dv_dt", desc="Voltage", units="V", x0=v0, dq=dq)

        self.add_electrical_port("positive", output="v", input="isum")

        self.add_diffeq("c * dv_dt + g * v - h - isum")


class LimBranch2(SymbolicDevice):

    def __init__(self, name, l, r=0.0, e=0.0, i0=0.0, dq=None):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("l", desc="Inductance", units="H", value=l)
        self.add_parameter("r", desc="Resistance", units="Ohm", value=r)
        self.add_parameter("e", desc="Source Voltage", units="V", value=e)

        self.add_state("i", "di_dt", desc="Current", units="A", x0=i0, dq=dq)

        self.add_electrical_port("positive", output="i", input="vpos")
        self.add_electrical_port("negative", output="i", input="vneg", sign=-1)

        self.add_diffeq("l * di_dt + r * i - e + vpos - vneg")


class LimNodeDQ2(SymbolicDevice):

    def __init__(self, name, c, g=0.0, ws=2*pi*60, theta=0.0, h=0.0, vd0=0.0,
                 vq0=0.0, dq=None):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("c", desc="Capacitance", units="F", value=c)
        self.add_parameter("g", desc="Conductance", units="S", value=g)
        self.add_parameter("h", desc="Source Current", units="A", value=h)
        self.add_parameter("ws", desc="Radian frequency", units="rad/s", value=ws)
        self.add_parameter("theta", desc="DQ Angle", units="rad", value=theta)

        self.add_state("vd", "dvd_dt", desc="D-axis Voltage", units="V", x0=vd0, dq=dq)
        self.add_state("vq", "dvq_dt", desc="Q-axis Voltage", units="V", x0=vq0, dq=dq)

        self.add_dq_port("positive", inputs=("id", "iq"), outputs=("vd", "vq"))

        self.add_diffeq("c * dvd_dt + (g + ws*c) * vd - h * cos(theta) - id")
        self.add_diffeq("c * dvq_dt + (g + ws*c) * vq - h * sin(theta) - iq")


class LimBranchDQ2(SymbolicDevice):

    def __init__(self, name, l, r=0.0, ws=2*pi*60, theta=0.0, e=0.0, id0=0.0,
                 iq0=0.0, dq=None):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("l", desc="Inductance", units="H", value=l)
        self.add_parameter("r", desc="Resistance", units="Ohm", value=r)
        self.add_parameter("e", desc="Source Voltage", units="V", value=e)
        self.add_parameter("ws", desc="Radian frequency", units="rad/s", value=ws)
        self.add_parameter("theta", desc="DQ Angle", units="rad", value=theta)

        self.add_state("id", "did_dt", desc="D-axis Current", units="A", x0=id0, dq=dq)
        self.add_state("iq", "diq_dt", desc="Q-axis Current", units="A", x0=iq0, dq=dq)

        self.add_dq_port("positive", inputs=("vposd", "vposq"), outputs=("id", "iq"))
        self.add_dq_port("negative", inputs=("vnegd", "vnegq"), outputs=("id", "iq"), sign=-1)

        self.add_diffeq("l * did_dt + (r + ws*l) * id - e * cos(theta) + vposd - vnegd")
        self.add_diffeq("l * diq_dt + (r + ws*l) * iq - e * sin(theta) + vposq - vnegq")


class InductionMachineDQ2(SymbolicDevice):

    def __init__(self, name, ws=30*pi, P=4, Tb=26.53e3, Rs=31.8e-3,
                 Lls=0.653e-3, Lm=38e-3, Rr=24.1e-3, Llr=0.658e-3, J=250.0,
                 iqs0=0.0, ids0=0.0, iqr0=0.0, idr0=0.0, wr0=0.0, dq_i=1e-2,
                 dq_wr=1e-1):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("ws",  desc="Synchronous Speed",         units="rad/s", value=ws)
        self.add_parameter("P",   desc="Pole Pairs",                units="",      value=P)
        self.add_parameter("Tb",  desc="Base Torque",               units="N.m",   value=Tb)
        self.add_parameter("Rs",  desc="Stator Resistance",         units="Ohm",   value=Rs)
        self.add_parameter("Lls", desc="Stator Leakage Inductance", units="H",     value=Lls)
        self.add_parameter("Lm",  desc="Magnetizing Inductance",    units="H",     value=Lm)
        self.add_parameter("Rr",  desc="Rotor Resistance",          units="Ohm",   value=Rr)
        self.add_parameter("Llr", desc="Rotor Leakage Inductance",  units="H",     value=Llr)
        self.add_parameter("J",   desc="Inertia",                   units="",      value=J)

        self.add_state("ids", "dids_dt", desc="Stator D-axis Current", units="A",     x0=ids0, dq=dq_i)
        self.add_state("iqs", "diqs_dt", desc="Stator Q-axis Current", units="A",     x0=iqs0, dq=dq_i)
        self.add_state("idr", "didr_dt", desc="Rotor D-axis Current",  units="A",     x0=idr0, dq=dq_i)
        self.add_state("iqr", "diqr_dt", desc="Rotor Q-axis Current",  units="A",     x0=iqr0, dq=dq_i)
        self.add_state("wr",  "dwr_dt",  desc="Rotor Speed",           units="rad/s", x0=wr0,  dq=dq_wr)

        self.add_dq_port("terminal", inputs=("vds", "vqs"), outputs=("ids", "iqs"))

        self.add_algebraic("fqs", "Lls * iqs + Lm * (iqs + iqr)")
        self.add_algebraic("fds", "Lls * ids + Lm * (ids + idr)")
        self.add_algebraic("fqr", "Llr * iqr + Lm * (iqr + iqs)")
        self.add_algebraic("fdr", "Llr * idr + Lm * (idr + ids)")
        self.add_algebraic("Tm", "Tb * (wr / ws)**3")
        self.add_algebraic("Te", "-3/2 * P/2 * (fds * iqs - fqs * ids)")

        self.add_diffeq("Rs * iqs + wr * fds + (Lls + Lm) * diqs_dt + Lm * diqr_dt - vqs")
        self.add_diffeq("Rs * ids - wr * fqs + (Lls + Lm) * dids_dt + Lm * didr_dt - vds")
        self.add_diffeq("Rr * iqr + (ws - wr) * fdr + (Llr + Lm) * diqr_dt + Lm * diqs_dt")
        self.add_diffeq("Rr * idr - (ws - wr) * fqr + (Llr + Lm) * didr_dt + Lm * dids_dt")
        self.add_diffeq("dwr_dt - P/2 * (Te - Tm) / J")


class SyncMachineDQ2(SymbolicDevice):

    def __init__(self, name, VLL=4160, ws=60*pi, P=4, rs=3e-3, Lls=2e-4,
                 Lmq=2e-3, Lmd=2e-3, rkq=5e-3, Llkq=4e-5, rkd=5e-3, Llkd=4e-5,
                 rfd=2e-2, Llfd=15e-5, vfdb=90.1, Kp=1e5, Ki=1e5, J=4221.7,
                 fkq0=0, fkd0=0, ffd0=0, wr0=0, th0=0, iqs0=0, ids0=0,
                 dq_i=1e-2, dq_f=1e-2, dq_wr=1e-1, dq_th=1e-3, dq_v=1e0):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("VLL" , value=VLL )
        self.add_parameter("ws"  , value=ws  )
        self.add_parameter("P"   , value=P   )
        self.add_parameter("rs"  , value=rs  )
        self.add_parameter("Lls" , value=Lls )
        self.add_parameter("Lmq" , value=Lmq )
        self.add_parameter("Lmd" , value=Lmd )
        self.add_parameter("rkq" , value=rkq )
        self.add_parameter("Llkq", value=Llkq)
        self.add_parameter("rkd" , value=rkd )
        self.add_parameter("Llkd", value=Llkd)
        self.add_parameter("rfd" , value=rfd )
        self.add_parameter("Llfd", value=Llfd)
        self.add_parameter("vfdb", value=vfdb)
        self.add_parameter("Kp"  , value=Kp  )
        self.add_parameter("Ki"  , value=Ki  )
        self.add_parameter("J"   , value=J   )
        self.add_parameter("vfd" , value=vfdb)

        self.add_state("fkq", "dfkq_dt", units="Wb"   , x0=fkq0, dq=dq_f )
        self.add_state("fkd", "dfkd_dt", units="Wb"   , x0=fkd0, dq=dq_f )
        self.add_state("ffd", "dffd_dt", units="Wb"   , x0=ffd0, dq=dq_f )
        self.add_state("wr" , "dwr_dt" , units="rad/s", x0=wr0 , dq=dq_wr)
        self.add_state("th" , "dth_dt" , units="rad"  , x0=th0 , dq=dq_th)
        self.add_state("iqs", "diqs_dt", units="A"    , x0=iqs0, dq=dq_i )
        self.add_state("ids", "dids_dt", units="A"    , x0=ids0, dq=dq_i )

        self.add_dq_port("terminal", inputs=("vds", "vqs"), outputs=("ids", "iqs"))

        self.add_output_port("vterm", output="vterm")
        self.add_input_port("vfd", input="vfd")

        self.add_algebraic("Lq", "Lls + (Lmq * Llkq) / (Llkq + Lmq)")
        self.add_algebraic("Ld", "Lls + (Lmd * Llfd * Llkd) / (Lmd * Llfd + Lmd * Llkd + Llfd * Llkd)")
        self.add_algebraic("fq", "Lmq / (Lmq + Llkq) * fkq")
        self.add_algebraic("fd", "Lmd * (Lmd * (fkd / Llkd + ffd / Llfd)) / (1 + Lmd / Llfd + Lmd / Llkd)")
        self.add_algebraic("Te", "3/2 * P/2 * (fds * iqs - fqs * ids)")
        self.add_algebraic("Tm", "Kp * (ws - wr) + th * Ki")
        self.add_algebraic("vterm", "sqrt(vds**2 + vqs**2)")

        self.add_diffeq("diqs_dt * Lls + rs * iqs + wr * Ld + wr * fd - vqs")
        self.add_diffeq("dids_dt * Lls + rs * ids - wr * Lq - wr * fq - vds")
        self.add_diffeq("dfkq_dt * Llkq + rkq * (fkq - Lq * iqs - fq + Lls * iqs)")
        self.add_diffeq("dfkd_dt * Llkd + rkd * (fkd - Ld * ids + fd + Lls * ids)")
        self.add_diffeq("dffd_dt * Llfd + rfd * (ffd - Ld * ids + fd + Lls * ids) - vfd*VLL")
        self.add_diffeq("dwr_dt * J + Tm - Te")
        self.add_diffeq("dth_dt + wr - ws")


class Exciter(SymbolicDevice):

    def __init__(self, name, VLL=4160, vref=1.0, Kpr=200.0, Kir=0.8, Kdr=1e-3, Tdr=1e-3,
                 Ka=1.0, Ta=1e-4, Vrmin=0.0, Vrmax=5.0, Te=1.0, Ke=1.0, x10=0.0,
                 x20=0.0, x30=0.0, vout0=0.0, dq_x1=1e-8, dq_x2=1e-8, dq_x3=1e-5,
                 dq_vout=1e-2):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("VLL"  , value=VLL  )
        self.add_parameter("vref" , value=vref )
        self.add_parameter("Kpr"  , value=Kpr  )
        self.add_parameter("Kir"  , value=Kir  )
        self.add_parameter("Kdr"  , value=Kdr  )
        self.add_parameter("Tdr"  , value=Tdr  )
        self.add_parameter("Ka"   , value=Ka   )
        self.add_parameter("Ta"   , value=Ta   )
        self.add_parameter("Vrmin", value=Vrmin)
        self.add_parameter("Vrmax", value=Vrmax)
        self.add_parameter("Te"   , value=Te   )
        self.add_parameter("Ke"   , value=Ke   )

        self.add_state("x1",   "dx1_dt",    x0=x10,   dq=dq_x1  )
        self.add_state("x2",   "dx2_dt",    x0=x20,   dq=dq_x2  )
        self.add_state("x3",   "dx3_dt",    x0=x30,   dq=dq_x3  )
        self.add_state("vout", "dvout_dt",  x0=vout0, dq=dq_vout)

        self.add_input_port("vterm", intput="vterm")
        self.add_output_port("vfd", output="vout")

        self.add_algebraic("vin", "vref - vterm / VLL")

        self.add_diffeq("dx1_dt + 1/Tdr * x1 - vin")
        self.add_diffeq("dx2_dt - x1")
        self.add_diffeq("dx3_dt - ((-Kdr/(Tdr**2) + Kir)*x1 + Kir/Tdr * x2 - 1/Ta * x3 + (Kdr/Tdr + Kpr) * vin)")
        self.add_diffeq("dvout_dt - (Ka/Ta * x3 - 1/Te * (vout * Te / Ke))")



class Ground(SymbolicDevice):

    def __init__(self, name):

        SymbolicDevice.__init__(self, name)

        self.add_electrical_port("positive", output="v", input="i")

        self.add_algebraic("v", "0")



class DCMotorSym(SymbolicDevice):

    def __init__(self, name, Vs=10, Gs=1e2, Cl=1e-6, Ra=0.1, La=0.001, Ke=0.1, Kt=0.1,
                 Jm=0.01, Bm=0.001, Jp=0.01, Fp=1, JL=0.5, TL=0, BL=0.1,
                 ia0=0.0, wr0=0.0, dq_ia=1e-1, dq_wr=1e-1):

        SymbolicDevice.__init__(self, name)

        self.add_parameter("Vs", desc="", units="", value=Vs)
        self.add_parameter("Gs", desc="", units="", value=Gs)
        self.add_parameter("Cl", desc="", units="", value=Cl)
        self.add_parameter("Ra", desc="", units="", value=Ra)
        self.add_parameter("La", desc="", units="", value=La)
        self.add_parameter("Ke", desc="", units="", value=Ke)
        self.add_parameter("Kt", desc="", units="", value=Kt)
        self.add_parameter("Jm", desc="", units="", value=Jm)
        self.add_parameter("Bm", desc="", units="", value=Bm)
        self.add_parameter("Jp", desc="", units="", value=Jp)
        self.add_parameter("Fp", desc="", units="", value=Fp)
        self.add_parameter("JL", desc="", units="", value=JL)
        self.add_parameter("TL", desc="", units="", value=TL)
        self.add_parameter("BL", desc="", units="", value=BL)

        self.add_state("ia", "dia_dt", desc="Armature Current", units="A",     x0=ia0, dq=dq_ia)
        self.add_state("wr", "dwr_dt", desc="Rotor Speed",      units="rad/s", x0=wr0, dq=dq_wr)

        self.add_electrical_port("positive", output="i", input="vpos")
        self.add_electrical_port("negative", output="i", input="vneg", sign=-1)

        self.add_diffeq("La * dia_dt + Ra * ia + Ke * wr - (vpos - vneg)")
        self.add_diffeq("Jm * dwr_dt + Bm * wr - Kt * ia + TL")


