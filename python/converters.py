from qdl import *
from models import *


dq_i = 1e-2
dq_v = 1
dq_ia = 1e-1
dq_wr = 1e-1

freq = 100.0
duty = 0.5

i0 = 0.0
v0 = 0.0

Vs = 10.0

sys = System(qss_method=QssMethod.LIQSS1)

ground = GroundNode("ground")

source = LimBranch("source", l=0.01, r=0.01, e0=Vs, i0=i0, dq=dq_i)

converter = Converter("converter", r=0.01, l=0.01, c=0.01, g=0.01, freq=freq, duty=duty,
                      vi0=v0, io0=i0, dq_i=dq_i, dq_v=dq_v)

bus = LimNode("bus", c=0.01, g=10.0, dq=dq_v)

sys.add_devices(ground, source, converter, bus)


source.connect(ground, converter)

converter.connect(source, ground, bus, terminal="j")

bus.connect(converter, terminal="j")

tstop = 5.0
dt = 1e-3
dc = 1
optimize_dq = 0
ode = 1
qss = 0

#def event1(sys): motor.ra = 0.1
#sys.schedule(event1, 2.0)

#def event2(sys): motor.Bm = 0.01
#sys.schedule(event2, 4.0)

#motor.ra = 1000.0

sys.initialize(dt=dt, dc=dc)

sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
        optimize_dq=optimize_dq)

plotargs = {"plot_ode":ode,
            "plot_qss":qss,
            "plot_zoh":1,
            "plot_upd":0,
            "errorband":0,
            }

#sys.plot(source.current, converter.e, converter.h, bus.voltage, **plotargs)
sys.plot(source.current, bus.voltage, **plotargs)



"""
ground = GroundNode("ground")

source = LimBranch("source", l=0.01, r=0.01, e0=Vs, i0=i0, dq=dq_i)

converter = Converter("converter", r=0.01, l=0.01, c=0.01, g=0.01, freq=freq, duty=duty,
                      vi0=v0, io0=i0, dq_i=dq_i, dq_v=dq_v)

bus = LimNode("bus", c=0.01, g=10.0, dq=dq_v)

motor = DCMotor("motor", ra=0.1, La=0.01, Jm=0.1, Bm=0.001, Kt=0.1, Ke=0.1,
                ia0=0.0, wr0=0.0, dq_ia=dq_ia, dq_wr=dq_wr)

sys.add_devices(ground, source, converter, bus, motor)

source.connect(ground, converter)
converter.connect(source, ground, bus, terminal="j")
bus.connect(converter, terminal="j")

bus.connect(motor, terminal="i")
motor.connect(bus, ground)

tstop = 15.0
dt = 1e-4
dc = 0
optimize_dq = 0
ode = 0
qss = 1

def event1(sys): motor.ra = 0.1
sys.schedule(event1, 5.0)

def event2(sys): motor.Bm = 0.01
sys.schedule(event2, 10.0)

motor.ra = 1000.0

sys.initialize(dt=dt, dc=dc)

sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
        optimize_dq=optimize_dq)

plotargs = {"plot_ode":ode,
            "plot_qss":qss,
            "plot_zoh":1,
            "plot_upd":0,
            "errorband":0,
            }

#sys.plot(source.current, converter.e, converter.h, bus.voltage, **plotargs)
sys.plot(source.current, bus.voltage, motor.ia, motor.wr, **plotargs)
"""
