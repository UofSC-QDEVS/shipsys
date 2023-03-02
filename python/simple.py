from qdl import *
from models import *


sys = System(qss_method=QssMethod.LIQSS1)

dq_v = 1e-2
dq_i = 1e-2

ground = GroundNode("ground")
node1 = LimNode("node1", 5.0, 0.1, 1.0, dq=dq_v)
branch1 = LimBranch("branch1", 5.0, 0.1, 1.0, dq=dq_i)

sys.add_devices(ground, node1, branch1)

branch1.connect(ground, node1)
node1.connect(branch1, terminal='j')

tstop = 1000.0
dt = 1.0e-2
dc = 1
optimize_dq = 0
ode = 1
qss = 1

sys.initialize(dt=dt, dc=dc)

def event1(sys):
    node1.g = 2.0

def event2(sys):
    node1.g = 1.0

sys.schedule(event1, 200.0)
sys.schedule(event2, 800.0)

sys.run(tstop, ode=ode, qss=qss, verbose=True, ode_method="LSODA",
        optimize_dq=optimize_dq)

plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
            "plot_upd":qss, "upd_bins":1000, "errorband":True,
            "legloc":"lower right"}

sys.plot(node1.voltage, branch1.current, **plotargs)
