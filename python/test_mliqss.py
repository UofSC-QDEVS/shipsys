
from qdl import *
from models import *

tf = 80
dq = 1
dt = 1e-2
dc = False
ode = True
qss = True

sys = System(qss_method=QssMethod.LIQSS1)
mliqss = MLiqssTest("mliqss", dq1=dq, dq2=dq)
sys.add_device(mliqss)

sys.initialize(dt=dt, dc=dc)

sys.run(tf, ode=ode, qss=qss, verbose=2, ode_method="LSODA")

plotargs = {"plot_ode":ode, "plot_qss":qss, "plot_zoh":qss,
            "plot_upd":qss, "errorband":qss}

sys.plot(mliqss.x1, mliqss.x2, **plotargs)

