
from qdl import *
from models import *

sys = System(qss_method=QssMethod.LIQSS1)

dq = 1

liqss = LiqssTest("liqss", x10=0.0, x20=20.0, dq1=dq, dq2=dq)

sys.add_device(liqss)

sys.initialize(dt=1e-2, dc=False)

sys.run(500, ode=True, qss=True, verbose=2, ode_method="LSODA",
        print_interval=1)

plotargs = {"plot_ode":True,
            "plot_qss":True,
            "plot_zoh":True,
            "plot_upd":False,
            "errorband":True,
            }

if 1:
    sys.plot(liqss.x1, liqss.x2, **plotargs)

if 0:
    plt.plot(liqss.x1.tzoh, liqss.x1.qzoh, "k-", label="$q_1(t)$", linewidth=1)
    plt.plot(liqss.x2.tzoh, liqss.x2.qzoh, "k-", label="$q_2(t)$", linewidth=1)
    plt.grid(linestyle="dotted", which="both")
    plt.xlabel("Time [sec]")
    plt.ylabel("$q_1(t)$, $q_2(t)$")
    plt.xlim((0, 500))
    plt.ylim((-5, 25))
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(50))
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))
    plt.text(350, 22, "$q_1(t)$")
    plt.text(350, 2, "$q_2(t)$")
    plt.show()