"""QDL Simulation tests.
"""

def test1():

    sys = System("test1", dq=1e-2)

    ground = Ground()
    node1 = Node("node1", c=1.0, g=1.0, h=1.0)
    branch1 = Branch("branch1", l=1.0, r=1.0, e=1.0)

    sys.add_devices(gnd, node1, branch1)

    sys.add_connection(branch1.positive, node1.positive)
    sys.add_connection(branch1.negative, gnd.positive)

    sys.initialize_ss(1e-3)
    sys.run_ss_to(10.0)

    sys.initialize()
    sys.run_to(10.0)
    sys.plot(n1, b1, plot_ss=True)


if __name__ == "__main__":

    test1()

