
from math import cos, pi
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


class System(object):

    def __init__(self):

        self.devices = []
        self.atoms = []
        self.n = 0

    def add_device(self, device):

        self.devices.append(device)
        for atom in device.atoms:
            atom.index = self.n
            self.n += 1

    def jac(self):

        jac = np.zeros((self.n, self.n))

        for device in self.devices:
            for atom in device.atoms:
                for other, func in atom.jac:
                    jac[atom.index, other.index] += func(device)

        return jac

    def beq(self):

        beq = np.zeros((self.n, 1))

        for device in self.devices:
            for atom in device.atoms:
                for func in atom.beq:
                    beq[atom.index, 0] += func(device)

        return beq

    def x(self):

        x = np.zeros((self.n, 1))

        for device in self.devices:
            for atom in device.atoms:
                x[atom.index, 0] = atom.x

        return x

    def run(self, dt, tstop):

        t = np.arange(0.0, tstop, dt)
        y = np.zeros((self.n, t.size))

        for device in self.devices:
            for atom in device.atoms:
                y[atom.index, 0] = atom.x0

        for i in range(0, t.size-1):

            y[:,i+1] = y[:,i] + np.dot(self.jac(), y[:,i]) * dt

            for device in self.devices:
                for atom in device.atoms:
                    atom.x = y[atom.index, 0]

        return y, t


class Atom(object):

    def __init__(self, error, x0=0.0):

        self.error = error
        self.x0 = x0

        self.jac = []
        self.beq = []
        self.index = -1
        self.x = x0

    def add_jac(self, other, func):

        self.jac.append((other, func))

    def add_beq(self, func):

        self.beq.append(func)


class Device(object):

    def __init__(self):

        self.atoms = []

    def add_atom(self, atom):

        self.atoms.append(atom)


class Pendulum(Device):

    def __init__(self, r=1.0, l=1.0, theta0=0.0, omega0=0.0):

        Device.__init__(self)

        self.l = l
        self.r = r
        self.g = 9.81

        self.theta = Atom(0.01, theta0)
        self.omega = Atom(0.01, omega0)

        self.theta.add_jac(self.omega, self.j12)
        self.omega.add_jac(self.theta, self.j21)
        self.omega.add_jac(self.omega, self.j22)

        self.theta.add_beq(self.u1)
        self.omega.add_beq(self.u2)

        self.add_atom(self.theta)
        self.add_atom(self.omega)

    @staticmethod
    def j12(self):
        return 1.0

    @staticmethod
    def j21(self):
        return -self.g / self.l * cos(self.theta.x)

    @staticmethod
    def j22(self):
        return -self.r

    @staticmethod
    def u1(self):
        return 0.0

    @staticmethod
    def u2(self):
        return 0.0


sys = System()
pen = Pendulum(theta0=pi/4)
sys.add_device(pen)
y, t = sys.run(1e-3, 20)

plt.plot(t, y[0, :])
plt.plot(t, y[1, :])
plt.show()




