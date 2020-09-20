"""LTI State space model with steady-state and time-domain solvers.
"""

import numpy as np
import numpy.linalg as la


class StateSpace(object):

    def __init__(self, a, b, c=None, d=None, x0=None, u0=None):

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x0 = x0
        self.x = x0
        self.u0 = u0
        self.n = self.a.shape[0]
        self.m = self.b.shape[1]
        self.dt = -1.0
        self.time = 0.0

    def initialize(self, dt, u0=None, reset_state=True):

        self.dt = dt

        self.n = self.a.shape[0]
        self.m = self.b.shape[1]

        if self.c is None:
            self.c = np.eye(self.n)

        self.p = self.c.shape[0]

        if self.d is None:
            self.d = np.zeros((self.p, self.m))

        if reset_state:

            if self.x0 is None:
                self.x = np.zeros((self.n, 1))
            else:
                self.x = self.x0

            self.t = [0]
            self.y = np.zeros((self.n, 1))
            self.time = 0.0

        if u0 is not None:
            self.u = u0
        elif self.u0 is not None:
            self.u = self.u0
        else:
            self.u = np.zeros((self.m, 1))

        eye = np.eye(self.n)

        self.apr = la.inv(eye - dt * self.a)
        self.bpr = np.dot(self.apr, dt * self.b)

    def step(self, u):

        self.u = u
        self.x = np.dot(self.apr, self.x) + np.dot(self.bpr, self.u)
        y = np.dot(self.c, self.x) + np.dot(self.d, self.u)

        return y

    def run_to(self, tstop):

        t = np.arange(self.time, tstop, self.dt)
        y = np.zeros((self.n, t.size))

        y[:,0:1] = np.dot(self.c, self.x) + np.dot(self.d, self.u0)

        for i in range(1, t.size):
            y[:,i:i+1] = self.step(self.u0)

        self.time = tstop

        self.t = np.concatenate([self.t, t], axis=0)
        self.y = np.concatenate([self.y, y], axis=1)

        return t, y


