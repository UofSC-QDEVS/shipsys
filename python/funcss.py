"""Delegate function-based State Space Framework.
"""

import lti
import numpy as np
import matplotlib.pyplot as plt


class FuncSystem(object):

    def __init__(self, num_states, num_inputs, num_outputs):

        self.n = num_states
        self.m = num_inputs
        self.p = num_outputs

        self.x = FuncVector(self.n)
        self.u = FuncVector(self.m)
        self.y = FuncVector(self.p)

        self.a = FuncMatrix(self.n, self.n)
        self.b = FuncMatrix(self.n, self.m)
        self.c = FuncMatrix(self.p, self.n)
        self.d = FuncMatrix(self.p, self.m)

    def compute_row(self, irow):

        val = 0.0
        for jcol in range(self.n):
            val += self.a.get_coef(irow, jcol) * self.x.get_coef(jcol)

        for jcol in range(self.m):
            val += self.b.get_coef(irow, jcol) * self.u.get_coef(jcol)

        return val

    def build_ss(self):

        self.a.update_numeric()
        self.b.update_numeric()
        self.c.update_numeric()
        self.d.update_numeric()
        self.u.update_numeric()

        self.ss = lti.StateSpace(self.a.numeric, self.b.numeric,
                                 self.c.numeric, self.d.numeric,
                                 u0=self.u.numeric)

    def run_to(self, dt, tstop):

        self.build_ss()
        self.ss.initialize(dt)
        return self.ss.run_to(tstop)


class FuncMatrix(object):

    def __init__(self, nrows, mcols):

        self.nrows = nrows
        self.mcols = mcols

        self.funcs = x = [[None for i in range(mcols)] for i in range(nrows)]
        self.callers = [[None for i in range(mcols)] for i in range(nrows)]
        self.is_func = [[False for i in range(mcols)] for i in range(nrows)]
        self.coefs = np.zeros((self.nrows, self.mcols))
        self.numeric = np.zeros((self.nrows, self.mcols))

    def set_cell_func(self, irow, jcol, func, caller=None):

        self.is_func[irow][jcol] = True
        self.callers[irow][jcol] = caller
        self.funcs[irow][jcol] = func

    def set_cell_coef(self, irow, jcol, coef):

        self.is_func[irow][jcol] = False
        self.coefs[irow][jcol] = coef

    def get_coef(self, irow, jcol):

        if self.is_func[irow][jcol]:
            if self.callers[irow][jcol]: 
                return self.callers[irow][jcol].funcs[irow][jcol]()
            else:
                return self.funcs[irow][jcol]()
        else:
            return self.coefs[irow][jcol]

    def update_numeric(self):

        for irow in range(self.nrows):
            for jcol in range(self.mcols):
                self.numeric[irow][jcol] = self.get_coef(irow, jcol)


class FuncVector(object):

    def __init__(self, nrows):

        self.nrows = nrows

        self.funcs = [None]*nrows
        self.callers = [None]*nrows
        self.is_func = [False]*nrows
        self.coefs = np.zeros((self.nrows, 1))
        self.numeric = np.zeros((self.nrows, 1))

    def set_cell_func(self, irow, func, caller=None):

        self.is_func[irow] = True
        self.callers[irow] = caller
        self.funcs[irow] = func
      
    def set_cell_coef(self, irow, coef):

        self.is_func[irow] = False
        self.coefs[irow] = coef

    def get_coef(self, irow):

        if self.is_func[irow]:
            if self.callers[irow]: 
                return self.callers[irow].funcs[irow]()
            else:
                return self.funcs[irow]()
        else:
            return self.coefs[irow]

    def update_numeric(self):

        for irow in range(self.nrows):
            self.numeric[irow] = self.get_coef(irow)


def test1():

    def a00(): return -1.0
    def a01(): return  1.0
    def a10(): return -1.0
    def a11(): return -1.0

    def b00(): return  1.0
    def b01(): return  0.0
    def b10(): return  0.0
    def b11(): return  1.0

    def c00(): return  1.0
    def c01(): return  0.0
    def c10(): return  0.0
    def c11(): return  1.0

    def d00(): return  0.0
    def d01(): return  0.0
    def d10(): return  0.0
    def d11(): return  0.0

    def u0():  return  1.0
    def u1():  return  1.0

    sys = FuncSystem(2, 2, 2)

    sys.a.set_cell_func(0, 0, a00)
    sys.a.set_cell_func(0, 1, a01)
    sys.a.set_cell_func(1, 0, a10)
    sys.a.set_cell_func(1, 1, a11)

    sys.b.set_cell_func(0, 0, b00)
    sys.b.set_cell_func(0, 1, b01)
    sys.b.set_cell_func(1, 0, b10)
    sys.b.set_cell_func(1, 1, b11)

    sys.c.set_cell_func(0, 0, c00)
    sys.c.set_cell_func(0, 1, c01)
    sys.c.set_cell_func(1, 0, c10)
    sys.c.set_cell_func(1, 1, c11)

    sys.d.set_cell_func(0, 0, d00)
    sys.d.set_cell_func(0, 1, d01)
    sys.d.set_cell_func(1, 0, d10)
    sys.d.set_cell_func(1, 1, d11)

    sys.u.set_cell_func(0, u0)
    sys.u.set_cell_func(1, u1)

    t, y = sys.run_to(1e-3, 10.0)

    plt.plot(t, y[0][:], label="x0")
    plt.plot(t, y[1][:], label="x1")
    plt.legend()
    plt.show()



if __name__ == "__main__":

     test1()
     #test2()
        