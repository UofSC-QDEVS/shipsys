
import os
from math import pi, sin, cos, atan2, sqrt, floor
from matplotlib import pyplot as plt

import liqss


def plot(*atoms, plot_updates=False):

    c, j = 2, 1
    r = floor(len(atoms)/2) + 1

    for atom in atoms:
    
        plt.subplot(r, c, j)
        ax1 = plt.gca()
        ax1.plot(atom.tzoh, atom.qzoh, 'b-')
        ax1.set_ylabel("{} ({})".format(atom.name, atom.units), color='b')

        if plot_updates:
            ax2 = ax1.twinx()
            ax2.plot(atom.tout, atom.nupd, 'r-')
            ax2.set_ylabel('total updates', color='r')

        plt.xlabel("t (s)")
        ax1.grid()
    
        j += 1
    
    plt.show()


def xfmr2winding():

    """
                             N=0.1 =>  10:1 (V)  1:10 (A)

               R1   L1          1:N                   R2   L2  
          .----VVV--UUU----.            .--------.----VVV--UUU---.
          |     ---->      |            |        |      ---->    |
       + ,-.      i1      /+\ v1/N     /^\       |  +    i2      |
    E1  (   )            (   )        ( | )  C1 === v1           |
         `-'              \-/     i1/N \ /       |  -            |
          |                |            |        |               |
          '----------------'            '--------'---------------' 


    E1 = i1*R1 + di1*L1 + v1/N
    i1/N = dv1*C1 + i2
    v1 = i2*R2 + di2*L2

    di1 = 1/L1 * (E1 - i1*R1 - v1/N)
    dv1 = 1/C1 * (i1/N - i2)
    di2 = 1/L2 * (v1 - i2*R2)

    """

    E1 = 10.0
    R1 = 0.05
    L1 = 0.01
    C1 = 0.01
    R2 = 10.0
    L2 = 0.01
    N = 0.1

    dq0 = 0.01

    di1 = lambda: 1/L1 * (E1 - i1.q*R1 - v1.q/N)
    dv1 = lambda: 1/C1 * (i1.q/N - i2.q)
    di2 = lambda: 1/L2 * (v1.q - i2.q * R2)

    sys = liqss.Module("shipsys", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    i1 = liqss.Atom("i1", func=di1, units="A", dq=dq0)
    v1 = liqss.Atom("v1", func=dv1, units="V", dq=dq0)
    i2 = liqss.Atom("i2", func=di2, units="A", dq=dq0)

    i1.connects(v1)
    v1.connects(i1, i2)        
    i2.connects(v1)

    sys.add_atoms(i1, v1, i2)

    sys.initialize()
    sys.run_to(10.0, verbose=True)#, fixed_dt=1.0e-4)

    plot(*sys.atoms.values(), plot_updates=True)


def xfmr3winding():

    """



             
               R1  L1       1:N1                                   
         .----VVV--UUU-----.      .------------------.
         |    --->         |      |        |         |      
      + ,-.    i1           ) || (         |         |      
    E1 (   )                ) || (         |         |
        `-'                 ) || (         |         |       
         |                 |      |        |         |        + 
         +-----------------'      '-.     _|_        <          
                                    |     ___ C1     <  G1   v1 
              R2   L2       1:N2    |      |         <          
         .----VVV--UUU-----.      .-'      |         |        - 
         |    --->         |      |        |         |
      + ,-.    12           ) || (         |         |
    E2 (   )                ) || (         |         |
        `-'                 ) || (         |         |
         |                 |      |        |         |
         +-----------------'      '--------+---------'
                     
               R1   L1          
          .----VVV--UUU----.      
          |     ---->      |      
       + ,-.      i1      /+\ v1/N1    (1:N1)
    E1  (   )  10 V      (   )    
         `-'              \-/                             
          |                |                 .--------.--------.
          '----------------'                 |        |        |
                                  i1/N1 +   /^\       |  +     |
                                  i2/N2    ( | )  C1 === v1   [ ] G1
               R2   L2                      \ /       |  -     |
          .----VVV--UUU----.                 |        |        |
          |     ---->      |                 '--------'--------' 
       + ,-.      i2      /+\ v1/N2 
    E2  (   ) 10 V       (   )        (1:N2)
         `-'              \-/      
          |                |       
          '----------------'       
  

    E1 = i1*R1 + di1*L1 + v1/N1
    E2 = i2*R2 + di2*L2 + v1/N2
    i1/N1 + i2/N2 = dv1*C1 + v1*G1

    di1 = 1/L1 * (E1 - i1*R1 - v1/N1)
    di2 = 1/L2 * (E2 - i2*R2 - v1/N2)
    dv1 = 1/C1 * (i1/N1 + i2/N2 - v1*G1)

    """

    E1 = 1.0
    R1 = 1.0
    L1 = 1.0e-2

    E2 = 1.0
    R2 = 1.0
    L2 = 1.0e-2

    C3 = 1.0e-2
    G3 = 1.0

    N1 = 0.1
    N2 = 0.1

    dq0 = 0.001

    di1 = lambda: 1/L1 * (E1 - i1.q*R1 - v3.q/n0.q)
    di2 = lambda: 1/L2 * (E2 - i2.q*R2 - v3.q/n0.q)
    dv3 = lambda: 1/C3 * (i1.q/n0.q + i2.q/n0.q - v3.q*G3)

    sys = liqss.Module("shipsys", dqmin=dq0, dqmax=dq0, dqerr=dq0)

    n0 = liqss.Atom("n0", source_type=liqss.SourceType.RAMP, x1=1.0, x2=0.01, t1=0.0, t2=1.0, dq=dq0)
    i1 = liqss.Atom("i1", func=di1, units="A", dq=dq0)
    i2 = liqss.Atom("i2", func=di2, units="A", dq=dq0)
    v3 = liqss.Atom("v3", func=dv3, units="V", dq=dq0)

    i1.connects(v3, n0)
    i2.connects(v3, n0)
    v3.connects(i1, i2, n0)        

    sys.add_atoms(i1, i2, v3, n0)

    sys.initialize()
    sys.run_to(1.0, verbose=True)

    plot(*sys.atoms.values())



if __name__ == "__main__":

    #xfmr2winding()
    xfmr3winding()