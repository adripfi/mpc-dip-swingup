from casadi import *
import matplotlib.pyplot as plt
import numpy as np
import timeit
import sys


def derive(x, u):
    
    m = 0.6  
    m1 = 0.2
    m2 = 0.2
    l1 = 0.25 
    l2 = 0.25
    g = 9.81  
    L1 = 2 * l1 
    L2 = 2 * l2
    J1 = m1 * L1 ** 2 / 12 
    J2 = m2 * L2 ** 2 / 12
   

    # Helper variables
    h1 = m + m1 + m2
    h2 = m1 * l1 + m2 * L1
    h3 = m2 * l2
    h4 = m1 * l1 ** 2 + m2 * L1 ** 1 + J1
    h5 = m2 * l2 * L1
    h6 = m2 * l2 ** 2 + J2
    h7 = m1 * l1 * g + m2 * L1 * g
    h8 = m2 * l2 * g

       
    # inertia matrix
    M = np.array([[h1, h2 * np.cos(x[1]), h3 * np.cos(x[2])], [h2 * np.cos(x[1]), h4, h5 * np.cos(x[1] - x[2])], [h3 * np.cos(x[2]), h5 * np.cos(x[1] - x[2]), h6] ])
    
    G = np.array([0, -h7 * np.sin(x[1]), -h8 * np.sin(x[2])])

    # Coriolis and centrifugal vector
    C = np.array([[0, -h2 * np.sin(x[1]) * x[4], -h3 * x[5] * np.sin(x[2])], [0, 0, h5 * x[5] * np.sin(x[1] - x[2])], [0, -h5 * x[4] * np.sin(x[1] - x[2]), 0]])

    Q = np.array([u, 0, 0])


    # Create state space
    M_invers = solve(M, SX.eye(M.shape[0]))
    q_dot = C @ x[3:6]
    q_dotdot = M_invers @ (Q.T - q_dot - G.T)

    # Create function
    x_dot = SX.sym("x_dot", 6, 1)
    x_dot[0] = x[3]
    x_dot[1] = x[4]
    x_dot[2] = x[5]
    x_dot[3:6] = q_dotdot

    return x_dot



if __name__ == "__main__":
    x = SX.sym("x", 6)
    u = SX.sym("u", 1)
    x_dot = derive(x, u)
    print(x_dot)
