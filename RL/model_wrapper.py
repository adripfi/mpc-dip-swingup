from casadi import *
import numpy as np
import json


class ModelWrapper:
    def __init__(self, file_name_equations="equations.json", dt=0.02):
        # time step
        self.dt = dt

        # sundials solver
        self.solver = self.read_model(file_name_equations)

        # init system state (hanging down)
        self.init_state = np.zeros((6, 1))
        self.init_state[2] = np.pi

        # curr state of the system
        self.curr_state = self.init_state

        # state vector if pendulum is fully erected and stable
        self.goal = np.zeros((6, 1))

        # steps simulated
        self.steps_simulated = 0

    def step(self, action):
        # simulate system
        self.curr_state = np.asarray(self.solver(x0=self.curr_state, p=action)["xf"])

        # increase step counter
        self.steps_simulated += 1

        # reformat state vector to match format used in PILCO example code
        # original format x = (xc, xc_dot, phi1, phi1_dot, phi2, phi2_dot).T
        # PILCO format x_h = (xc, phi1, phi2, xc_dot, phi1_dot, phi2_dot).T
        obs = np.asarray([self.curr_state[0], self.curr_state[2], self.curr_state[4],
                          self.curr_state[1], self.curr_state[3], self.curr_state[5]])

        # dummy reward (gym env reward function is not used by PILCO)
        r = 1

        # check if pendulum is stable and erected e.g. check of error between current position and goal is small enough
        done = bool(np.sum(np.abs(self.curr_state - self.goal)) < 0.1)

        return obs, r, done, {}

    def reset(self):
        self.curr_state = self.init_state
        self.steps_simulated = 0

    def render(self):
        """
        Dummy function
        """
        pass

    def read_model(self, file_name):
        """
        Wrapper function for importing sympy equation in casadi.
        Second order ODE are reduced to a system of first order ODES. Afterwards a sundials solver is initialized and
        returned.
        """

        # get equation in string format
        with open(file_name) as json_file:
            sympy_sol = json.load(json_file)

        # define symbolic casadi variables used in ODEs
        x_c = SX.sym('x_c')
        phi_1 = SX.sym('phi_1')
        phi_2 = SX.sym('phi_2')
        xdot_c = SX.sym('xdot_c')
        phidot_1 = SX.sym('phidot_1')
        phidot_2 = SX.sym('phidot_2')
        xddot_c = SX.sym('xddot_c')

        x = SX.sym("x", 6)
        u = SX.sym('u', 1)

        # replace all "sin" "cos" functions in equation string with numpy functions
        for var in sympy_sol:
            sympy_sol[var] = sympy_sol[var].replace("sin", "np.sin")
            sympy_sol[var] = sympy_sol[var].replace("cos", "np.cos")

        # create casadi function for each ODE this will make the reduction of order easier
        xddot_c_cas_symb = eval(sympy_sol["xddot_c"])
        xddot_c_fun = Function("xddot_c", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [xddot_c_cas_symb])

        phiddot_1_cas_symb = eval(sympy_sol["phiddot_1"])
        phiddot_1_fun = Function("phiddot_1", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [phiddot_1_cas_symb])

        phiddot_2_cas_symb = eval(sympy_sol["phiddot_2"])
        phiddot_2_fun = Function("phiddot_2", [phi_1, phi_2, xdot_c, phidot_1, phidot_2, u], [phiddot_2_cas_symb])

        # create system of first order ODEs
        xdot = np.array([[x[1]],  # xdot_1
                         [xddot_c_fun(x[2], x[4], x[1], x[3], x[5], u)],  # xdot_2
                         [x[3]],  # xdot_3
                         [phiddot_1_fun(x[2], x[4], x[1], x[3], x[5], u)],  # xdot_3
                         [x[5]],  # xdot_4
                         [phiddot_2_fun(x[2], x[4], x[1], x[3], x[5], u)]])  # xdot_4

        # dict for options and ode needed for solver object
        ode = {'x': x, 'ode': xdot, 'p': u}
        opts = {'tf': self.dt}

        # return solver object
        return integrator('F', 'idas', ode, opts)


if __name__ == "__main__":
    obj = ModelWrapper()



