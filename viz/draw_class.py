import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

import numpy as np
import time

class Visualizer:
    """Class for plotting states and visualizing the double pendulum on a cart in 2D

    Methods:


    """

    def __init__(self, l_1=0.5, l_2=0.5, dt=0.02):
        """Init visualizer with pole lengths

        Args:
        """
        self.l_1 = l_1
        self.l_2 = l_2
        self.dt = dt

    def simulation_viz(self, X, obs_pos=None, obs_rad=None, U=None, save_animation=False):
        """Visualizes system by utilizing a 2D matplotlib plot
        Args:
            X (ndarray): Matrix containing all states for each timesteps as columns
            tdata:
        """
        x0 = X[:, 0]
        # convert time steps into actual time in sec
        time_steps = np.arange(0, X.shape[1]*self.dt, self.dt)
        total_time = time_steps[len(time_steps)-1]
        fps = len(time_steps) / total_time
        # get init (x,y) positions
        [p_c, p_1, p_2] = self.calc_pole_end_pos(x0[0], x0[1], x0[2])

        # plot circle/obstacle if there is any
        if obs_pos is not None:
            fig, ax = plt.subplots()
            a_circle = plt.Circle((obs_pos[0], obs_pos[1]), obs_rad, color="tab:gray")
            ax.add_artist(a_circle)
        else:
            fig = plt.figure()

        # setup plot
        xmin = np.min([np.min(X[0, :]), obs_pos[0]]) -1
        xmax = np.max([np.max(X[0, :]), obs_pos[0]]) +1
        ymin = -1.5
        ymax = 1.5

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.axhline(y=0, color="tab:gray")
        plt.draw()
        # plt.grid()
        plt.pause(0.5)
        plt.show(block=False)
        plt.gca().set_aspect('equal', adjustable='box')

        # init all objects
        timer_handle = plt.text(np.max(X[0, :]) + 0.3, 1.15, '0.00s', fontsize=11);
        cart_handle, = plt.plot(p_c[0], p_c[1], 'ks', markersize=20, linewidth=0.5)
        pole_one_handle, = plt.plot([p_c[0], p_1[0]], [p_c[1], p_1[1]], color="tab:orange",
                                    linewidth=6)
        pole_two_handle, = plt.plot([p_1[0], p_2[0]], [p_1[1], p_2[1]], color="tab:orange",
                                    linewidth=6)

        joint_zero_handle, = plt.plot(p_c[0], p_c[1], 'ko', markersize=5, color="tab:gray")
        joint_one_handle, = plt.plot(p_1[0], p_1[1], 'ko', markersize=5, color="tab:gray")
        # joint_two_handle, = plt.plot(p_2[0], p_2[1], 'ko', markersize=5, color="tab:gray")

        if U is not None:
            # normalize U
            U /= np.max(U)
            inputs_arrow = plt.arrow(p_c[0], 0, dx=U[0, 0], dy=0, width=0.022, color="red", zorder=2,
                                     label="normalized input")
            plt.legend([inputs_arrow], ["normalized input"], frameon=False, loc=2)

        # iterate over all states and update objects
        writer = FFMpegWriter(fps=fps, bitrate=64000)
        with writer.saving(fig, "writer_test.mp4", 300):
            for k in range(0, X.shape[1]):
                # extract current states and retrieve new (x,y) positions of all objects
                x = X[:, k]
                [p_c, p_1, p_2] = self.calc_pole_end_pos(x[0], x[1], x[2])

                # update time
                timer_handle.set_text('{:.2f}s'.format(time_steps[k]))

                # update objects in plot
                cart_handle.set_data(x[0], 0)
                pole_one_handle.set_data([p_c[0], p_1[0]], [p_c[1], p_1[1]])
                pole_two_handle.set_data([p_1[0], p_2[0]], [p_1[1], p_2[1]])
                joint_zero_handle.set_data(p_c[0], p_c[1])
                joint_one_handle.set_data(p_1[0], p_1[1])
                # joint_two_handle.set_data(p_2[0], p_2[1])

                if U is not None and k < U.shape[1]:
                    # delete previous arrow
                    inputs_arrow.remove()
                    inputs_arrow = plt.arrow(p_c[0], 0, dx=U[0, k], dy=0, width=0.022, color="red", zorder=2,
                                             label="normalized input")
                if save_animation:
                    writer.grab_frame()
                plt.pause(0.001)
                time.sleep(0.01)




        #plt.show()
        # plt.close(fig)

    def calc_pole_end_pos(self, x_c, phi_1, phi_2):
        """Calculate start and end of position of both poles

      Args:
        x_c (float): cart position
        phi_1 (float): angle of first pole
        phi_2 (float): angle of second pole

      Returns:
        pos_c:
        pos_1:
        pos_2:
      """
        pos_c = np.array([x_c, 0])
        pos_1 = pos_c + self.l_1 * np.array([np.sin(phi_1), np.cos(phi_1)])
        pos_2 = pos_c + self.l_2 * np.array([np.sin(phi_1), np.cos(phi_1)]) + \
                self.l_1 * np.array([np.sin(phi_2), np.cos(phi_2)])
        return pos_c, pos_1, pos_2

    def plot_data(self, sim_data, input_data=None):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        fig.set_size_inches(18, 14)

        # plot cart position and velocity
        ax1.set_title("Cart position and velocity")

        ax1.plot(sim_data[0, :], label="$x_c$")
        ax1.plot(sim_data[3, :], label="$\dotx_c$")
        if input_data is not None:
            ax1.plot(input_data.T, label="$u$")
        ax1.legend()
        ax1.grid()

        # plot angle phi_1 and phi_2
        ax2.set_title("Pole angles")
        ax2.plot(sim_data[1, :], label=" $\phi_1$")
        ax2.plot(sim_data[2, :], label="$\phi_2$")

        ax2.set_ylabel("$rad$")

        ax2.grid()
        ax2.legend()

        # plot angular velocities
        ax3.set_title("Angluar velocities")
        ax3.plot(sim_data[4, :], label="$\dot\phi_1$")
        ax3.plot(sim_data[5, :], label="$\dot\phi_2$")
        ax3.set_ylabel("$rad/s$")
        ax3.set_xlabel("time")
        ax3.grid()
        ax3.legend()

        plt.show()


if __name__ == "__main__":
    # debugging
    # obstacle_2: (2, 1.3) radius 0.6
    # obstacle :  (-2, 1.5)   radius 0.8
    with open("/home/jonas/Dokumente/SS20/MPC/Code/git/viz/obstacle_states2.npy", "rb") as f:
        states = np.load(f)
    with open("/home/jonas/Dokumente/SS20/MPC/Code/git/viz/obstacle_inputs2.npy", "rb") as f:
        inputs = np.load(f)

    viz = Visualizer()
    viz.simulation_viz(states, U=inputs, obs_pos=(2, 1.3), obs_rad=0.6, save_animation=True)
