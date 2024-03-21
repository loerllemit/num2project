# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import glob
from PIL import Image


class MolDyn:
    def __init__(
        self, temp=94, box_scale=1, thermo_rate=10, equilibration=200, Niter=1500
    ):
        self.Kb = 1.380649e-23  # in J/K
        self.mass = 39.95 * 1.6747e-24 / 1000  # in kilograms
        self.sigma = 3.4e-10  # in meters
        self.eps = 120 * self.Kb  # in J
        self.del_t = 1e-14  # time difference in s
        self.Rcut = 2.25 * self.sigma  # cutoff radius
        self.box_scale = box_scale
        self.L = 10.229 * self.sigma * self.box_scale  # box length
        self.Temp = temp  # in K
        self.N = 864  # number of particles
        self.Niter = Niter  # total # of time steps
        self.equilibration = (
            equilibration  # equilibrate simulation for given # timesteps
        )
        self.thermo_rate = thermo_rate  # apply thermostat at every specified intervals
        self.pos_config = np.empty([self.Niter, self.N, 3])
        self.vel_config = np.empty([self.Niter, self.N, 3])
        self.acc_config = np.empty(
            [2, self.N, 3]
        )  # 0th index for prev and 1st index for next
        self.temp_arr = np.empty(self.Niter)
        self.pos = self.init_pos()
        self.vel = self.init_vel()

    def init_pos(self):
        # create equally spaced points in the box
        spacing = 0.8 * self.sigma
        x = np.linspace(
            self.L / 2 - spacing * 4, self.L / 2 + spacing * 4, 8, endpoint=False
        )
        y = np.linspace(
            self.L / 2 - spacing * 4, self.L / 2 + spacing * 5, 9, endpoint=False
        )
        z = np.linspace(
            self.L / 2 - spacing * 6, self.L / 2 + spacing * 6, 12, endpoint=False
        )

        xyz = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T
        return xyz

    def init_vel(self):
        # for maxwell-boltzmann dist
        self.std_dev = np.sqrt(
            self.Kb * self.Temp * 1.0 / self.mass
        )  # use higher temp to thermalize faster
        return self.std_dev * np.random.randn(self.N, 3)

    def get_temp(self, vel):
        vel_squared = np.linalg.norm(vel, ord=2, axis=1) ** 2
        mean_vel_squared = np.mean(vel_squared)
        return self.mass * mean_vel_squared / (3 * self.Kb)

    def vel_scaling(self, vel):
        vel = vel * np.sqrt(self.Temp / self.T_curr)
        return vel

    def len_jones(self, r):
        ratio = self.sigma / r
        return 4 * self.eps * (ratio**12 - ratio**6)

    def min_image(self, vec_r):  # vec_r = [x,y,z]
        return vec_r - np.round(vec_r / self.L) * self.L

    # get displacements and distances for a given ith particle
    def get_specific_dist(self, pos, index):
        pos_diff = pos[index] - pos
        # remove self interaction
        pos_diff = np.delete(pos_diff, index, 0)
        pos_diff = self.min_image(pos_diff)
        dist = np.linalg.norm(
            pos_diff, ord=2, axis=1
        )  # distance between ith particle and others
        return pos_diff, dist

    def get_acc(self, pos):
        acc = np.empty([self.N, 3])  # matrix containing acceleration for each particle
        for i in range(self.N):  # ith particle
            pos_diff, dist = self.get_specific_dist(pos, i)
            # include only within cutoff
            pos_diff = pos_diff[dist <= self.Rcut]
            dist = dist[dist <= self.Rcut]

            sums = np.zeros([1, 3])
            for j in range(len(dist)):
                ratio = self.sigma / dist[j]
                # potential using lennard jones
                sums += pos_diff[j] * (2 * ratio**12 - ratio**6) / dist[j] ** 2

            acc[i] = 24 * self.eps * sums / self.mass
        return acc

    def update_pos(self, pos, vel, acc):
        pos = pos + vel * self.del_t + 0.5 * acc * self.del_t**2
        return pos % self.L  # apply periodic BCs

    def update_vel(self, vel, timestep):
        vel = vel + 0.5 * self.del_t * (self.acc_config[0] + self.acc_config[1])
        # apply thermostat at each given interval and within equilibration
        if timestep % self.thermo_rate == 0:  # and timestep <= self.equilibration:
            return self.vel_scaling(vel)
        return vel

    def run_md(self):
        self.pos_config[0] = self.pos
        self.vel_config[0] = self.vel
        self.acc_config[0] = self.get_acc(self.pos)
        self.T_curr = self.get_temp(self.vel_config[0])
        self.temp_arr[0] = self.T_curr
        print(f"Step: {0}")
        print(f"T={self.T_curr}")

        for t in range(1, self.Niter):  # specific timestep
            self.pos_config[t] = self.update_pos(
                self.pos_config[t - 1], self.vel_config[t - 1], self.acc_config[0]
            )
            self.acc_config[1] = self.get_acc(self.pos_config[t])
            self.vel_config[t] = self.update_vel(self.vel_config[t - 1], t)
            self.acc_config[0] = self.acc_config[1]

            self.T_curr = self.get_temp(self.vel_config[t])
            self.temp_arr[t] = self.T_curr
            print(f"Step: {t}")
            print(f"T={self.T_curr}")

        # save positions for all timesteps
        np.save(
            f"data/positions_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
            self.pos_config,
        )
        # save temperatures for all timesteps
        np.save(
            f"data/temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
            self.temp_arr,
        )


class RDF(MolDyn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bin_num = 100

    def get_all_pair_dist(self, pos):
        dist_list = []
        for i in range(self.N):  # ith particle
            _, dist = self.get_specific_dist(pos, i)
            dist = dist[i:]
            # cut off up  to L/2
            dist = dist[dist <= self.L * np.sqrt(3) / 2]
            dist_list.extend(list(dist))
        return dist_list

    def get_rdf(self, dist_list):
        h, bin_edges = np.histogram(
            dist_list, density=0, bins=self.bin_num, range=(0, self.L * np.sqrt(3) / 2)
        )
        dr = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        denominator = 4 * np.pi * bin_centers**2 * dr * self.N / self.L**3
        return bin_centers, h / denominator

    def combine_rdf(self):
        snapshots = range(-1, -self.Niter // 2, -10)
        rdf_config = np.empty([len(snapshots), self.bin_num])
        for count, t in enumerate(snapshots):
            dist_list = self.get_all_pair_dist(self.pos_config[t])
            bin_centers, single_rdf_arr = self.get_rdf(dist_list)
            rdf_config[count] = single_rdf_arr

        x_vals = bin_centers / self.sigma
        avg = np.mean(rdf_config, axis=0)
        errors = np.std(rdf_config, axis=0, ddof=1)  # / np.sqrt(len(snapshots))
        return x_vals, avg, errors


# %%


class Plots(RDF):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_temp_arr(self):
        self.temp_arr = np.load(
            f"data/temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy"
        )

    def load_pos_config(self):
        self.pos_config = np.load(
            f"data/positions_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
        )

    def plot_temp(self, start_time, fig, ax):
        self.load_temp_arr()
        ax.plot(range(start_time, self.Niter), self.temp_arr[start_time:])
        plt.hlines(
            np.mean(self.temp_arr[start_time:]),
            start_time,
            self.Niter,
            linestyles="dotted",
        )
        np.mean(self.temp_arr[start_time:])
        ax.set_xlabel(r"timestep (t/$\Delta$t)")
        ax.set_ylabel("Temperature (K)")
        fig.savefig(
            f"temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.pdf",
            bbox_inches="tight",
        )

    def plot_rdf(
        self,
        label,
        fig,
        ax,
        fmt="o-",
    ):
        self.load_pos_config()
        x_vals, avg, errors = self.combine_rdf()
        ax.errorbar(
            x_vals, avg, yerr=errors, markersize=4, fmt=fmt, capsize=3.5, label=label
        )
        ax.set_ylabel("radial distribution function", fontsize=15)
        ax.set_xlabel(r"r/$\sigma$", fontsize=15)
        ax.set_xlim(0, 5)
        ax.set_ylim(0)
        ax.legend()

    def make_animation(self, frame_folder):
        self.load_pos_config()
        ax = plt.figure().add_subplot(projection="3d")
        plt.subplots_adjust(right=1, top=1, left=0, bottom=0)
        for t in range(0, self.Niter):
            ax.clear()
            ax.set_xlabel(r"x ($\AA$)")
            ax.set_ylabel(r"y ($\AA$)")
            ax.set_zlabel(r"z ($\AA$)")
            ax.set_xlim(0, self.L / 1e-10)
            ax.set_ylim(0, self.L / 1e-10)
            ax.set_zlim(0, self.L / 1e-10)
            ax.scatter(
                self.pos_config[t, :, 0] / 1e-10,
                self.pos_config[t, :, 1] / 1e-10,
                self.pos_config[t, :, 2] / 1e-10,
            )
            plt.savefig(f"{frame_folder}/{t}.png", dpi=100)

        frames = [
            Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))
        ]
        frame_one = frames[0]
        frame_one.save(
            f"animate_T{self.Temp}_L{self.box_scale}.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=10,
            loop=1,
        )
