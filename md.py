# %%
import numpy as np
import matplotlib.pyplot as plt


class MolDyn:
    def __init__(self):
        self.mass = 39.95 * 1.6747e-24 / 1000  # in kilograms
        self.sigma = 3.4e-10  # in meters
        self.Kb = 1.380649e-23  # in J/K
        self.Temp = 94.4  # in K
        self.eps = 120 * self.Kb  # in J
        self.del_t = 1e-14  # time difference in s
        self.Rcut = 2.25 * self.sigma  # cutoff radius
        self.N = 864  # 864  # number of particles
        self.Niter = 1000  # total # of time steps
        self.L = 10.229 * self.sigma  # box length
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
        xyz = (
            np.mgrid[
                0 : self.L : self.L / 8,
                0 : self.L : self.L / 9,
                0 : self.L : self.L / 12,
            ]
            .reshape(3, -1)
            .T
        )
        return xyz[: self.N]

    def init_vel(self):
        # for maxwell-boltzmann dist
        self.std_dev = np.sqrt(
            self.Kb * self.Temp * 1.2 / self.mass
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
        if timestep < 100:  # for thermalization/warmup
            return self.vel_scaling(vel)
        return vel

    def main(self):
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


class RDF(MolDyn):
    def __init__(self):
        super().__init__()
        self.bin_num = 60

    def get_all_pair_dist(self, pos):
        dist_list = []
        for i in range(self.N):  # ith particle
            _, dist = self.get_specific_dist(pos, i)
            dist = dist[i:]
            # cut off up  to L/2
            dist = dist[dist <= self.L / 2]
            dist_list.extend(list(dist))
        return dist_list

    def get_gofr(self, dist_list):
        h, bin_edges = np.histogram(
            dist_list, density=0, bins=self.bin_num, range=(0, self.L / 2)
        )
        dr = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        denominator = 4 * np.pi * bin_centers**2 * dr * ins.N / ins.L**3
        return bin_centers, h / denominator

    def plot_gofr(self, x_vals, avg, errors):
        fig, ax = plt.subplots()
        ax.errorbar(
            x_vals,
            avg,
            yerr=errors,
            markersize=4,
            fmt="o-",
            capsize=3.5,
        )
        ax.set_ylabel("radial distribution function", fontsize=15)
        ax.set_xlabel(r"r/$\sigma$", fontsize=15)
        ax.set_xlim(0, 5)
        ax.set_ylim(0)

    def combine_gofr(self):
        snapshots = range(self.Niter - 50, self.Niter)
        gofr_config = np.empty([len(snapshots), self.bin_num])
        for count, t in enumerate(snapshots):
            dist_list = self.get_all_pair_dist(self.pos_config[t])
            bin_centers, single_gofr_arr = self.get_gofr(dist_list)
            gofr_config[count] = single_gofr_arr

        avg = np.mean(gofr_config, axis=0)
        errors = np.std(gofr_config, axis=0, ddof=1) / np.sqrt(len(snapshots))

        self.plot_gofr(bin_centers / self.sigma, avg, errors)


ins = RDF()
ins.main()
ins.combine_gofr()

# %%
# %matplotlib qt
ax = plt.figure().add_subplot(projection="3d")
plt.subplots_adjust(right=1, top=1, left=0, bottom=0)

for t in range(ins.Niter):
    ax.clear()
    ax.set_xlim(0, ins.L)
    ax.set_ylim(0, ins.L)
    ax.set_zlim(0, ins.L)
    ax.scatter(
        ins.pos_config[t, :, 0][:],
        ins.pos_config[t, :, 1][:],
        ins.pos_config[t, :, 2][:],
    )
    plt.pause(0.05)
plt.show()

# %%

plt.plot(ins.temp_arr[0:])
plt.hlines(np.mean(ins.temp_arr[100:]), 0, ins.Niter, linestyles="dotted")
np.mean(ins.temp_arr[100:])

# %%
temp_dat = ins.temp_arr[200:1300]
np.sqrt(np.mean(temp_dat**2) - np.mean(temp_dat) ** 2) / np.mean(temp_dat)
# %%
"""
# rdf
pos = ins.pos_config[-1]
i = ins.N // 2
pos_diff = pos[i] - pos
# remove self interaction (j=i)
pos_diff = np.delete(pos_diff, i, 0)
pos_diff = ins.min_image(pos_diff)
dist = np.linalg.norm(
    pos_diff, ord=2, axis=1
)  # distance between ith particle and others

# cut up to L/2
dist = dist[dist <= ins.L / 2]

# fig, ax = plt.subplots()
# h, bin_edges, patches = ax.hist(
#     dist,
#     density=0,
#     bins=30,
#     # histtype="step",
#     # label="T=100 K",
#     # color=colors[0],
#     linewidth=2.0,
# )
h, bin_edges = np.histogram(
    dist,
    density=0,
    bins=30,
)
dr = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
denominator = 4 * np.pi * bin_centers**2 * dr * ins.N / ins.L**3
gofr = h / denominator

# ax.vlines(ins.L / 2, 0, 60, color="red")
# %%

fig, ax = plt.subplots()
ax.plot(bin_centers, gofr, marker="o")

"""
