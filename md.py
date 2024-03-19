# %%
import numpy as np
import matplotlib.pyplot as plt


class MolDyn:
    def __init__(self, temp=94, box_scale=1, thermo_rate=10, equilibration=200):
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
        self.Niter = 1500  # total # of time steps
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
        if timestep % self.thermo_rate == 0:  # apply thermostat at each given interval
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
            f"positions_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}.npy",
            self.pos_config,
        )

        np.save(
            f"temperatures_T{self.Temp}_L{self.box_scale}_Tr{self.thermo_rate}.npy",
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
            dist = dist[dist <= self.L / 2]
            dist_list.extend(list(dist))
        return dist_list

    def get_gofr(self, dist_list):
        h, bin_edges = np.histogram(
            dist_list, density=0, bins=self.bin_num, range=(0, self.L / 2)
        )
        dr = bin_edges[1] - bin_edges[0]
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        denominator = 4 * np.pi * bin_centers**2 * dr * self.N / self.L**3
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
        fig.savefig(f"T{self.Temp}.pdf", bbox_inches="tight")

    def combine_gofr(self):
        snapshots = range(-1, -200, -10)
        gofr_config = np.empty([len(snapshots), self.bin_num])
        for count, t in enumerate(snapshots):
            dist_list = self.get_all_pair_dist(self.pos_config[t])
            bin_centers, single_gofr_arr = self.get_gofr(dist_list)
            gofr_config[count] = single_gofr_arr

        x_vals = bin_centers / self.sigma
        avg = np.mean(gofr_config, axis=0)
        errors = np.std(gofr_config, axis=0, ddof=1)  # / np.sqrt(len(snapshots))
        return x_vals, avg, errors

        # self.plot_gofr(bin_centers / self.sigma, avg, errors)


box_scale = 1
# %%
ins_50 = RDF(temp=50, box_scale=box_scale)
ins_50.run_md()
# %%
ins_94 = RDF(temp=94, box_scale=box_scale)
ins_94.run_md()
# %%
ins_400 = RDF(temp=400, box_scale=box_scale)
ins_400.run_md()

# %%
fig, ax = plt.subplots()

x_vals, avg, errors = ins_50.combine_gofr()
ax.errorbar(
    x_vals, avg, yerr=errors, markersize=4, fmt="o-", capsize=3.5, label="T=50 K"
)
x_vals, avg, errors = ins_94.combine_gofr()
ax.errorbar(
    x_vals, avg, yerr=errors, markersize=4, fmt="s-", capsize=3.5, label="T=94 K"
)
x_vals, avg, errors = ins_400.combine_gofr()
ax.errorbar(
    x_vals, avg, yerr=errors, markersize=4, fmt="^-", capsize=3.5, label="T=400 K"
)
ax.set_ylabel("radial distribution function", fontsize=15)
ax.set_xlabel(r"r/$\sigma$", fontsize=15)
ax.set_xlim(0, 5)
ax.set_ylim(0)
ax.legend()

fig.savefig(f"rdf_combined.pdf", bbox_inches="tight")

# %%
# %matplotlib qt
ax = plt.figure().add_subplot(projection="3d")
plt.subplots_adjust(right=1, top=1, left=0, bottom=0)

ins = ins_400
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
    plt.pause(0.03)
plt.show()

# %% temp equilibrium
ins = ins_94
plt.plot(ins.temp_arr[00:])
plt.hlines(np.mean(ins.temp_arr[200:]), 0, ins.Niter, linestyles="dotted")
np.mean(ins.temp_arr[150:])
plt.xlabel(r"timestep (t/$\Delta$t)")
plt.ylabel("Temperature (K)")
plt.savefig(f"temp_eq_10rate.pdf", bbox_inches="tight")

# %% temp rms deviation
temp_dat = ins.temp_arr[200 : ins.Niter]
np.sqrt(np.mean(temp_dat**2) - np.mean(temp_dat) ** 2) / np.mean(temp_dat)

# %%
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

fig, ax = plt.subplots()
h, bin_edges, patches = ax.hist(
    dist,
    density=0,
    bins=30,
    # histtype="step",
    # label="T=100 K",
    # color=colors[0],
    linewidth=2.0,
)
h, bin_edges = np.histogram(
    dist,
    density=0,
    bins=30,
)
dr = bin_edges[1] - bin_edges[0]
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
denominator = 4 * np.pi * bin_centers**2 * dr * ins.N / ins.L**3
gofr = h / denominator

ax.vlines(ins.L / 2, 0, 60, color="red")
# %%

fig, ax = plt.subplots()
ax.plot(bin_centers, gofr, marker="o")


# %%
