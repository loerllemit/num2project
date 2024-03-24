# %%
import numpy as np
import scipy.stats as stats


class MolDyn:
    def __init__(
        self,
        temp=94,
        box_scale=1,
        rcut_scale=1,
        thermo_rate=10,
        equilibration=200,
        Niter=1500,
    ):
        self.Kb = 1.380649e-23  # in J/K
        self.mass = 39.95 * 1.6747e-24 / 1000  # in kilograms
        self.sigma = 3.4e-10  # in meters
        self.eps = 120 * self.Kb  # in J
        self.del_t = 1e-14  # time difference in s
        self.box_scale = box_scale
        self.rcut_scale = rcut_scale
        self.L = 10.229 * self.sigma * self.box_scale  # box length
        self.Rcut = 2.25 * self.sigma * self.rcut_scale  # cutoff radius
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

    def init_pos1(self):
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

    def init_pos(self):
        lat_const = self.L / self.box_scale
        xyz = np.genfromtxt("crystal_structure/Ar.vasp", skip_header=8)
        return xyz * lat_const

    def init_vel1(self):
        # for maxwell-boltzmann dist
        self.std_dev = np.sqrt(self.Kb * self.Temp * 1.0 / self.mass)
        return self.std_dev * np.random.randn(self.N, 3)

    def init_vel2(self):
        # for maxwell-boltzmann dist
        self.std_dev = np.sqrt(self.Kb * self.Temp * 1.0 / self.mass)
        # truncated dist. to remove ~5% ultrafast velocities
        vel = stats.truncnorm(
            -1 * self.std_dev, 1 * self.std_dev, loc=0, scale=self.std_dev
        )

        return vel.rvs([self.N, 3])

    def init_vel(self):
        # self.std_dev = np.sqrt(self.Kb * self.Temp  / self.mass)
        return abs(-1 + 2 * np.random.rand(self.N, 3))

    def init_vel3(self):
        vel0_mag = np.sqrt(3 * self.Kb * self.Temp / self.mass)
        vel = np.random.randn(self.N, 3)
        vel_norm = np.linalg.norm(vel, ord=2, axis=1)
        vel = vel0_mag * np.random.randn(self.N, 3) / vel_norm[:, None]
        return vel

    # Maxwell-Boltzmann Distribution
    def vel_MB(self, v):
        self.std_dev = np.sqrt(self.Kb * self.Temp / self.mass)
        return (
            1
            / (2 * np.pi * self.std_dev**2) ** 0.5
            * np.exp(-(v**2) / (2 * self.std_dev**2))
        )

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
            f"data/positions_T{self.Temp}_L{self.box_scale}_Rc{self.rcut_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
            self.pos_config,
        )
        # save velocities for all timesteps
        np.save(
            f"data/velocities_T{self.Temp}_L{self.box_scale}_Rc{self.rcut_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
            self.vel_config,
        )
        # save temperatures for all timesteps
        np.save(
            f"data/temperatures_T{self.Temp}_L{self.box_scale}_Rc{self.rcut_scale}_Tr{self.thermo_rate}_Eq{self.equilibration}_step{self.Niter}.npy",
            self.temp_arr,
        )
