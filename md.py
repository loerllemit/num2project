# %%
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


class MolDyn:
    def __init__(self):
        self.mass = 39.95 * 1.6747e-24 / 1000  # in kilograms
        self.sigma = 3.4e-10  # in meters
        self.Kb = 1.380649e-23  # in J/K
        self.Temp = 94  # in K
        self.eps = 120 * self.Kb  # in J
        self.del_t = 1e-14  # time difference in s
        self.Rcut = 2.25 * self.sigma  # cutoff radius
        self.N = 864  # 864  # number of particles
        self.Niter = 2000  # total # of time steps
        self.L = 10.229 * self.sigma  # box length
        self.pos_config = np.empty([self.Niter, self.N, 3])
        self.vel_config = np.empty([self.Niter, self.N, 3])
        self.acc_config = np.empty(
            [2, self.N, 3]
        )  # 0th index for prev and 1st index for next
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
        T_curr = self.get_temp(vel)
        print(T_curr)
        vel = vel * np.sqrt(self.Temp / T_curr)
        return vel

    def len_jones(self, r):
        ratio = self.sigma / r
        return 4 * self.eps * (ratio**12 - ratio**6)

    def min_image(self, vec_r):  # vec_r = [x,y,z]
        return vec_r - np.round(vec_r / self.L) * self.L

    def get_acc(self, pos):
        acc = np.empty([self.N, 3])  # matrix containing acceleration for each particle
        for i in range(self.N):  # ith particle
            pos_diff = pos[i] - pos
            # remove self interaction (j=i)
            pos_diff = np.delete(pos_diff, i, 0)
            pos_diff = self.min_image(pos_diff)
            dist = np.linalg.norm(
                pos_diff, ord=2, axis=1
            )  # distance between ith particle and others

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

    def update_vel(self, vel):
        vel = vel + 0.5 * self.del_t * (self.acc_config[0] + self.acc_config[1])
        return self.vel_scaling(vel)

    def main(self):
        self.pos_config[0] = self.pos
        self.vel_config[0] = self.vel
        self.acc_config[0] = self.get_acc(self.pos)

        for t in range(1, self.Niter):  # specific timestep
            self.pos_config[t] = self.update_pos(
                self.pos_config[t - 1], self.vel_config[t - 1], self.acc_config[0]
            )
            self.acc_config[1] = self.get_acc(self.pos_config[t])
            self.vel_config[t] = self.update_vel(self.vel_config[t - 1])
            self.acc_config[0] = self.acc_config[1]


ins = MolDyn()
ins.main()
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
