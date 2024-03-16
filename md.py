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
        self.del_t = 1e-12  # time difference in s
        self.Rcut = 2.25 * self.sigma  # cutoff radius
        self.N = 164  # 864  # number of particles
        self.Niter = 100  # total # of time steps
        self.L = 10.229 * self.sigma  # box length
        self.pos_config = np.empty([self.Niter, self.N, 3])
        self.vel_config = np.empty([self.Niter, self.N, 3])
        self.pos = np.random.rand(self.N, 3) * self.L
        self.vel = self.init_vel()
        # self.vel = np.zeros([self.N, 3])

    def init_vel(self):
        R = np.random.rand(self.N, 3) - 0.5
        return R * np.sqrt(3 * self.Kb * self.Temp / self.mass)

    def vel_scaling(self, v_vec):
        vel_squared = np.linalg.norm(v_vec, ord=2, axis=1) ** 2
        mean_vel_squared = np.mean(vel_squared)
        T_curr = self.mass * mean_vel_squared / (3 * self.Kb)
        return v_vec * np.sqrt(self.Temp / T_curr)

    def len_jones(self, r):
        ratio = self.sigma / r
        return 4 * self.eps * (ratio**12 - ratio**6)

    def min_image(self, vec_r):  # vec_r = [x,y,z]
        return vec_r - np.round(vec_r / self.L) * self.L

    def fr(self, v_vec):  # time derivative of pos
        return v_vec

    def fv(self, pos_diff):  # time derivative of velocity
        self.dist = np.linalg.norm(
            pos_diff, ord=2, axis=1
        )  # distance between pairs for a given particle
        sums = np.zeros([1, 3])
        for j in range(len(self.dist)):
            if self.dist[j] != 0 and self.dist[j] <= self.Rcut:
                ratio = self.sigma / self.dist[j]
                sums += pos_diff[j] * (2 * ratio**12 - ratio**6) / self.dist[j] ** 2
        return 24 * self.eps * sums / self.mass

    def euler(self, r_vec, v_vec, pos_diff):
        r_vec = r_vec + self.fr(v_vec) * self.del_t
        v_vec = v_vec + self.fv(pos_diff) * self.del_t

        return r_vec, v_vec

    def main(self):
        for t in range(self.Niter):  # specific timestep
            for i in range(self.N):  # ith particle
                pos_diff = self.pos - self.pos[i]
                pos_diff = self.min_image(pos_diff)
                rvec, vvec = self.euler(self.pos[i], self.vel[i], pos_diff)
                self.pos[i] = rvec % self.L  # update position on ith particle
                self.vel[i] = self.vel_scaling(vvec)  # update velocity on ith particle
            self.pos_config[t] = self.pos
            self.vel_config[t] = self.vel


ins = MolDyn()
ins.main()
# %%
# %matplotlib qt

ax = plt.figure().add_subplot(projection="3d")

for t in range(ins.Niter):
    ax.clear()
    ax.set_xlim(0, ins.L)
    ax.set_ylim(0, ins.L)
    ax.set_zlim(0, ins.L)
    ax.scatter(
        ins.pos_config[t, :, 0],
        ins.pos_config[t, :, 1],
        ins.pos_config[t, :, 2],
    )
    plt.pause(0.5)
plt.show()

# %%
