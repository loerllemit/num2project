# %%
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


class MolDyn:
    def __init__(self):
        self.mass = 39.95 * 1.6747e-24 / 1000  # in kilograms
        self.sigma = 3.4e-10  # in meters
        self.Kb = 1.380649e-23  # in J/K
        self.eps = 120 * self.Kb  # in J
        self.del_t = 1e-12  # time difference in s
        self.R = 2.25 * self.sigma  # cutoff radius
        self.N = 10  # 864  # number of particles
        self.Niter = 100  # total # of time steps
        self.L = 10.229 * self.sigma  # box length
        self.pos_config = np.empty([self.Niter, self.N, 3])
        self.vel_config = np.empty([self.Niter, self.N, 3])
        self.vel = np.zeros([self.N, 3])
        self.pos = np.random.rand(self.N, 3) * self.L

    def len_jones(self, r):
        ratio = self.sigma / r
        return 4 * self.eps * (ratio**12 - ratio**6)

    def pbc(self, vec_r):  # vec_r = [x,y,z]
        return vec_r - np.round(vec_r / self.L)

    def fr(self, v_vec):  # time derivative of pos
        return v_vec

    def fv(self, pos_diff):  # time derivative of velocity
        dist = np.linalg.norm(
            pos_diff, ord=2, axis=1
        )  # distance between pairs for a given particle
        sums = np.zeros([1, 3])
        for j in range(len(dist)):
            if dist[j] != 0:
                ratio = self.sigma / dist[j]
                sums += pos_diff[j] * (2 * ratio**12 - ratio**6) / dist[j] ** 2
        return 24 * self.eps * sums / self.mass

    def euler(self, r_vec, v_vec, pos_diff):
        r_vec = r_vec + self.fr(v_vec) * self.del_t
        v_vec = v_vec + self.fv(pos_diff) * self.del_t

        return r_vec, v_vec

    def main(self):
        for t in range(self.Niter):
            for i in range(self.N):  # ith particle
                pos_diff = self.pos - self.pos[i]
                pos_diff = self.pbc(pos_diff)
                rvec, vvec = self.euler(self.pos[i], self.vel[i], pos_diff)
                self.pos[i] = rvec  # update position on ith particle
                self.vel[i] = vvec  # update velocity on ith particle
            self.pos_config[t] = self.pos
            self.vel_config[t] = self.vel


ins = MolDyn()
ins.main()
# %%
# %matplotlib qt

ax = plt.figure().add_subplot(projection="3d")
for t in range(10):
    ax.scatter(
        ins.pos_config[t, :, 0], ins.pos_config[t, :, 1], ins.pos_config[t, :, 2]
    )
    # ax.clear()
    sleep(1)
plt.show()

# %%
