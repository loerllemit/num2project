import numpy as np


class RDF:
    def __init__(self, **kwargs):
        self.bin_num = 100
        self.N = 864  # number of particles
        self.L = 30  # box length in angstrom
        self.equilibration = 500  # equilibrate simulation for given # timesteps
        self.Niter = 1000  # total # of time steps

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

    def get_all_pair_dist(self, pos):
        dist_list = []
        for i in range(self.N):  # ith particle
            _, dist = self.get_specific_dist(pos, i)
            dist = dist[i:]
            # cut off up  to largest possible distance
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
        snapshots = np.linspace(self.equilibration, self.Niter, num=8, endpoint=False)
        rdf_config = np.empty([len(snapshots), self.bin_num])
        for count, t in enumerate(snapshots):
            dist_list = self.get_all_pair_dist(self.pos_config[int(t)])
            bin_centers, single_rdf_arr = self.get_rdf(dist_list)
            rdf_config[count] = single_rdf_arr / self.N

        x_vals = bin_centers
        avg = np.mean(rdf_config, axis=0)
        errors = np.std(rdf_config, axis=0, ddof=1) / np.sqrt(len(snapshots))
        return x_vals, avg, errors
