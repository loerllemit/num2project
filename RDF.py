from MolDyn import MolDyn
import numpy as np


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

    def get_rdf(self, dist_list):
        h, bin_edges = np.histogram(
            dist_list, density=0, bins=self.bin_num, range=(0, self.L / 2)
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
