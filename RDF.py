from MolDyn import MolDyn
import numpy as np


class RDF(MolDyn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bin_num = 100 * self.box_scale  # for binning the rdf histogram

    # get all distances from unique pairs of particles
    def get_all_pair_dist(self, pos):
        dist_list = []
        for i in range(self.N):  # ith particle
            _, dist = self.get_specific_dist(pos, i)
            dist = dist[i:]
            # cut off up to largest possible distance
            dist = dist[dist <= self.L * np.sqrt(3) / 2]
            dist_list.extend(list(dist))
        return dist_list

    def get_rdf(self, dist_list):
        h, bin_edges = np.histogram(
            dist_list, density=0, bins=self.bin_num, range=(0, self.L * np.sqrt(3) / 2)
        )
        self.dr = bin_edges[1] - bin_edges[0]  # bin width
        self.bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2.0
        denominator = 4 * np.pi * self.bin_centers**2 * self.dr * self.N / self.L**3
        # twice rdf to take account min.image
        return self.bin_centers, 2 * h / denominator

    # get radial dist. func. for specific timestep
    def get_specific_rdf(self, time_snapshot):
        dist_list = self.get_all_pair_dist(self.pos_config[int(time_snapshot)])
        self.bin_centers, single_rdf_arr = self.get_rdf(dist_list)
        single_rdf_arr = single_rdf_arr / self.N

        return self.bin_centers, single_rdf_arr

    # combine multiple RDF from different time steps
    def combine_rdf(self):
        snapshots = np.linspace(self.equilibration, self.Niter, num=8, endpoint=False)
        rdf_config = np.empty([len(snapshots), self.bin_num])
        for count, t in enumerate(snapshots):
            dist_list = self.get_all_pair_dist(self.pos_config[int(t)])
            self.bin_centers, single_rdf_arr = self.get_rdf(dist_list)
            rdf_config[count] = single_rdf_arr / self.N

        x_vals = self.bin_centers
        avg = np.mean(rdf_config, axis=0)
        errors = np.std(rdf_config, axis=0, ddof=1)  # / np.sqrt(len(snapshots))
        return x_vals, avg, errors

    # get cumulative dist. func. for specific timestep
    def get_specific_cdf(self, time_snapshot):
        x_vals, avg = self.get_specific_rdf(time_snapshot)
        running_sum = (
            np.cumsum(avg * self.bin_centers**2)
            * self.dr
            * 4
            * np.pi
            * self.N
            / self.L**3
        )
        # to include particle less than sigma
        running_sum = running_sum + 1
        return x_vals, running_sum

    # combine multiple CDF from different time steps
    def combine_cdf(self):
        snapshots = np.linspace(self.equilibration, self.Niter, num=8, endpoint=False)
        cdf_config = np.empty([len(snapshots), self.bin_num])

        for count, t in enumerate(snapshots):
            x_vals, running_sum = self.get_specific_cdf(int(t))
            cdf_config[count] = running_sum

        avg = np.mean(cdf_config, axis=0)
        errors = np.std(cdf_config, axis=0, ddof=1)  # / np.sqrt(len(snapshots))
        return x_vals, avg, errors
