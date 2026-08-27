import arch
import scipy
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

from ._functions import self_tuning_spectral_clustering


class Vol_Regime:
    def __init__(self, data):
        self.data = data
        self.vol = None
        self.attr = None
        self.changepoints = None
        self.clusters = None
        self.conversion_dict = None
        self.regime_labels = None

    def get_vol(self):
        real_am = arch.arch_model(self.data)
        real_am._adjust_sample(None, None)
        resids = np.asarray(real_am.resids(real_am.starting_values()), dtype=float)
        scale = resids.var()
        rescale = 1

        while not 0.1 <= scale < 10_000.0 and scale > 0:
            if scale < 1.0:
                rescale *= 10
            else:
                rescale /= 10
            scale = scale * rescale**2

        real_am = arch.arch_model(self.data * rescale)
        real_res = real_am.fit(update_freq=0, disp=0)
        vol = real_res.conditional_volatility
        self.vol = vol

    def A_local(self, pd):
        dim = pd.shape[0]
        dist_ = scipy.spatial.distance.pdist(pd)
        sigmas = np.zeros(dim)
        for i in range(len(pd)):
            sigmas[i] = sorted(pd[i])[7]

        A = np.zeros([dim, dim])
        dist = iter(dist_)
        for i in range(dim):
            for j in range(i+ 1, dim):
                d = np.exp(-1 * next(dist)**2 / (sigmas[i] * sigmas[j]))
                A[i, j] = d
                A[j, i] = d
        
        return A
    
    def get_attr(self):
        graph = np.zeros(shape=(len(self.changepoints), len(self.changepoints)))
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if j >= i:
                    if (i == 0) and (j == 0):
                        dist = scipy.stats.wasserstein_distance(
                            self.data[:self.changepoints[i]],
                            self.data[:self.changepoints[j]]
                        )
                    elif i == 0:
                        dist = scipy.stats.wasserstein_distance(
                            self.data[:self.changepoints[i]],
                            self.data[self.changepoints[j-1] : self.changepoints[j]]
                        )
                    else:
                        dist = scipy.stats.wasserstein_distance(
                            self.data[self.changepoints[i-1] : self.changepoints[i]],
                            self.data[self.changepoints[j-1] : self.changepoints[j]]
                        )
                    
                    graph[i, j] = dist
                    graph[j, i] = dist

        self.attr = self.A_local(graph)

    def get_changepoints(self, pen=10):
        algo = rpt.Pelt(model='rbf').fit(self.vol)
        cp = algo.predict(pen=pen)
        self.changepoints = cp

    def assign_clusters(self, max_clusters=None):
        cp = self.changepoints.copy()
        cp[:0] = [0]
        
        clusters = self_tuning_spectral_clustering(
            self.attr, max_n_cluster=max_clusters
        )
        clusters_dict = {i: item for i, item in enumerate(clusters)}
        clusters_assign = {}
        for c, l in clusters_dict.items():
            for period in l:
                clusters_assign[period] = c
        self.clusters = clusters_assign

        var = {}
        for l, c in clusters_assign.items():
            if c in var:
                var[c].append(np.var(self.data[cp[l] : cp[l+1]]))
            else:
                var[c] = [np.var(self.data[cp[l] : cp[l+1]])]

        for c, l in var.items():
            var[c] = np.mean(np.mean(l))
        var = dict(sorted(var.items(), key=lambda item: item[1]))

        conversion_dict = {}
        for i, item in enumerate(var):
            conversion_dict[item] = i
        self.conversion_dict = conversion_dict

        clusters_relabel = np.ones(len(self.data))
        for l, c in clusters_assign.items():
            clusters_relabel[cp[l] : cp[l+1]] = conversion_dict[clusters_assign[l]]
        self.regime_labels = clusters_relabel

    def plot_regimes(self, figsize=(15, 5), save=None):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.figure(figsize=figsize)
        plt.plot(self.data)
        for i in range(len(self.changepoints) - 1):
            plt.axvspan(
                self.changepoints[i], 
                self.changepoints[i+1],
                alpha=0.3,
                color=colors[self.conversion_dict[self.clusters[i]]]
            )
        if save is not None:
            plt.savefig(save)

    def run(self):
        self.get_vol()
        self.get_changepoints()
        self.get_attr()
        self.assign_clusters()
