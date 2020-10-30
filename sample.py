from __future__ import print_function

import time
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

NUM_SAMPLES = 200
STEPS = 100
MEANS = [[2, 3],
         [5, 1],
         [6, 4]]
COVS = [0.5, 0.5, 0.5]
NUM_MIXTURES = len(MEANS)


def target(x):
    w = 0
    for m, c in zip(MEANS, COVS):
        w += multivariate_normal.pdf(x, mean=m, cov=(c * np.identity(2)))
    return w


def target_multi(pts):
    w = np.zeros(len(pts))
    for m, c in zip(MEANS, COVS):
        w += multivariate_normal.pdf(np.array(pts), mean=m, cov=(c * np.identity(2)))
    return w


def gt():
    pts = []
    for m, c in zip(MEANS, COVS):
        pts += [[np.random.normal(ele, c) for ele in m] for i in range(NUM_SAMPLES / NUM_MIXTURES)]

    return pts


def dist2(v1, v2):
    return np.sqrt(sum([(x1 - x2)**2 for x1, x2 in zip(v1, v2)]))


def set_dist2(set1, set2):
    diff = np.expand_dims(set1, 0) - np.expand_dims(set2, 1)
    dists = np.sqrt(np.sum(diff * diff, 2))

    return np.mean(np.min(dists, 0))


def mcmc(likelihoods, num_samples):
    idx = np.random.choice(len(likelihoods))
    fx = likelihoods[idx]

    indices = [idx]

    count = 0
    while len(indices) < num_samples:
        idx_prop = np.random.choice(len(likelihoods))
        f_prime = likelihoods[idx_prop]

        if np.random.random() <= f_prime / fx:
            fx = f_prime
            idx = idx_prop

            if count < 10:
                continue

            indices.append(idx)

        count += 1

    return indices


def importance(likelihoods, num_samples):
    indices = []

    for i in range(0, num_samples):
        r = np.random.random()
        s = likelihoods[0]
        idx = 0
        while s < r:
            idx += 1
            s += likelihoods[idx]

        indices.append(idx)

    return indices


def low_variance(likelihoods, num_samples):
    indices = []
    r = np.random.uniform(0, 1. / num_samples)
    idx = 0
    s = likelihoods[idx]

    for i in range(0, num_samples):
        u = r + i * (1. / num_samples)
        while u > s:
            idx += 1
            s += likelihoods[idx]

        indices.append(idx)

    return indices


def jitter(pts):
    pts = np.array(pts)
    return (pts + np.random.normal(0, 0.1, size=pts.shape)).tolist()


def reweight(samples):
    # likelihoods = target_multi(samples)
    likelihoods = np.ones(len(samples))
    likelihoods = likelihoods / np.sum(likelihoods)

    return likelihoods.tolist()


def cluster_to_target(samples):
    clusters = [[], [], []]
    for pt in samples:
        dists = [dist2(pt, m) for m in MEANS]
        clusters[np.argmin(dists)].append(pt)

    return clusters


def plot_set(samples, rows, cols, iteration, clusters=True):
    plt.subplot(rows, cols, iteration + 3)
    plt.xlim(-1, 9)
    plt.ylim(-1, 9)
    plt.title("i = {}".format(iteration + 1))

    if not clusters:
        plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1])
    else:
        clusters = cluster_to_target(samples)
        for cluster in clusters:
            x_plt, y_plt = [ele[0] for ele in cluster], [ele[1] for ele in cluster]
            plt.scatter(x_plt, y_plt)


def test_one(init_pts, gt_pts, sampling_fn, num_steps=18, name="", clusters=True, plot=True):
    cols = 5
    rows = int(np.ceil(float(num_steps + 2)) / cols)

    samples = init_pts[:]

    if plot:
        plt.figure()
        plt.suptitle(name)

        # Ground truth.
        plt.subplot(rows, cols, 1)
        plt.xlim(-1, 9)
        plt.ylim(-1, 9)
        plt.title("GT")

        if not clusters:
            plt.scatter(np.array(gt_pts)[:, 0], np.array(gt_pts)[:, 1])
        else:
            clusters = cluster_to_target(gt_pts)
            for cluster in clusters:
                x_plt, y_plt = [ele[0] for ele in cluster], [ele[1] for ele in cluster]
                plt.scatter(x_plt, y_plt)

        # Initial samples.
        plot_set(init_pts, rows, cols, -1, clusters)

    total_time = 0

    for iteration in range(num_steps):
        likelihoods = reweight(samples)

        start = time.clock()
        indices = sampling_fn(likelihoods, NUM_SAMPLES)
        total_time += time.clock() - start

        samples = [samples[i] for i in indices]

        if plot:
            plot_set(samples, rows, cols, iteration, clusters)

        samples = jitter(samples)

    return total_time * 1000 / num_steps, set_dist2(gt_pts, samples)


def compare_all(init_pts, gt_pts):
    samples_mcmc = init_pts[:]
    samples_imp = init_pts[:]
    samples_lowvar = init_pts[:]

    for iteration in range(STEPS):
        mcmc_indices = mcmc(reweight(samples_mcmc), NUM_SAMPLES)
        samples_mcmc = [samples_mcmc[i] for i in mcmc_indices]

        imp_indices = importance(reweight(samples_imp), NUM_SAMPLES)
        samples_imp = [samples_imp[i] for i in imp_indices]

        lowvar_indices = low_variance(reweight(samples_lowvar), NUM_SAMPLES)
        samples_lowvar = [samples_lowvar[i] for i in lowvar_indices]

        if (iteration < 3 or iteration > STEPS - 5):
            plt.figure(iteration)
            plt.subplot(1, 4, 1)
            plt.xlim(-1, 9)
            plt.ylim(-1, 9)
            plt.title("GT")
            plt.scatter(np.array(gt_pts)[:, 0], np.array(gt_pts)[:, 1])

            plt.subplot(1, 4, 2)
            plt.xlim(-1, 9)
            plt.ylim(-1, 9)
            plt.title("MCMC")
            plt.scatter(np.array(samples_mcmc)[:, 0], np.array(samples_mcmc)[:, 1])

            plt.subplot(1, 4, 3)
            plt.xlim(-1, 9)
            plt.ylim(-1, 9)
            plt.title("Importance")
            plt.scatter(np.array(samples_imp)[:, 0], np.array(samples_imp)[:, 1])

            plt.subplot(1, 4, 4)
            plt.xlim(-1, 9)
            plt.ylim(-1, 9)
            plt.title("Low Variance")
            plt.scatter(np.array(samples_lowvar)[:, 0], np.array(samples_lowvar)[:, 1])

        samples_mcmc = jitter(samples_mcmc)
        samples_imp = jitter(samples_imp)
        samples_lowvar = jitter(samples_lowvar)


def test():
    var_times, imp_times = [], []
    var_dist, imp_dist = [], []

    for i in range(100):
        pts_gt = gt()
        pts_init = [[np.random.uniform(0, 8), np.random.uniform(0, 8)] for i in range(NUM_SAMPLES)]

        t, d = test_one(pts_init, pts_gt, low_variance, clusters=False, plot=False, num_steps=10)
        var_times.append(t)
        var_dist.append(d)

        t, d = test_one(pts_init, pts_gt, importance, clusters=False, plot=False, num_steps=10)
        imp_times.append(t)
        imp_dist.append(d)

    var_times, imp_times = np.array(var_times), np.array(imp_times)
    var_dist, imp_dist = np.array(var_dist), np.array(imp_dist)

    print("Times:")
    print("\tLow Variance: mean {} ms, var {} ms".format(np.mean(var_times), np.std(var_times)))
    print("\tImportance:   mean {} ms, var {} ms".format(np.mean(imp_times), np.std(imp_times)))
    print("Set distances:")
    print("\tLow Variance: mean {}, var {}".format(np.mean(var_dist), np.std(var_dist)))
    print("\tImportance:   mean {}, var {}".format(np.mean(imp_dist), np.std(imp_dist)))


if __name__ == '__main__':
    # test()

    pts_gt = gt()
    pts_init = [[np.random.uniform(0, 8), np.random.uniform(0, 8)] for i in range(NUM_SAMPLES)]

    # compare_all(pts_init, pts_gt)

    print("Testing Low Variance")
    t, d = test_one(pts_init, pts_gt, low_variance, name="Low Variance")
    print("\tAverage time: {} ms  Average Distance: {}".format(t, d))

    print("Importance")
    t, d = test_one(pts_init, pts_gt, importance, name="Importance")
    print("\tAverage time: {} ms  Average Distance: {}".format(t, d))

    plt.show()
