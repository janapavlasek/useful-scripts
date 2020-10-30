import numpy as np
import matplotlib.pyplot as plt


class Gaussian(object):
    def __init__(self, mean, std):
        self.mean = float(mean)
        self.std = float(std)

    def pdf(self, x):
        y = 1. / (self.std * np.sqrt(2 * np.pi))
        return y * np.exp(-0.5 * ((x - self.mean) / self.std) ** 2)

    def sample(self):
        return np.random.normal(self.mean, self.std)


class Mixture(object):
    def __init__(self, gaussians, weights, normalize=True):
        self.gaussians = gaussians
        self.weights = [max(0, w) for w in weights]

        if normalize:
            self.normalize()

    def normalize(self):
        self.weights = [w / np.sum(self.weights) for w in self.weights]

    def pdf(self, x):
        f = 0
        for g, w in zip(self.gaussians, self.weights):
            f += w * g.pdf(x)
        return f

    def num_components(self):
        return len(self.gaussians)

    def at(self, i):
        return self.gaussians[i]

    def weight(self, i):
        return self.weights[i]

    def add(self, mix):
        self.gaussians += mix.gaussians
        self.weights += mix.weights

    def product(self, dist):
        gaussians = []
        weights = []
        for i in range(self.num_components()):
            for j in range(dist.num_components()):
                std_inv = 1. / self.at(i).std + 1. / dist.at(j).std
                std = 1. / std_inv
                m = std * (self.at(i).mean / self.at(i).std + dist.at(j).mean / dist.at(j).std)
                g = Gaussian(m, std)

                w = self.weight(i) * self.at(i).pdf(m) * dist.weight(j) * dist.at(j).pdf(m)
                w /= g.pdf(m)

                # print(m, 1. / std_inv, w)

                gaussians.append(g)
                weights.append(w)

        return Mixture(gaussians, weights)


# def product(gaussians, M):
#     d = len(gaussians)

#     gaussians = []
#     weights = []

#     y = np.arange(M)
#     grid = np.meshgrid(*[y for i in range(d)])
#     idx = np.array([g.reshape(-1) for g in grid]).T
#     N, _ = idx.shape

#     for i in range(0, N):
#         pass


def product(gaussians):
    std_inv = 0
    mu = 0
    for f in gaussians:
        std_inv += 1. / f.std
        mu += f.mean / f.std
    std = 1. / std_inv
    mu = std * mu

    return mu, std


def gibbs_sample_product(mixtures, M, K):
    d = len(mixtures)
    labels = np.zeros(d, dtype=np.int)

    estimate = Mixture([Gaussian(0, 1) for i in range(M)], [1 for i in range(M)])

    # Initialize the labels.
    for j, mix in enumerate(mixtures):
        labels[j] = np.random.choice(M, p=mix.weights)

    # Iteration loop.
    for it in range(K):
        for j, mix in enumerate(mixtures):
            ks = np.arange(d)
            ks = ks[np.delete(ks, j)]

            mu_star, std_star = product([mixtures[k].at(labels[k]) for k in ks])
            f_star = Gaussian(mu_star, std_star)

            for i, comp_i in enumerate(mix.gaussians):
                mu_i, std_i = product([comp_i, f_star])
                f_i = Gaussian(mu_i, std_i)

                w_i = mix.weight(i) * comp_i.pdf(mu_i) * f_star.pdf(mu_i) / f_i.pdf(mu_i)

                estimate.gaussians[i] = f_i
                estimate.weights[i] = w_i

            estimate.normalize()
            labels[j] = np.random.choice(M, p=estimate.weights)

    mu, std = product([mix.at(labels[j]) for j, mix in enumerate(mixtures)])
    f = Gaussian(mu, std)

    return f.sample()


def mixture_product(mixtures, M, K):
    samples = [gibbs_sample_product(mixtures, M, K) for i in range(100)]
    return samples


d = 2
M = 5
# y = np.arange(M)

# grid = np.meshgrid(*[y for i in range(d)])
# print("d = {} M = {} M^d = {}".format(d, M, M**d))
# idx = np.array([g.reshape(-1) for g in grid])
# print(idx.shape, idx)
# for i in range(d):
#     print(grid[i])


f = Mixture([Gaussian(np.random.uniform(0, 9), 1) for i in range(M)], [np.random.normal(0.5, 0.2) for i in range(M)])
g = Mixture([Gaussian(np.random.uniform(0, 9), 1) for i in range(M)], [np.random.normal(0.5, 0.2) for i in range(M)])
h = f.product(g)

m = mixture_product([f, g], M, 10)

x = np.linspace(-2, 11, 500)

fig, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

a0.set_xlim(-2, 11)
a0.plot(x, f.pdf(x), label="f(x)")
a0.plot(x, g.pdf(x), label="g(x)")
a0.plot(x, h.pdf(x), label="f(x) * g(x)")
# a0.plot(x, m.pdf(x), label="gibbs")
a0.legend()

a1.set_xlim(-2, 11)
a1.bar(m, np.full(len(m), 1), 0.02)


plt.show()
