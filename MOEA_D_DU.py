# moeaddu.py
# a compact, readable MOEA/D implementation with a distance-based updating (DU) variant
# requires: numpy
import numpy as np
import matplotlib.pyplot as plt
# -------------------------
# helper: uniform weight vectors (simple grid)
# -------------------------
def uniform_weights(n_weights, n_obj):
    # produce roughly-uniform weight vectors on a simplex by simple random + normalize if n_obj>2
    if n_obj == 2:
        w = np.linspace(0, 1, n_weights)[:, None]
        W = np.hstack([w, 1 - w])
        return W
    # for n_obj>2, sample random and normalize (simple and practical)
    X = np.random.rand(n_weights, n_obj)
    X /= X.sum(axis=1, keepdims=True)
    return X
def das_dennis_generate(M, d):
    # M: objectives, d: divisions (positive integer)
    def recursive_gen(M, left, depth):
        if depth == M - 1:
            return [[left]]
        res = []
        for i in range(left + 1):
            tails = recursive_gen(M, left - i, depth + 1)
            for t in tails:
                res.append([i] + t)
        return res

    combos = recursive_gen(M, d, 0)
    W = np.array(combos, dtype=float) / float(d)
    # remove duplicates (shouldn't be necessary)
    # ensure unit directions (norm not zero)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W = W / norms
    return W
# -------------------------
# scalarizing: Tchebycheff
# -------------------------
def tchebycheff(f, w, z):
    # f: objectives (M,)
    # w: weight vector (M,)
    # z: ideal point (M,)
    # return scalar value (the smaller the better)
    diff = np.abs(f - z)
    # avoid zero weights
    w_safe = np.where(w == 0, 1e-12, w)
    return np.max(diff / w_safe)

# -------------------------
# DE operator (rand/1/bin)
# -------------------------
def de_rand1_bin(pop, F=0.5, CR=0.9, xmin=None, xmax=None):
    # pop: (N, D)
    N, D = pop.shape
    idx = np.arange(N)
    u = np.empty_like(pop)
    for i in range(N):
        a, b, c = np.random.choice(idx[idx != i], 3, replace=False)
        mutant = pop[a] + F * (pop[b] - pop[c])
        # crossover binomial
        cross = np.random.rand(D) < CR
        jrand = np.random.randint(D)
        cross[jrand] = True
        trial = np.where(cross, mutant, pop[i])
        # bounds
        if xmin is not None:
            trial = np.maximum(trial, xmin)
        if xmax is not None:
            trial = np.minimum(trial, xmax)
        u[i, :] = trial
    return u

# -------------------------
# compute perpendicular (vertical) distance from objective point f to the direction w
# used to rank neighbors in DU (common interpretation: distance between solution and weight vector)
# -------------------------
def vertical_distance(f, w, ref = np.zeros(0)):
    # project f (objectives) on direction w (weights)
    # both f and w are arrays of shape (M,)
    # we compute distance from point f to the ray along w through origin (or ref)
    # if ref given (ideal), subtract it first
    if ref.size:
        f = f - ref
    wn = w / (np.linalg.norm(w) + 1e-12)
    proj = np.dot(f, wn) * wn
    perp = f - proj
    return np.linalg.norm(perp)

def problem_eval(x):
    x = np.atleast_2d(x)
    n = x.shape[1]
    g = np.sum((x[:, 2:] - 0.5)**2, axis=1)
    f1 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
    f2 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
    f3 = (1 + g) * np.sin(0.5 * np.pi * x[:, 0])
    F =np.vstack([f1, f2, f3]).T
    return F.squeeze()

# %%
# MOEA/D with DU like update function
#

#problem_eval,    # function x -> f (array of M objectives)
D = 12               # decision dimension
n_obj = 3           # number of objectives
nPop=100
max_gen=200
neigh_size=20
nr=2            # max replacements per offspring
Fm=0.5 
CR=0.9
xmin=0 
xmax=1
seed=0

# %%
if seed is not None:
    np.random.seed(seed)

# 1) weight vectors and neighborhoods
#W = uniform_weights(nPop, n_obj)            # (nPop, n_obj)
W = das_dennis_generate(3,13)
rows_to_delete = np.random.choice(W.shape[0], 5, replace=False)
W = np.delete(W, rows_to_delete, axis=0)
# compute neighbors by Euclidean distance in weight space
distW = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
neighborhoods = np.argsort(distW, axis=1)[:, :neigh_size]

# 2) initialize population (random uniformly in bounds if given, else normal(0,1))
X = np.random.rand(nPop, D) * (xmax - xmin) + xmin
F = np.array([problem_eval(X[i]) for i in range(nPop)])  # (nPop, n_obj)

# ideal point
z = np.min(F, axis=0)
# %%
for gen in range(max_gen):
    # create offspring population via DE (vectorized: produce nPop offspring corresponding to each subproblem)
    U = de_rand1_bin(X, F=Fm, CR=CR, xmin=xmin, xmax=xmax)  # (nPop, D)
    FU = np.array([problem_eval(U[i]) for i in range(nPop)])
    # update ideal point
    z = np.minimum(z, np.min(FU, axis=0))

    # for each subproblem i, apply DU update using its neighborhood
    for i in range(nPop):
        child = U[i]
        child_f = FU[i]
        # candidate indices: neighborhood of i (you could also use whole population)
        cand_idx = neighborhoods[i].copy()

        # compute vertical distance between candidate solutions' objectives and their weight vectors
        # note: for DU, one uses vertical distance between solution and weight vector in objective space;
        # here we compute distance of candidate's objective to the weight vector direction of candidate's subproblem
        dists = np.zeros_like(cand_idx, dtype=float)
        for k, j in enumerate(cand_idx):
            dists[k] = vertical_distance(F[j], W[j], ref=z)  # rank by this distance

        # order candidates by ascending vertical distance (closest first)
        order = np.argsort(dists)
        replaced = 0
        for idx_in_order in order:
            j = cand_idx[idx_in_order]
            # compare scalarizing values
            val_child = tchebycheff(child_f, W[j], z)
            val_j = tchebycheff(F[j], W[j], z)
            if val_child < val_j:
                # replace
                X[j] = child.copy()
                F[j] = child_f.copy()
                replaced += 1
                if replaced >= nr:
                    break
    # optional: shuffle order of subproblems per generation to avoid bias
    perm = np.random.permutation(nPop)
    X = X[perm]
    F = F[perm]
    W = W[perm]
    neighborhoods = neighborhoods[perm][:, :]  # note: neighborhoods indices relate to old ordering; for simplicity we recompute distances each gen
    # recompute neighborhoods properly (cheap for moderate nPop)
    distW = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    neighborhoods = np.argsort(distW, axis=1)[:, :neigh_size]

# %%
# Tạo figure
    # fig = plt.figure(1)
    # plt.clf()
    
    # # Vẽ Pareto front
    # plt.plot(F[:, 0], F[:, 1], 'o', color='g')
    # None
    # # Cập nhật đồ thị theo từng iteration
    # plt.pause(0.001)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # cần import để tạo trục 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10, c='b')

ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
plt.title("NSGA-III approximate front (ZDT1)")
ax.view_init(elev=25, azim=35)
#plt.gca().invert_yaxis()
plt.show()