import numpy as np
from .Normalize_functions import global_normalized
#from Domination_functions import get_pareto_front

# -------------------------
# scalarizing: Tchebycheff
# -------------------------
def tchebycheff(f, w, RP):
    #diff = np.abs(f - z)
    diff = global_normalized(f, RP)
    w_safe = np.where(w == 0, 1e-12, w)
    return np.max(diff / w_safe)

# -------------------------
# compute perpendicular distance
# -------------------------
def vertical_distance(f, w, ref = np.zeros(0)):
    if ref.size:
        f = f - ref
    wn = w / (np.linalg.norm(w) + 1e-12)
    proj = np.dot(f, wn) * wn
    perp = f - proj
    return np.linalg.norm(perp)

# -------------------------
# compute associate distance to RP
# -------------------------
def associate_to_reference(F, W, ideal):
    Z = F - ideal
    Znorm = Z / (np.max(np.abs(Z), axis=0, keepdims=True) + 1e-12)
    rd = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    projections = np.dot(Znorm, rd.T)
    proj_vecs = projections[:, :, None] * rd[None, :, :]
    perps = np.linalg.norm(Znorm[:, None, :] - proj_vecs, axis=2)
    ref_idx = np.argmin(perps, axis=1)
    dist = perps[np.arange(perps.shape[0]), ref_idx]
    return ref_idx, dist

# ----------------------------
# uniform reference generation
# ----------------------------
def weight_assign(pop,RP):
    pop.sort(key=lambda p: p['Cost'][0], reverse=True)
    # x_sta = RP[:,0].flatten()
    # x_nad = RP[:,1].flatten()
    x_sta = np.zeros(2, dtype=int)
    x_nad = np.ones(2, dtype=int)
    Npop = len(pop)
    Nf = len(pop[0]['Cost'])
    w = np.zeros((Npop, Nf), dtype=int)
    
    weights = np.linspace(x_sta,x_nad,Npop)
    weights[:, 1] = weights[::-1, 1]
    #weights[:, [0, 1]] = weights[:, [1, 0]]
    w = weights
    
    return pop, w

# ----------------------------
# Das & Dennis reference generation
# ----------------------------
def das_dennis_generate(N_obj, d):
    def recursive_gen(N_obj, left, depth):
        if depth == N_obj - 1:
            return [[left]]
        res = []
        for i in range(left + 1):
            tails = recursive_gen(N_obj, left - i, depth + 1)
            for t in tails:
                res.append([i] + t)
        return res
    combos = recursive_gen(N_obj, d, 0)
    W = np.array(combos, dtype=float) / float(d)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W = W / norms
    return W

# ----------------------------
# random reference generation
# ----------------------------
def random_weights(n_weights, n_obj):
    if n_obj == 2:
        w = np.linspace(0, 1, n_weights)[:, None]
        W = np.hstack([w, 1 - w])
        return W
    X = np.random.rand(n_weights, n_obj)
    X /= X.sum(axis=1, keepdims=True)
    return X