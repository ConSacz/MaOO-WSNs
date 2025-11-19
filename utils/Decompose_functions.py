import numpy as np
from .Normalize_functions import global_normalized

# %% scalarizing: Tchebycheff
def tchebycheff(f, w, RP):
    #diff = np.abs(f - z)
    diff = global_normalized(f, RP)
    w_safe = np.where(w == 0, 1e-12, w)
    return np.max(diff / w_safe)

# %% scalarizing: compute perpendicular distance
def vertical_distance(f, w, ref = np.zeros(0)):
    if ref.size:
        f = f - ref
    wn = w / (np.linalg.norm(w) + 1e-12)
    proj = np.dot(f, wn) * wn
    perp = f - proj
    return np.linalg.norm(perp)

# %% scalarizing: compute associate distance to RP
def associate_to_reference(F, W, RP):
    Znorm = global_normalized(F, RP)
    rd = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    projections = np.dot(Znorm, rd.T)
    proj_vecs = projections[:, :, None] * rd[None, :, :]
    perps = np.linalg.norm(Znorm[:, None, :] - proj_vecs, axis=2)
    ref_idx = np.argmin(perps, axis=1)
    dist = perps[np.arange(perps.shape[0]), ref_idx]
    return ref_idx, dist, perps

# %%
def weight_assign(F, W, RP):
    from scipy.optimize import linear_sum_assignment

    def hungarian_assign(perps):
        n = perps.shape[0]
        eps = 1e-9
        cost = perps + (np.arange(n)[:, None] * eps)
        ind_row, ind_col = linear_sum_assignment(cost)
        ref_idx = ind_col
        dist = perps[np.arange(n), ref_idx]
        return ref_idx, dist
    
    def greedy_assign(perps):
        n = perps.shape[0]
        pairs = []
        for i in range(n):
            for j in range(n):
                pairs.append((perps[i, j], i, j))
    
        pairs.sort(key=lambda x: x[0])
    
        ref_idx = np.full(n, -1)
        used_rp = np.zeros(n, dtype=bool)
        assigned = 0
        for d, i, j in pairs:
            if ref_idx[i] == -1 and not used_rp[j]:
                ref_idx[i] = j
                used_rp[j] = True
                assigned += 1
                if assigned == n:
                    break

        dist = perps[np.arange(n), ref_idx]
        return ref_idx, dist
    
    _, _, perps = associate_to_reference(F, W, RP)
    ref_idx, dist = greedy_assign(perps)
    W = W[ref_idx]
    return W

# %% Das & Dennis reference generation
def das_dennis_generate_L2(N_obj, d):
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

    # No L2-normalize!
    return W


# %% random reference generation
def random_weights(n_weights, n_obj):
    if n_obj == 2:
        w = np.linspace(0, 1, n_weights)[:, None]
        W = np.hstack([w, 1 - w])
        return W
    X = np.random.rand(n_weights, n_obj)
    X /= X.sum(axis=1, keepdims=True)
    return X