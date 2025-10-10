"""
IMODE (practical Python implementation) with dominance-based comparisons
Based on the code you provided. Replaced single-objective comparisons with
Pareto dominance logic for multi-objective problems.

Dependencies:
    numpy, scipy

Usage example:
    from imode_dominance import imode
    def sphere(X): return np.sum(X**2, axis=1).reshape(-1,1)  # single-objective example
    res = imode(obj_func=lambda X: dtlz2(X, n_obj=3), dim=12, max_fes=20000, seed=1)

This file contains:
 - dominance helpers: dominates, nondominated_front, prune_archive
 - updated get_phi_idx for Fvals shape (NP, n_obj)
 - replace_parent_if_better using dominance
 - archive pruning using nondominated-first heuristic
 - operator bookkeeping updated to use dominance/tie-break by sum

Note: Fvals is always treated as shape (NP, n_obj)
"""

import numpy as np
from scipy.optimize import minimize

# ---------- dominance helpers ----------

def dominates(a, b):
    """
    Return True if a dominates b (a <= b component-wise and a < b for at least one).
    a, b: 1D arrays same length
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.all(a <= b) and np.any(a < b)


def nondominated_front(F):
    """
    Return boolean mask (length n) marking nondominated points in F (shape (n, m)).
    O(n^2) straightforward implementation.
    """
    F = np.asarray(F)
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if j == i or not mask[j]:
                continue
            if dominates(F[j], F[i]):
                mask[i] = False
                break
        # if i still nondominated, remove any j that i dominates
        if mask[i]:
            for j in range(n):
                if j == i or not mask[j]:
                    continue
                if dominates(F[i], F[j]):
                    mask[j] = False
    return mask


def prune_archive(X_archive, obj_archive, max_size):
    """
    Shrink archive to max_size.
    - keep nondominated first; if still > max_size, choose by sum(obj) heuristic.
    X_archive: (k, dim)
    obj_archive: (k, n_obj)
    """
    if X_archive.shape[0] <= max_size:
        return X_archive
    k = X_archive.shape[0]
    mask_nd = nondominated_front(obj_archive)
    # if nondominated count <= max_size -> keep them and fill with best dominated
    if np.sum(mask_nd) <= max_size:
        keep_idx = np.where(mask_nd)[0].tolist()
        rem_idx = [i for i in range(k) if i not in keep_idx]
        if len(rem_idx) > 0:
            sums = np.sum(obj_archive[rem_idx], axis=1)
            order = np.argsort(sums)
            need = max_size - len(keep_idx)
            keep_idx += [rem_idx[i] for i in order[:need]]
        return X_archive[np.array(keep_idx)]
    else:
        # too many nondominated -> select top by sum
        nd_objs = obj_archive[mask_nd]
        nd_idx = np.where(mask_nd)[0]
        sums = np.sum(nd_objs, axis=1)
        order = np.argsort(sums)
        keep_idx = nd_idx[order[:max_size]]
        return X_archive[keep_idx]


# ---------- small helpers ----------

def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x

# clamp within bounds
def clamp(X, xl, xu):
    return np.minimum(np.maximum(X, xl), xu)

# get_phi_idx: choose an index (row) from top 10% best solutions in Farr (shape (NP, n_obj))
def get_phi_idx(Farr):
    Farr = np.asarray(Farr)
    if Farr.ndim != 2:
        Farr = Farr.reshape(Farr.shape[0], -1)
    # use sum as score (smaller better) as a simple scalarization for selection
    scores = np.sum(Farr, axis=1)
    N = Farr.shape[0]
    top_k = max(1, int(np.ceil(0.1 * N)))
    best_rows = np.argsort(scores)[:top_k]
    return int(_rng.choice(best_rows, size=None))


# ---------- mutation operators as in paper ----------

def mutation_current_to_phi_best_with_archive(X, idx, phi_idx, r1_idx, r2_vec, F):
    # v = x_i + F*(x_phi - x_i + x_r1 - x_r2)
    v = X[idx] + F * (X[phi_idx] - X[idx] + X[r1_idx] - r2_vec)
    return v.flatten()


def mutation_current_to_phi_best_no_archive(X, idx, phi_idx, r1_idx, r3_idx, F):
    # v = x_i + F*(x_phi - x_i + x_r1 - x_r3)
    v = X[idx] + F * (X[phi_idx] - X[idx] + X[r1_idx] - X[r3_idx])
    return v.flatten()


def mutation_weighted_rand_to_phi_best(X, idx, x_phi, x_r1, x_r3, F):
    v = F * X[x_r1] + (X[x_phi] - X[x_r3])
    return v.flatten()

# ---------------- crossover ----------------

def crossover_binomial(x, v, Cr):
    D = x.size
    u = x.copy()
    jrand = _rng.integers(0, D)
    mask = _rng.random(D) <= Cr
    mask[jrand] = True
    u[mask] = v[mask]
    return u


def crossover_exponential(x, v, Cr):
    D = x.size
    u = x.copy()
    start = _rng.integers(0, D)
    L = 0
    while L < D:
        j = (start + L) % D
        u[j] = v[j]
        L += 1
        if _rng.random() > Cr:
            break
    return u


# ---------- selection / replace using dominance ----------
def replace_parent_if_better(pop_X, pop_F, child_X, child_F, idx):
    """
    If child dominates parent -> replace and return True + old parent.
    child_F and pop_F[idx] are 1D arrays of length n_obj.
    """
    if dominates(child_F, pop_F[idx]):
        old = pop_X[idx].copy()
        pop_X[idx] = child_X.copy()
        pop_F[idx] = child_F.copy()
        return True, old
    return False, None


# ---------- Example problem: DTLZ2 ----------
def dtlz2(X, n_obj=3):
    """
    DTLZ2 with decision variables in [0,1].
    assumes X shape (pop, nvar) with nvar >= n_obj - 1
    returns shape (pop, n_obj)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    pop, nvar = X.shape
    k = nvar - (n_obj - 1)
    x1 = X[:, :n_obj - 1]
    x2 = X[:, n_obj - 1:]
    g = np.sum((x2 - 0.5) ** 2, axis=1)
    F = np.zeros((pop, n_obj))
    for i in range(pop):
        xi = x1[i]
        fi = []
        for m in range(n_obj):
            val = 1 + g[i]
            for j in range(n_obj - m - 1):
                val *= np.cos(xi[j] * np.pi / 2.0)
            if m > 0:
                val *= np.sin(xi[n_obj - m - 1] * np.pi / 2.0)
            fi.append(val)
        F[i, :] = fi
    return F


# ---------- IMODE main function ----------

def imode(obj_func,
          dim=12,
          n_obj=3,
          bounds=(0, 1),
          max_fes=100000,
          seed=None,
          NP_init=None,
          NP_min=4,
          archive_rate=2.6,
          use_local_search=False):
    """
    Returns: X, Fvals, history dict
    """
    # rng
    global _rng
    if seed is not None:
        _rng = np.random.default_rng(seed)
    else:
        _rng = np.random.default_rng()

    xl, xu = bounds if isinstance(bounds, (tuple, list)) and np.isscalar(bounds[0]) else (None, None)
    if xl is None:
        xl = np.asarray(bounds[0])
        xu = np.asarray(bounds[1])
    else:
        xl = float(xl); xu = float(xu)

    if NP_init is None:
        NP_init = max(6 * dim * dim, 10)

    NP = int(NP_init)
    NPmin = int(NP_min)
    MAXFES = int(max_fes)
    FES = 0

    archive_size = max(1, int(np.round(archive_rate * dim)))
    archive_X = np.zeros((0, dim))

    # population
    X = _rng.random((NP, dim)) * (xu - xl) + xl
    Fvals = obj_func(X)
    FES += X.shape[0]

    # initial operator assignment
    n_ops = 3
    NPop = int(np.floor(NP / n_ops)) * np.ones(n_ops, dtype=int)
    remainder = NP - np.sum(NPop)
    for i in range(remainder):
        NPop[i % n_ops] += 1

    Pls = 0.1
    CFEls = max(1, int(0.01 * MAXFES))
    use_ls = bool(use_local_search)
    history = {'best_f': [], 'FES': []}

    gen = 0

    # main loop
    while FES < MAXFES:
        gen += 1
        perm = _rng.permutation(NP)
        splits = np.split(perm, np.cumsum(NPop)[:-1])

        op_children = []
        op_children_idx = []
        op_best_vals = np.full((n_ops, n_obj), np.inf)
        op_best_X = np.zeros((n_ops, dim))

        # adapt F and Cr
        F_schedule = np.clip(_rng.normal(loc=0.5, scale=0.3, size=NP), 0.05, 0.9)
        Cr_schedule = np.clip(_rng.random(NP), 0.0, 1.0)

        for op in range(n_ops):
            idxs = splits[op]
            if idxs.size == 0:
                op_children.append(np.zeros((0, dim))); op_children_idx.append(np.array([], dtype=int))
                continue
            children = np.zeros((idxs.size, dim))
            child_fs = np.zeros((idxs.size, n_obj))

            for ii, pid in enumerate(idxs):
                F_i = F_schedule[pid]
                Cr_i = Cr_schedule[pid]
                phi_idx = get_phi_idx(Fvals)

                idx_pool = list(set(range(NP)) - {pid})
                if len(idx_pool) < 3:
                    r1_idx = idx_pool[0]; r2_idx = idx_pool[0]; r3_idx = idx_pool[0]
                else:
                    r = _rng.choice(idx_pool, size=3, replace=False)
                    r1_idx, r2_idx, r3_idx = r[0], r[1], r[2]

                if op == 0:
                    if archive_X.shape[0] > 0 and _rng.random() < 0.5:
                        union = np.vstack([X, archive_X])
                        # pick random index in union
                        rid = int(_rng.integers(0, union.shape[0]))
                        r2_vec = union[rid]
                    else:
                        r2_vec = X[r2_idx]
                    v = mutation_current_to_phi_best_with_archive(X, pid, phi_idx, r1_idx, r2_vec, F_i)
                elif op == 1:
                    v = mutation_current_to_phi_best_no_archive(X, pid, phi_idx, r1_idx, r3_idx, F_i)
                else:
                    v = mutation_weighted_rand_to_phi_best(X, pid, phi_idx, r1_idx, r3_idx, F_i)

                v = clamp(v, xl, xu)

                if _rng.random() <= 0.5:
                    u = crossover_binomial(X[pid], v, Cr_i)
                else:
                    u = crossover_exponential(X[pid], v, Cr_i)

                # evaluate child
                fu = obj_func(u.reshape(1, -1))[0]
                FES += 1
                children[ii] = u
                child_fs[ii] = fu

                # track best for operator using dominance + tie-break sum
                if np.all(np.isinf(op_best_vals[op])):
                    op_best_vals[op] = fu.copy()
                    op_best_X[op, :] = u.copy()
                else:
                    if dominates(fu, op_best_vals[op]):
                        op_best_vals[op] = fu.copy()
                        op_best_X[op, :] = u.copy()
                    elif not dominates(op_best_vals[op], fu):
                        if np.sum(fu) < np.sum(op_best_vals[op]):
                            op_best_vals[op] = fu.copy()
                            op_best_X[op, :] = u.copy()

            op_children.append(children); op_children_idx.append(idxs)

        # selection: replace parent if child dominates parent
        Dop = np.zeros(n_ops)
        QRop = np.zeros(n_ops)
        IRVop = np.zeros(n_ops)

        for op in range(n_ops):
            children = op_children[op]
            idxs = op_children_idx[op]
            if idxs.size == 0:
                Dop[op] = 0.0
                QRop[op] = np.inf
                continue
            for k, pid in enumerate(idxs):
                child = children[k]
                # find child's fitness from child_fs (we didn't keep it out here) -> recompute
                fu = obj_func(child.reshape(1, -1))[0]
                FES += 1
                if dominates(fu, Fvals[pid]):
                    # move parent to archive
                    archive_X = np.vstack([archive_X, X[pid].reshape(1, -1)])
                    # prune archive if needed
                    if archive_X.shape[0] > archive_size:
                        arch_f = obj_func(archive_X)
                        archive_X = prune_archive(archive_X, arch_f, archive_size)
                    # replace
                    X[pid] = child.copy()
                    Fvals[pid] = fu.copy()

            # compute diversity and quality for operator
            members = idxs
            if members.size > 0:
                m_idxs = members
                m_F = Fvals[m_idxs]
                m_nd = nondominated_front(m_F)
                if np.any(m_nd):
                    cand = np.where(m_nd)[0]
                    sums = np.sum(m_F[cand], axis=1)
                    chosen = cand[np.argmin(sums)]
                    best_idx = m_idxs[chosen]
                else:
                    sums = np.sum(m_F, axis=1)
                    chosen = np.argmin(sums)
                    best_idx = m_idxs[chosen]

                dists = np.linalg.norm(X[members] - X[best_idx], axis=1)
                Dop[op] = np.mean(dists)
                QRop[op] = np.sum(Fvals[best_idx])
            else:
                Dop[op] = 0.0
                QRop[op] = np.inf

        # normalize diversity
        sumD = np.sum(Dop)
        if sumD == 0:
            DRop = np.ones(n_ops) / n_ops
        else:
            DRop = Dop / sumD

        # normalize QRop (smaller sum = better). convert to finite
        finite_mask = np.isfinite(QRop)
        if np.any(finite_mask):
            max_finite = np.max(QRop[finite_mask])
        else:
            max_finite = 1.0
        QRop_finite = np.where(np.isfinite(QRop), QRop, max_finite)
        sumQ = np.sum(QRop_finite)
        if sumQ == 0:
            QRates = np.ones(n_ops) / n_ops
        else:
            # we want higher QRates = better, so invert QRop_finite
            inv = 1.0 / (QRop_finite + 1e-12)
            QRates = inv / np.sum(inv)

        IRVop = (1.0 - QRates) + DRop

        sumIRV = np.sum(IRVop)
        if sumIRV == 0:
            ratios = np.ones(n_ops) / n_ops
        else:
            ratios = IRVop / sumIRV

        frac = np.clip(ratios, 0.1, 0.9)
        frac = frac / np.sum(frac)
        newNPop = np.floor(frac * NP).astype(int)
        diff = NP - np.sum(newNPop)
        for i in range(abs(diff)):
            idx = i % n_ops
            if diff > 0:
                newNPop[idx] += 1
            elif newNPop[idx] > 0:
                newNPop[idx] -= 1
        NPop = newNPop.copy()

        # linear population reduction
        NP_next = int(np.round(((NPmin - NP_init) / float(MAXFES)) * FES + NP_init))
        NP_next = max(NPmin, NP_next)
        if NP_next < NP:
            to_remove = NP - NP_next
            # remove worst individuals -> define worst by dominated status then sum
            nd_mask = nondominated_front(Fvals)
            if np.sum(nd_mask) < Fvals.shape[0]:
                # remove dominated first (largest sum among dominated)
                dom_idx = np.where(~nd_mask)[0]
                sums = np.sum(Fvals[dom_idx], axis=1)
                order = np.argsort(sums)[-to_remove:]
                worst_idx = dom_idx[order]
            else:
                sums = np.sum(Fvals, axis=1)
                worst_idx = np.argsort(sums)[-to_remove:]

            keep_mask = np.ones(NP, dtype=bool)
            keep_mask[worst_idx] = False
            X = X[keep_mask]
            Fvals = Fvals[keep_mask]
            NP = NP_next
            # redistribute NPop proportionally
            if NP > 0:
                prop = np.maximum(1, (NPop / np.sum(NPop)) * NP).astype(int)
                diff2 = NP - np.sum(prop)
                for i in range(abs(diff2)):
                    prop[i % n_ops] += 1 if diff2 > 0 else -1
                NPop = prop
        else:
            NP = NP_next

        # find best overall from nondominated front
        nd_mask_all = nondominated_front(Fvals)
        if np.any(nd_mask_all):
            nd_indices = np.where(nd_mask_all)[0]
            sums = np.sum(Fvals[nd_indices], axis=1)
            best_rel = np.argmin(sums)
            best_idx_overall = int(nd_indices[best_rel])
            best_f = Fvals[best_idx_overall].copy()
        else:
            sums = np.sum(Fvals, axis=1)
            best_idx_overall = int(np.argmin(sums))
            best_f = Fvals[best_idx_overall].copy()

        history['best_f'].append(best_f)
        history['FES'].append(FES)

        # local search
        if use_ls and FES >= 0.85 * MAXFES:
            if _rng.random() <= Pls:
                x_start = X[best_idx_overall].copy()
                try:
                    res = minimize(lambda z: obj_func(z.reshape(1, -1))[0],
                                   x_start,
                                   method='SLSQP',
                                   bounds=[(xl, xu)] * dim,
                                   options={'maxiter': max(1, int(CFEls / 10)), 'ftol': 1e-8})
                    if res.success:
                        f_sqp = obj_func(res.x.reshape(1, -1))[0]
                        FES += 1
                        # compare via dominance
                        if dominates(f_sqp, best_f) or (not dominates(best_f, f_sqp) and np.sum(f_sqp) < np.sum(best_f)):
                            X[best_idx_overall] = res.x.copy()
                            Fvals[best_idx_overall] = f_sqp.copy()
                            best_f = f_sqp.copy()
                            Pls = 0.1
                        else:
                            Pls = 0.0001
                    else:
                        Pls = 0.0001
                except Exception:
                    Pls = 0.0001

        # safety: prune archive if too large
        if archive_X.shape[0] > archive_size:
            arch_f = obj_func(archive_X)
            archive_X = prune_archive(archive_X, arch_f, archive_size)
        print("Fitness function count: ", FES )
        if FES >= MAXFES:
            break

    # final best
    nd_mask_all = nondominated_front(Fvals)
    if np.any(nd_mask_all):
        nd_indices = np.where(nd_mask_all)[0]
        sums = np.sum(Fvals[nd_indices], axis=1)
        best_rel = np.argmin(sums)
        best_idx_overall = int(nd_indices[best_rel])
        best_f = Fvals[best_idx_overall].copy()
    else:
        best_idx_overall = int(np.argmin(np.sum(Fvals, axis=1)))
        best_f = Fvals[best_idx_overall].copy()

    return X, Fvals, history


# %% ---------------- Quick run for testing ----------------
if __name__ == "__main__":
    # small smoke test on DTLZ2 (3 objectives)
    n_obj = 3
    n_var = 12
    pop_size = 100

    def obj(X):
        return dtlz2(X, n_obj=n_obj)

    X, F, history = imode(obj_func=obj,
                             dim=n_var,
                             n_obj=n_obj,
                             bounds=(0, 1),
                             max_fes=20000,
                             seed=1,
                             NP_init=pop_size,
                             NP_min=60,
                             archive_rate=2.6,
                             use_local_search=False)


    # print few nondominated solutions
    nd = nondominated_front(F)
    print("nondominated count:", np.sum(nd))
    
    # %% print final approx PF
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D  # cần import để tạo trục 3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10, c='b')
    
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    plt.title("NSGA-III approximate front (ZDT1)")
    ax.view_init(elev=25, azim=35)
    #plt.gca().invert_yaxis()
    # plt.show()

