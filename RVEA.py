"""
Simple RVEA implementation (practical, educational)
- Uses numpy
- SBX crossover and polynomial mutation
- Reference vectors by Das & Dennis
- Selection via PBI (d1 + theta * d2), theta adaptive
"""

import numpy as np

# ---------- utility / genetic operators ----------
def uniform_reference_vectors(M, p=1):
    """
    Create roughly uniform reference vectors on the unit simplex using Das & Dennis approach.
    M: number of objectives
    p: divisions (higher p -> more vectors). Default p=1 gives M vectors; increase for more.
    Returns array V (n_vectors x M), normalized.
    """
    # For simplicity: generate a grid of integer compositions summing to p, then normalize.
    # We'll use recursion to generate combinations.
    def compositions(n, k):
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for tail in compositions(n - i, k - 1):
                    yield (i,) + tail
    comps = np.array([c for c in compositions(p, M)], dtype=float)
    V = comps / np.linalg.norm(comps, axis=1, keepdims=True)
    # remove any NaN rows (all zeros) and return normalized
    valid = ~np.isnan(V).any(axis=1)
    return V[valid]

def sbx_crossover(parent1, parent2, eta_c=20, xl=0.0, xu=1.0):
    """Simulated Binary Crossover for two parents (vectors in [0,1]^D)"""
    D = parent1.size
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(D):
        if np.random.rand() <= 0.5:
            if abs(parent1[i] - parent2[i]) > 1e-14:
                x1 = min(parent1[i], parent2[i])
                x2 = max(parent1[i], parent2[i])
                rand = np.random.rand()
                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta_c + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))
                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta_c + 1)
                if rand <= 1.0 / alpha:
                    betaq = (rand * alpha) ** (1.0 / (eta_c + 1))
                else:
                    betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1))
                c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
                c1 = np.clip(c1, xl, xu)
                c2 = np.clip(c2, xl, xu)
                if np.random.rand() <= 0.5:
                    child1[i] = c2
                    child2[i] = c1
                else:
                    child1[i] = c1
                    child2[i] = c2
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]
    return child1, child2

def polynomial_mutation(x, eta_m=20, pm=0.1, xl=0.0, xu=1.0):
    """Polynomial mutation on vector x in [xl,xu]"""
    y = x.copy()
    D = x.size
    for i in range(D):
        if np.random.rand() < pm:
            delta1 = (y[i] - xl) / (xu - xl)
            delta2 = (xu - y[i]) / (xu - xl)
            rnd = np.random.rand()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rnd <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1.0 - val ** mut_pow
            y[i] = y[i] + deltaq * (xu - xl)
            y[i] = np.clip(y[i], xl, xu)
    return y

# ---------- objective / normalization helpers ----------
def normalize_objectives(F, ideal, nadir=None):
    """
    Normalizes F with respect to ideal and (optionally) nadir.
    If nadir is None, use max across population.
    """
    if nadir is None:
        nadir = np.max(F, axis=0)
    denom = np.where(nadir - ideal == 0, 1e-12, nadir - ideal)
    return (F - ideal) / denom

def get_ideal(F):
    return np.min(F, axis=0)

# PBI calculation for individuals (rows) vs single reference vector v
def pbi_values(F_norm, v):
    """
    Compute (d1, d2) for each row in F_norm w.r.t unit vector v
    F_norm: N x M
    v: M vector (should be unit)
    returns d1 (N,), d2 (N,)
    """
    # projection length along v
    proj = F_norm.dot(v)
    d1 = proj  # scalar projection (since v unit)
    # compute perpendicular distance: ||F - d1*v||
    F_proj = np.outer(proj, v)
    diff = F_norm - F_proj
    d2 = np.linalg.norm(diff, axis=1)
    return d1, d2

# ---------- RVEA main ----------
def rvea(obj_func, n_var, n_obj, pop_size=100, max_gen=200, p_ref=3,
         crossover_prob=0.9, eta_c=20, eta_m=20, pm=1.0,
         theta0=5.0, alpha=2.0, seed=None):
    """
    obj_func: function X -> F (X: population x n_var -> F: population x n_obj)
    n_var, n_obj: dims
    pop_size: population size
    max_gen: number of generations
    p_ref: Das & Dennis divisions (higher -> more reference vectors)
    pm: mutation probability per variable (if None use 1/n_var)
    theta0: initial penalty parameter for PBI
    alpha: exponent controlling theta growth
    returns final population X and F
    """
    if seed is not None:
        np.random.seed(seed)
    if pm is None:
        pm = 1.0 / n_var

    # initialize population in [0,1]^n_var
    X = np.random.rand(pop_size, n_var)
    F = obj_func(X)  # evaluate

    # reference vectors
    V = uniform_reference_vectors(n_obj, p=p_ref)  # K x n_obj
    K = V.shape[0]

    # ideal point
    ideal = get_ideal(F)

    # main loop
    for gen in range(1, max_gen + 1):
        # mating: binary tournament selection based on domination rank or random
        # we'll use random mating pool and SBX
        mating_order = np.random.permutation(pop_size)
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = X[mating_order[i % pop_size]]
            p2 = X[mating_order[(i + 1) % pop_size]]
            if np.random.rand() < crossover_prob:
                c1, c2 = sbx_crossover(p1, p2, eta_c=eta_c, xl=0.0, xu=1.0)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = polynomial_mutation(c1, eta_m=eta_m, pm=pm, xl=0.0, xu=1.0)
            c2 = polynomial_mutation(c2, eta_m=eta_m, pm=pm, xl=0.0, xu=1.0)
            offspring.append(c1); offspring.append(c2)
        offspring = np.array(offspring)[:pop_size]
        F_off = obj_func(offspring)

        # Combine
        X_all = np.vstack([X, offspring])
        F_all = np.vstack([F, F_off])

        # update ideal point
        ideal = np.min(F_all, axis=0)

        # normalization (use nadir as max)
        nadir = np.max(F_all, axis=0)
        F_norm = normalize_objectives(F_all, ideal, nadir)

        # associate each individual to a reference vector by minimal angle
        # compute angles (via cosine)
        # ensure V unit
        V_unit = V / np.linalg.norm(V, axis=1, keepdims=True)
        # normalize F_norm direction vectors (avoid zero)
        dir_norms = np.linalg.norm(F_norm, axis=1, keepdims=True)
        dir_unit = np.where(dir_norms == 0, 0, F_norm / dir_norms)

        cos = dir_unit.dot(V_unit.T)  # (2N) x K
        # angle = arccos(cos)
        # assign to vector with max cosine (min angle)
        assoc = np.argmax(cos, axis=1)  # index of associated ref vector

        # For selection: for each reference vector, compute PBI fitness d1 + theta * d2
        t = gen
        T = max_gen
        theta = theta0 * (t / T) ** alpha

        chosen_indices = []
        for k in range(K):
            members_idx = np.where(assoc == k)[0]
            if members_idx.size == 0:
                continue
            F_members = F_norm[members_idx]
            d1, d2 = pbi_values(F_members, V_unit[k])
            fitness = d1 + theta * d2
            # pick best (smallest fitness)
            best_local = members_idx[np.argmin(fitness)]
            chosen_indices.append(best_local)

        # if we didn't fill pop_size (because K < pop_size), fill by best remaining fitness globally
        if len(chosen_indices) < pop_size:
            remaining = np.setdiff1d(np.arange(F_all.shape[0]), chosen_indices)
            # global ranking by non-dominated sort or simple crowding: we'll use sum of objectives (min)
            global_score = np.sum(normalize_objectives(F_all, ideal, nadir), axis=1)
            fill_order = remaining[np.argsort(global_score[remaining])]
            need = pop_size - len(chosen_indices)
            chosen_indices.extend(list(fill_order[:need]))
        elif len(chosen_indices) > pop_size:
            # if more (rarely), trim by fitness
            chosen_indices = chosen_indices[:pop_size]

        # form next generation
        X = X_all[chosen_indices]
        F = F_all[chosen_indices]

        # optional: print progress
        if gen % max(1, max_gen // 10) == 0 or gen == 1 or gen == max_gen:
            print(f"gen {gen}/{max_gen}, theta={theta:.4f}, ideal={ideal}")

    return X, F

# ---------- Example problem: DTLZ2 ----------
def dtlz2(X, n_obj=3):
    """
    DTLZ2 with decision variables in [0,1].
    assumes X shape (pop, nvar) with nvar >= n_obj - 1
    """
    pop, nvar = X.shape
    k = nvar - (n_obj - 1)
    # split
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

# ---------- Quick run ----------
if __name__ == "__main__":
    # small test on DTLZ2 (3 objectives), 7 variables
    n_obj = 3
    n_var = 12
    pop_size = 100
    max_gen = 200

    def objf_pop(X):
        return dtlz2(X, n_obj=n_obj)

    X, F = rvea(objf_pop, n_var=n_var, n_obj=n_obj,
                  pop_size=pop_size, max_gen=max_gen, p_ref=6,
                  crossover_prob=0.9, eta_c=20, eta_m=20, pm=None,
                  theta0=5.0, alpha=2.0, seed=42)

    # print final approx PF first 5 rows
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
