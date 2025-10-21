# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
"""
RVEA (Reference Vector Guided Evolutionary Algorithm) - pop-only version
- pop: list of dicts, each dict has keys:
    'Position' : ndarray shape (n_var,)
    'Cost'     : ndarray shape (n_obj,)
- No persistent X/F arrays (only transient arrays used for computations)
- Uses SBX crossover + polynomial mutation, Das & Dennis reference vectors, PBI selection
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.Decompose_functions import das_dennis_generate
from utils.GA_functions import sbx_crossover, polynomial_mutation

# ---------- objective / normalization helpers ----------
def normalize_objectives(F, ideal, nadir=None):
    """Normalize F rows by (nadir - ideal). If nadir None, use max across rows."""
    if nadir is None:
        nadir = np.max(F, axis=0)
    denom = np.where(nadir - ideal == 0, 1e-12, nadir - ideal)
    return (F - ideal) / denom

def get_ideal(F):
    return np.min(F, axis=0)

def pbi_values(F_norm, v):
    """Compute d1 (projection length) and d2 (perpendicular distance) for rows of F_norm w.r.t unit v"""
    proj = F_norm.dot(v)  # (N,)
    # projected points:
    F_proj = np.outer(proj, v)
    diff = F_norm - F_proj
    d2 = np.linalg.norm(diff, axis=1)
    d1 = proj
    return d1, d2

# ---------- pop utilities ----------
def init_pop_uniform(pop_size, n_var, eval_func, xl=0.0, xu=1.0):
    """Return pop list with random positions in [xl,xu] and evaluated costs"""
    pop = []
    X = np.random.rand(pop_size, n_var) * (xu - xl) + xl
    F = eval_func(X)  # evaluate all at once (obj_func expects array)
    for i in range(pop_size):
        pop.append({'Position': X[i].copy(), 'Cost': F[i].copy()})
    return pop

def pop_positions(pop):
    return np.array([ind['Position'] for ind in pop])

def pop_costs(pop):
    return np.array([ind['Cost'] for ind in pop])

# ---------- RVEA main (pop-only) ----------
def rvea_pop(obj_func, n_var, n_obj, pop_size=100, max_gen=200, p_ref=1,
             crossover_prob=0.9, eta_c=20, eta_m=20, pm=None,
             theta0=5.0, alpha=2.0, xl=0.0, xu=1.0, seed=None):
    """
    obj_func: function positions_array (N x n_var) -> costs_array (N x n_obj)
    Returns: pop (list), V (reference vectors), ideal (final ideal point)
    """
    xmin = np.zeros(n_var)
    xmax = np.ones(n_var)
    if seed is not None:
        np.random.seed(seed)
    if pm is None:
        pm = 1.0 / n_var

    # 1) initialize pop
    pop = init_pop_uniform(pop_size, n_var, eval_func=obj_func, xl=xl, xu=xu)

    # 2) reference vectors
    V = das_dennis_generate(n_obj, p_ref)  # K x n_obj
    if V.shape[0] == 0:
        raise ValueError("No reference vectors generated. Increase p_ref or check M.")
    V_unit = V / np.linalg.norm(V, axis=1, keepdims=True)

    # 3) ideal point
    F_curr = pop_costs(pop)
    ideal = get_ideal(F_curr)

    # main loop
    for gen in range(1, max_gen + 1):
        # 4) mating & variation -> produce offspring positions
        # random mating order
        mating_order = np.random.permutation(pop_size)
        offspring_positions = []
        for i in range(0, pop_size, 2):
            p1 = pop[mating_order[i % pop_size]]['Position']
            p2 = pop[mating_order[(i + 1) % pop_size]]['Position']
            if np.random.rand() < crossover_prob:
                c1, c2 = sbx_crossover(p1, p2, eta=eta_c, pc=1.0, xmin=xmin, xmax=xmax)
            else:
                c1, c2 = p1.copy(), p2.copy()
            c1 = polynomial_mutation(c1, eta=eta_m, pm=pm, xmin=xmin, xmax=xmax)
            c2 = polynomial_mutation(c2, eta=eta_m, pm=pm, xmin=xmin, xmax=xmax)
            offspring_positions.append(c1); offspring_positions.append(c2)
        offspring_positions = np.array(offspring_positions)[:pop_size]

        # 5) evaluate offspring
        F_off = obj_func(offspring_positions)

        # 6) combine pop and offspring into transient arrays for selection
        X_all = np.vstack([pop_positions(pop), offspring_positions])   # (2N x n_var)
        F_all = np.vstack([pop_costs(pop), F_off])                    # (2N x n_obj)

        # 7) update ideal
        ideal = np.min(F_all, axis=0)

        # 8) normalization
        nadir = np.max(F_all, axis=0)
        F_norm = normalize_objectives(F_all, ideal, nadir)

        # 9) associate each individual to a reference vector by minimal angle (max cosine)
        # prepare dir unit vectors (handle zero rows)
        dir_norms = np.linalg.norm(F_norm, axis=1, keepdims=True)
        # avoid division by zero
        dir_unit = np.where(dir_norms == 0, 0.0, F_norm / dir_norms)
        cos = dir_unit.dot(V_unit.T)  # (2N) x K
        assoc = np.argmax(cos, axis=1)  # index of associated reference vector for each individual

        # 10) PBI-based selection per reference vector
        t = gen
        T = max_gen
        theta = theta0 * (t / T) ** alpha

        chosen_indices = []
        K = V_unit.shape[0]
        for k in range(K):
            members_idx = np.where(assoc == k)[0]
            if members_idx.size == 0:
                continue
            F_members = F_norm[members_idx]
            d1, d2 = pbi_values(F_members, V_unit[k])
            fitness = d1 + theta * d2
            best_local = members_idx[np.argmin(fitness)]
            chosen_indices.append(best_local)

        # 11) fill up or trim to pop_size
        if len(chosen_indices) < pop_size:
            remaining = np.setdiff1d(np.arange(F_all.shape[0]), chosen_indices)
            # Use simple global score: sum of normalized objectives (lower better)
            global_score = np.sum(F_norm, axis=1)
            fill_order = remaining[np.argsort(global_score[remaining])]
            need = pop_size - len(chosen_indices)
            chosen_indices.extend(list(fill_order[:need]))
        elif len(chosen_indices) > pop_size:
            # trim (rare) by global score
            chosen_indices = chosen_indices[:pop_size]

        # 12) form next pop (as list of dicts)
        new_pop = []
        for idx in chosen_indices:
            new_pop.append({
                'Position': X_all[idx].copy(),
                'Cost'    : F_all[idx].copy()
            })
        pop = new_pop

        # optional logging
        if gen % max(1, max_gen // 10) == 0 or gen == 1 or gen == max_gen:
            print(f"gen {gen}/{max_gen}, theta={theta:.4f}, ideal={ideal}")

    return pop, V_unit, ideal

# ---------- Example problem: DTLZ2 (kept the same) ----------
def dtlz2(X, n_obj=3):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    pop, nvar = X.shape
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

# ---------- Quick run example ----------
if __name__ == "__main__":
    n_obj = 3
    n_var = 12
    pop_size = 100
    max_gen = 200

    def objf_pop(X):
        return dtlz2(X, n_obj=n_obj)

    pop, V, ideal = rvea_pop(objf_pop, n_var=n_var, n_obj=n_obj,
                             pop_size=pop_size, max_gen=max_gen, p_ref=12,
                             crossover_prob=0.9, eta_c=20, eta_m=20, pm=None,
                             theta0=5.0, alpha=2.0, xl=0.0, xu=1.0, seed=42)

    # plot final front from pop

    final_costs = pop_costs(pop)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(final_costs[:, 0], final_costs[:, 1], final_costs[:, 2], s=10)
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_zlabel('f3')
    plt.title("RVEA final front (pop-only)")
    ax.view_init(elev=25, azim=35)
    plt.show()
