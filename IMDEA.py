try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
from utils.Domination_functions import check_domination, nondominated_front
from utils.GA_functions import crossover_binomial, crossover_exponential
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
from utils.Normalize_functions import global_normalized
from utils.Plot_functions import plot2D, plot3D, plot3D_adjustable

# %% ---------- dominance helpers ----------

def prune_archive(archive, RP, max_size):
    if len(archive) <= max_size:
        return archive
    F = np.array([ind['Cost'] for ind in archive])[:, 0]
    F = global_normalized(F, RP)
    mask_nd = nondominated_front(F)
    nd_inds = [archive[i] for i, m in enumerate(mask_nd) if m]
    if len(nd_inds) <= max_size:
        rem = [archive[i] for i, m in enumerate(mask_nd) if not m]
        if rem:
            sums = np.array([np.sum(ind['Cost']) for ind in rem])
            order = np.argsort(sums)
            nd_inds += [rem[i] for i in order[:max_size - len(nd_inds)]]
        return nd_inds
    else:
        nd_objs = np.array([ind['Cost'] for ind in nd_inds])[:, 0]
        sums = np.sum(nd_objs, axis=1)
        order = np.argsort(sums)
        return [nd_inds[i] for i in order[:max_size]]

# ---------- helpers ----------

def ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x

def clamp(X, xl, xu):
    return np.minimum(np.maximum(X, xl), xu)

def get_phi_idx(pop):
    F = np.array([ind['Cost'] for ind in pop])[:, 0]
    scores = np.sum(F, axis=1)
    N = len(pop)
    top_k = max(1, int(np.ceil(0.1 * N)))
    best_rows = np.argsort(scores)[:top_k]
    return int(_rng.choice(best_rows))

# ---------- mutation operators ----------

def mutation_current_to_phi_best_with_archive(pop, idx, phi_idx, r1_idx, r2_vec, F):
    x_i = pop[idx]['Position']
    x_phi = pop[phi_idx]['Position']
    x_r1 = pop[r1_idx]['Position']
    v = x_i + F * (x_phi - x_i + x_r1 - r2_vec)
    return v

def mutation_current_to_phi_best_no_archive(pop, idx, phi_idx, r1_idx, r3_idx, F):
    x_i = pop[idx]['Position']
    x_phi = pop[phi_idx]['Position']
    x_r1 = pop[r1_idx]['Position']
    x_r3 = pop[r3_idx]['Position']
    v = x_i + F * (x_phi - x_i + x_r1 - x_r3)
    return v  

def mutation_weighted_rand_to_phi_best(pop, idx, x_phi, x_r1, x_r3, F):
    v = F * pop[x_r1]['Position'] + (pop[x_phi]['Position'] - pop[x_r3]['Position'])
    return v

# ---------- Cost Function ----------
def CostFunction(pop, stat, RP, Obstacle_Area, Covered_Area):
    return CostFunction_3F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area)
# %% ---------- Main Parameters ----------
# algorithm parameter
bounds = (0, 100)
xl, xu = bounds
max_fes = 10000
seed = 2
NP_init = 200
NP_min = 100
archive_rate = 0.5
# Network Parameter
N = 60
rc = 20
stat = np.zeros((2, N), dtype=float)  # tạo mảng 2xN
stat[1, 0] = rc         # rc
rs = (8,12)
sink = np.array([xu//2, xu//2])
RP = np.zeros((3, 2))   
RP[:,0] = [1, 1, 1]          # first col are ideal values
RP[:,0] = [0.1, 0.1, 0.1]    # second col are nadir values

# %% Initialization
# rng
global _rng
_rng = np.random.default_rng(seed)

# environment
Covered_Area = np.zeros((xu, xu), dtype=int)
Obstacle_Area = np.ones((xu, xu), dtype=int)

# archive
archive = []

# population init
FES = 0
pop = []
for _ in range(NP_init):
    alpop = np.zeros((N, 3))
    pos0 = np.random.uniform(10, 90, (N, 2)) 
    pos0[0] = sink
    rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
    alpop[:,:2] = pos0
    alpop[:,2] = rs0[:, 0]
    alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
    RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
    RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
    pop.append({'Position': alpop, 'Cost': alpop_cost})
FES += NP_init

archive_size = max(1, int(np.round(archive_rate * NP_min)))

# operator setup
n_ops = 3
NPop = np.full(n_ops, NP_init // n_ops, dtype=int)
for i in range(NP_init - np.sum(NPop)):
    NPop[i % n_ops] += 1

Pls = 0.1
CFEls = max(1, int(0.01 * max_fes))
history = {'best': [], 'FES': []}

gen = 0

# %% ---------- Main loop ----------
while FES < max_fes:
    gen += 1
    NP = len(pop)
    perm = _rng.permutation(NP)
    splits = np.split(perm, np.cumsum(NPop)[:-1])
    F_schedule = np.clip(_rng.normal(0.5, 0.3, NP), 0.05, 0.9)
    Cr_schedule = np.clip(_rng.random(NP), 0.0, 1.0)

    for op, idxs in enumerate(splits):
        if len(idxs) == 0:
            continue
        for pid in idxs:
            F_i = F_schedule[pid]
            Cr_i = Cr_schedule[pid]
            phi_idx = get_phi_idx(pop)

            idx_pool = list(set(range(NP)) - {pid})
            r = _rng.choice(idx_pool, size=3, replace=False)
            r1_idx, r2_idx, r3_idx = r[0], r[1], r[2]

            if op == 0:
                if len(archive) > 0 and _rng.random() < 0.5:
                    r2_vec = _rng.choice(archive)['Position']
                else:
                    r2_vec = pop[r2_idx]['Position']
                v = mutation_current_to_phi_best_with_archive(pop, pid, phi_idx, r1_idx, r2_vec, F_i)
            elif op == 1:
                v = mutation_current_to_phi_best_no_archive(pop, pid, phi_idx, r1_idx, r3_idx, F_i)
            else:
                v = mutation_weighted_rand_to_phi_best(pop, pid, phi_idx, r1_idx, r3_idx, F_i)

            v = clamp(v, xl, xu)
            if _rng.random() <= 0.5:
                u = crossover_binomial(pop[pid]['Position'], v, Cr_i)
            else:
                u = crossover_exponential(pop[pid]['Position'], v, Cr_i)

            u_cost = CostFunction(u, stat, RP, Obstacle_Area, Covered_Area.copy())
            RP[:,0] = np.minimum(RP[:,0], u_cost[0])
            RP[:,1] = np.maximum(RP[:,1], u_cost[0])
            # u_cost = global_normalized(u_cost, RP)
            FES += 1

            # selection
            if check_domination(u_cost, pop[pid]['Cost']) == 1:
                archive.append(pop[pid].copy())
                if len(archive) > archive_size:
                    archive = prune_archive(archive, RP, archive_size)
                pop[pid] = {'Position': u, 'Cost': u_cost}

    # best individual
    F = np.array([ind['Cost'] for ind in pop])[:, 0]
    nd_mask = nondominated_front(F)
    nd_indices = np.where(nd_mask)[0]
    if len(nd_indices) > 0:
        sums = np.sum(F[nd_indices], axis=1)
        best_idx = nd_indices[np.argmin(sums)]
    else:
        best_idx = np.argmin(np.sum(F, axis=1))
    best = pop[best_idx]
    history['best'].append(best)
    history['FES'].append(FES)

    print(f"Gen {gen}, FES={FES}")
    plot3D(pop)
# ---------- Final results ----------
nd_mask = nondominated_front(np.array([ind['Cost'] for ind in pop])[:, 0])
nd = [pop[i] for i, m in enumerate(nd_mask) if m]
print("Nondominated count:", len(nd))

# %%---------- Plot ----------

plot3D_adjustable(pop)
