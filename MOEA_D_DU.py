# moeaddu_pop_only.py
# MOEA/D-DU implementation that uses only `pop` (list of dicts) — no X, F arrays.
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass

# %%
import numpy as np
from utils.Decompose_functions import das_dennis_generate, tchebycheff, vertical_distance
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
from utils.Plot_functions import plot3D, plot3D_adjustable, plot_deployment2D

# %% DE operator variant that works directly from pop (rand/1/bin)
# returns children positions array shape (nPop, D)

def de_rand1_bin_pop(pop, CR=0.9, xmin=None, xmax=None):
    # pop: list of dicts, each dict has 'Position' (ndarray)
    positions = np.array([p['Position'] for p in pop])  # (N, D)
    nPop, N, D = positions.shape
    idx = np.arange(nPop)
    children = np.empty_like(positions)
    for i in range(nPop):
        # choose three distinct indices != i
        a, b, c = np.random.choice(idx[idx != i], 3, replace=False)
        F = np.random.uniform(-1, 1, (N, D))
        mutant = positions[a] + F * (positions[b] - positions[c])
        # binomial crossover
        cross = np.random.rand(N, D) < CR
        jrand = np.random.randint(N)
        cross[jrand] = True
        trial = np.where(cross, mutant, positions[i])
        trial[:,2] = np.clip(trial[:,2], 8, 12)
        trial[:,:2] = np.clip(trial[:,:2],xmin, xmax)
        children[i, :] = trial
    return children

# -------------------------
# Utilities for pop creation / extraction
# -------------------------

def pop_positions(pop):
    return np.array([p['Position'] for p in pop])

def pop_costs(pop):
    return np.array([p['Cost'] for p in pop])


# %% ---------- Main Parameters ----------
# algorithm parameter
n_obj = 3
nPop = 200
max_fes = 50000
neigh_size = 20
nr = 2
CR = 0.9
xmin = 0
xmax = 100
seed = 3
# Network Parameter
N = 60
rc = 20
stat = np.zeros((2, N))  # tạo mảng 2xN
stat[1, 0] = rc         # rc
rs = (8,12)
sink = np.array([xmax//2, xmax//2])
RP = np.zeros((3, 2))   
RP[:,0] = [1, 1, 1]          # first col are ideal values
RP[:,1] = [0.00001, 0.00001, 0.00001]    # second col are nadir values

# %% Initialization
if seed is not None:
    np.random.seed(seed)

# 1) weight vectors and neighborhoods
# W = uniform_weights(nPop, n_obj)
W = das_dennis_generate(n_obj, 19)
# optionally remove random rows if W too large (kept from your prior code)
if W.shape[0] > nPop:
    rows_to_delete = np.random.choice(W.shape[0], W.shape[0] - nPop, replace=False)
    W = np.delete(W, rows_to_delete, axis=0)
# if still not equal to nPop, adjust nPop to W length
if W.shape[0] != nPop:
    nPop = W.shape[0]

distW = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
neighborhoods = np.argsort(distW, axis=1)[:, :neigh_size]

# environment
Covered_Area = np.zeros((xmax, xmax), dtype=int)
Obstacle_Area = np.ones((xmax, xmax), dtype=int)

# 2) initialize pop
FES=0
pop = []
for _ in range(nPop):
    alpop = np.zeros((N, 3))
    pos0 = np.random.uniform(20, 80, (N, 2))
    pos0[0] = sink
    rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
    alpop[:,:2] = pos0
    alpop[:,2] = rs0[:, 0]
    alpop_cost = CostFunction_3F1C_MOO(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
    RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
    RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
    pop.append({'Position': alpop, 'Cost': alpop_cost})
FES += nPop
# ideal point
RP[:,0] = (np.min(pop_costs(pop), axis=0)).T.flatten()
RP[:,1] = (np.max(pop_costs(pop), axis=0)).T.flatten()

# %% main loop
gen = 0
while FES < max_fes:
    gen+=1
    # create offspring positions via DE (one offspring per subproblem)
    U = de_rand1_bin_pop(pop, CR=CR, xmin=xmin, xmax=xmax)  # (nPop, D)
    FU = np.array([CostFunction_3F1C_MOO(U[i], stat, RP, Obstacle_Area, Covered_Area.copy()) for i in range(nPop)])     # (nPop, n_obj)
    FES += nPop
    # update reference point
    F = FU[:,0]
    z0 = np.minimum(RP[:,0].reshape(1,-1), np.min(FU, axis=0))
    z1 = np.maximum(RP[:,1].reshape(1,-1), np.max(FU, axis=0))
    RP[:,0] = z0.T.flatten()
    RP[:,1] = z1.T.flatten()

    # for each subproblem i, apply DU update using its neighborhood
    for i in range(nPop):
        child_pos = U[i].copy()
        child_f = FU[i].copy()
        cand_idx = neighborhoods[i].copy()

        # compute vertical distance between candidate solutions' objectives and their weight vectors
        dists = np.zeros_like(cand_idx, dtype=float)
        for k, j in enumerate(cand_idx):
            dists[k] = vertical_distance(pop[j]['Cost'], W[j], ref=z0)

        order = np.argsort(dists)  # ascending
        replaced = 0
        for idx_in_order in order:
            j = cand_idx[idx_in_order]
            val_child = tchebycheff(child_f, W[j], RP)
            val_j = tchebycheff(pop[j]['Cost'], W[j], RP)
            if val_child < val_j:
                pop[j]['Position'] = child_pos.copy()
                pop[j]['Cost'] = child_f.copy()
                replaced += 1
                if replaced >= nr:
                    break

    # optional: shuffle subproblems to avoid bias
    perm = np.random.permutation(nPop)
    pop = [pop[i] for i in perm]
    W = W[perm]
    # recompute neighborhoods properly after permuting W
    distW = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    neighborhoods = np.argsort(distW, axis=1)[:, :neigh_size]

    # logging
    print(f"Gen {gen}, FES: {FES}: pop size {len(pop)}")
    plot3D(pop)
    
# %%---------- final Plot ----------
plot_name = 'MOEA-D-DU'
plot3D_adjustable(pop, plot_name)
