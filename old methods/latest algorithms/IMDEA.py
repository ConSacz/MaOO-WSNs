try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import pyvista as pv
from utils.Domination_functions import check_domination, nondominated_front
from utils.GA_functions import crossover_binomial, crossover_exponential
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
# %% ---------- dominance helpers ----------

def prune_archive(archive, max_size):
    if len(archive) <= max_size:
        return archive
    F = np.array([ind['Cost'] for ind in archive])
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
        nd_objs = np.array([ind['Cost'] for ind in nd_inds])
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
    F = np.array([ind['Cost'] for ind in pop])
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
    return v.flatten()

def mutation_current_to_phi_best_no_archive(pop, idx, phi_idx, r1_idx, r3_idx, F):
    x_i = pop[idx]['Position']
    x_phi = pop[phi_idx]['Position']
    x_r1 = pop[r1_idx]['Position']
    x_r3 = pop[r3_idx]['Position']
    v = x_i + F * (x_phi - x_i + x_r1 - x_r3)
    return v.flatten()

def mutation_weighted_rand_to_phi_best(pop, idx, x_phi, x_r1, x_r3, F):
    v = F * pop[x_r1]['Position'] + (pop[x_phi]['Position'] - pop[x_r3]['Position'])
    return v.flatten()

# ---------- Problem ----------
def obj_func(X, n_obj=3):
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

# ---------- Main Parameters ----------
n_obj = 3
dim = 12
bounds = (0, 1)
max_fes = 20000
seed = 2
NP_init = 200
NP_min = 100
archive_rate = 2.6
use_local_search = True

# rng
global _rng
_rng = np.random.default_rng(seed)
xl, xu = bounds
FES = 0

# archive
archive = []

# population init
pop = []
for _ in range(NP_init):
    pos = _rng.random(dim) * (xu - xl) + xl
    cost = obj_func(pos.reshape(1, -1))[0]
    pop.append({'Position': pos, 'Cost': cost})
FES += NP_init

archive_size = max(1, int(np.round(archive_rate * dim)))

# operator setup
n_ops = 3
NPop = np.full(n_ops, NP_init // n_ops, dtype=int)
for i in range(NP_init - np.sum(NPop)):
    NPop[i % n_ops] += 1

Pls = 0.1
CFEls = max(1, int(0.01 * max_fes))
history = {'best': [], 'FES': []}

gen = 0

# ---------- Main loop ----------
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

            fu = obj_func(u.reshape(1, -1))[0]
            FES += 1

            # selection
            if check_domination(fu, pop[pid]['Cost']) == 1:
                archive.append(pop[pid].copy())
                if len(archive) > archive_size:
                    archive = prune_archive(archive, archive_size)
                pop[pid] = {'Position': u, 'Cost': fu}

    # best individual
    F = np.array([ind['Cost'] for ind in pop])
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

# ---------- Final results ----------
nd_mask = nondominated_front(np.array([ind['Cost'] for ind in pop]))
nd = [pop[i] for i, m in enumerate(nd_mask) if m]
print("Nondominated count:", len(nd))

# %%---------- Plot ----------

F = np.array([ind['Cost'] for ind in pop])
points = F[:, :3]  # f1, f2, f3
cloud = pv.PolyData(points)

# gen plotter
plotter = pv.Plotter()
plotter.add_points(
    cloud,
    color="blue",                # color
    point_size=8,                # size
    render_points_as_spheres=True  # sphere point
)
plotter.show_grid(
    xtitle='f1',
    ytitle='f2',
    ztitle='f3',
    color='gray',
    grid='back',     # vẽ lưới phía sau điểm
    location='outer' # hiển thị nhãn ngoài khung
)
plotter.show_bounds(grid='front', color='black')
plotter.add_axes(line_width=10)
plotter.add_text("IMODE Pareto Front", position='upper_edge', font_size=14, color='black')
plotter.view_vector((-35, -25, 1))  # try view_isometric(), view_yx(),...
plotter.show(title="IMODE Pareto Front 3D")

