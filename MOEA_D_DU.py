# moeaddu_pop_only.py
# MOEA/D-DU implementation that uses only `pop` (list of dicts) — no X, F arrays.
import numpy as np
import pyvista as pv
from utils.Decompose_functions import das_dennis_generate, tchebycheff, vertical_distance

# -------------------------
# Problem evaluation
# -------------------------
def problem_eval(x):
    x = np.atleast_2d(x)
    g = np.sum((x[:, 2:] - 0.5)**2, axis=1)
    f1 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
    f2 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
    f3 = (1 + g) * np.sin(0.5 * np.pi * x[:, 0])
    F =np.vstack([f1, f2, f3]).T
    return F.squeeze()

# -------------------------
# DE operator variant that works directly from pop (rand/1/bin)
# returns children positions array shape (nPop, D)
# -------------------------
def de_rand1_bin_pop(pop, F=0.5, CR=0.9, xmin=None, xmax=None):
    # pop: list of dicts, each dict has 'Position' (ndarray)
    positions = np.array([p['Position'] for p in pop])  # (N, D)
    N, D = positions.shape
    idx = np.arange(N)
    children = np.empty_like(positions)
    for i in range(N):
        # choose three distinct indices != i
        a, b, c = np.random.choice(idx[idx != i], 3, replace=False)
        mutant = positions[a] + F * (positions[b] - positions[c])
        # binomial crossover
        cross = np.random.rand(D) < CR
        jrand = np.random.randint(D)
        cross[jrand] = True
        trial = np.where(cross, mutant, positions[i])
        # bounds (support scalar or arrays)
        if xmin is not None:
            trial = np.maximum(trial, xmin)
        if xmax is not None:
            trial = np.minimum(trial, xmax)
        children[i, :] = trial
    return children

# -------------------------
# Utilities for pop creation / extraction
# -------------------------
def init_pop_uniform(nPop, D, xmin=0.0, xmax=1.0, eval_func=None):
    pop = []
    X = np.random.rand(nPop, D) * (xmax - xmin) + xmin
    if eval_func is None:
        raise ValueError("eval_func required to initialize costs")
    F = np.array([eval_func(X[i]) for i in range(nPop)])
    for i in range(nPop):
        pop.append({'Position': X[i].copy(), 'Cost': F[i].copy()})
    return pop

def pop_positions(pop):
    return np.array([p['Position'] for p in pop])

def pop_costs(pop):
    return np.array([p['Cost'] for p in pop])

# -------------------------
# Main MOEA/D-DU loop (pop-only)
# -------------------------
def moead_du(problem_eval,
                      D=12,
                      n_obj=3,
                      nPop=100,
                      max_gen=200,
                      neigh_size=20,
                      nr=2,
                      Fm=0.5,
                      CR=0.9,
                      xmin=0.0,
                      xmax=1.0,
                      seed=None):
    if seed is not None:
        np.random.seed(seed)

    # 1) weight vectors and neighborhoods
    # W = uniform_weights(nPop, n_obj)
    W = das_dennis_generate(n_obj, 13)
    # optionally remove random rows if W too large (kept from your prior code)
    if W.shape[0] > nPop:
        rows_to_delete = np.random.choice(W.shape[0], W.shape[0] - nPop, replace=False)
        W = np.delete(W, rows_to_delete, axis=0)
    # if still not equal to nPop, adjust nPop to W length
    if W.shape[0] != nPop:
        nPop = W.shape[0]

    distW = np.linalg.norm(W[:, None, :] - W[None, :, :], axis=2)
    neighborhoods = np.argsort(distW, axis=1)[:, :neigh_size]

    # 2) initialize pop
    pop = init_pop_uniform(nPop, D, xmin=xmin, xmax=xmax, eval_func=problem_eval)

    # ideal point
    z = np.min(pop_costs(pop), axis=0)

    # main loop
    for gen in range(max_gen):
        # create offspring positions via DE (one offspring per subproblem)
        U = de_rand1_bin_pop(pop, F=Fm, CR=CR, xmin=xmin, xmax=xmax)  # (nPop, D)
        FU = np.array([problem_eval(U[i]) for i in range(nPop)])     # (nPop, n_obj)

        # update ideal point
        z = np.minimum(z, np.min(FU, axis=0))

        # for each subproblem i, apply DU update using its neighborhood
        for i in range(nPop):
            child_pos = U[i].copy()
            child_f = FU[i].copy()
            cand_idx = neighborhoods[i].copy()

            # compute vertical distance between candidate solutions' objectives and their weight vectors
            dists = np.zeros_like(cand_idx, dtype=float)
            for k, j in enumerate(cand_idx):
                dists[k] = vertical_distance(pop[j]['Cost'], W[j], ref=z)

            order = np.argsort(dists)  # ascending
            replaced = 0
            for idx_in_order in order:
                j = cand_idx[idx_in_order]
                val_child = tchebycheff(child_f, W[j], z)
                val_j = tchebycheff(pop[j]['Cost'], W[j], z)
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
        if (gen + 1) % 1 == 0:
            print(f"Gen {gen+1:4d}: pop size {len(pop)}")

    return pop, W, z

# -------------------------
# Example run (parameters similar to your original)
# -------------------------
if __name__ == "__main__":
    # params
    D = 12
    n_obj = 3
    nPop = 100
    max_gen = 200
    neigh_size = 20
    nr = 2
    Fm = 0.5
    CR = 0.9
    xmin = 0.0
    xmax = 1.0
    seed = 0

    pop, W, z = moead_du(problem_eval,
                                  D=D,
                                  n_obj=n_obj,
                                  nPop=nPop,
                                  max_gen=max_gen,
                                  neigh_size=neigh_size,
                                  nr=nr,
                                  Fm=Fm,
                                  CR=CR,
                                  xmin=xmin,
                                  xmax=xmax,
                                  seed=seed)

    # %%plot final front from pop
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
    plotter.add_text("MOEAD-DU Pareto Front", position='upper_edge', font_size=14, color='black')
    plotter.view_vector((-35, -25, 1))  # try view_isometric(), view_yx(),...
    plotter.show(title="MOEAD-DU Pareto Front 3D")
