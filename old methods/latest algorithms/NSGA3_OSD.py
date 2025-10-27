try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass

import numpy as np
import pyvista as pv
from utils.Domination_functions import NS_Sort
from utils.GA_functions import sbx_crossover, polynomial_mutation
from utils.Decompose_functions import das_dennis_generate


# ============================================================
# OSD without sklearn
# ============================================================

def associate_to_reference_OSD(F, ref_dirs, ideal):
    Z = F - ideal
    # Normalize objective vectors
    Znorm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    # Normalize ref directions
    rd = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-12)

    # Similarity projection
    proj = np.dot(Znorm, rd.T)
    proj_vec = proj[:, :, None] * rd[None, :, :]
    # perpendicular distance as diversity measure
    dist = np.linalg.norm(Znorm[:, None, :] - proj_vec, axis=2)

    ref_idx = np.argmin(dist, axis=1)
    d_perp = dist[np.arange(dist.shape[0]), ref_idx]

    return ref_idx, d_perp

def osd_selection(F, fronts, pop_size, ideal, ref_dirs):
    chosen = []

    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            chosen.extend(front)
        else:
            needed = pop_size - len(chosen)
            last = np.array(front)
            lastF = F[last]

            # Decomposition assignment
            ref_idx, dpp = associate_to_reference_OSD(lastF, ref_dirs, ideal)

            selected = []
            K = ref_dirs.shape[0]

            # chọn mỗi region một đại diện (nhỏ nhất dpp)
            for k in range(K):
                idx_k = np.where(ref_idx == k)[0]
                if len(idx_k) == 0:
                    continue
                best_local = idx_k[np.argmin(dpp[idx_k])]
                selected.append(best_local)
                if len(selected) >= needed:
                    break

            if len(selected) < needed:
                remain = np.setdiff1d(np.arange(len(last)), selected)
                fill_order = np.argsort(dpp[remain])
                need_more = needed - len(selected)
                fill = remain[fill_order[:need_more]]
                selected = np.concatenate([selected, fill])

            chosen.extend(list(last[selected]))
            break

    return chosen



# ============================================================
# Problem definition (unchanged)
# ============================================================
def problem_eval(x):
    x = np.atleast_2d(x)
    g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
    f1 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
    f2 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
    f3 = (1 + g) * np.sin(0.5 * np.pi * x[:, 0])
    return np.vstack([f1, f2, f3]).T.squeeze()


# ============================================================
# Main NSGA-III-OSD loop
# ============================================================
D = 12
n_obj = 3
pop_size = 100
max_gen = 100
xmin = np.zeros(D)
xmax = np.ones(D)
ref_divisions = 11
sbx_eta = 30
mut_eta = 20
pm = None
seed = 3

if seed is not None:
    np.random.seed(seed)

# keep ref_dirs for comparability (not used in OSD)
ref_dirs = das_dennis_generate(n_obj, ref_divisions)

# %% initialize pop
pop = []
for i in range(pop_size):
    position = np.random.rand(D) * (xmax - xmin) + xmin
    cost = problem_eval(position)
    ind = {'Position': position, 'Cost': cost,
           'DominationSet': set(), 'DominationCount': 0, 'Rank': 0}
    pop.append(ind)
    
# %% main loop
for gen in range(max_gen):
    # 1) Mating: use actual parent pool size; produce exactly pop_size offspring
    offspring = []
    N_parent = len(pop)
    # create a random parent sequence (with wrap) of length >= pop_size
    parents_idx = np.random.permutation(N_parent)
    # if sequence too short, tile it
    while parents_idx.size < pop_size:
        parents_idx = np.concatenate([parents_idx, np.random.permutation(N_parent)])
    parents_idx = parents_idx[:pop_size]

    # pair parents sequentially (0-1, 2-3, ...) and wrap the partner if odd
    i = 0
    while len(offspring) < pop_size:
        idx_a = parents_idx[i % parents_idx.size]
        idx_b = parents_idx[(i + 1) % parents_idx.size]
        p1 = pop[int(idx_a)]['Position']
        p2 = pop[int(idx_b)]['Position']
        c1, c2 = sbx_crossover(p1, p2, eta=sbx_eta, pc=1.0, xmin=xmin, xmax=xmax)
        c1 = polynomial_mutation(c1, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        c2 = polynomial_mutation(c2, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        offspring.append({'Position': c1, 'Cost': problem_eval(c1),
                          'DominationSet': set(), 'DominationCount': 0, 'Rank': 0})
        if len(offspring) < pop_size:
            offspring.append({'Position': c2, 'Cost': problem_eval(c2),
                              'DominationSet': set(), 'DominationCount': 0, 'Rank': 0})
        i += 2

    # 2) Merge
    pop_all = pop + offspring
    F_all = np.array([ind['Cost'] for ind in pop_all])
    ideal = np.min(F_all, axis=0)

    # 3) Non-dominated sorting
    fronts = NS_Sort(pop_all)

    # 4) Selection using OSD
    chosen_indices = osd_selection(F_all, fronts, pop_size, ideal, ref_dirs)
    pop = [pop_all[i] for i in chosen_indices]

    # 5) Print progress
    if (gen + 1) % max(1, max_gen // 10) == 0 or gen == 0:
        print(f"Gen {gen+1:4d}: pop size {len(pop)}")

# %% Final plot
F = np.array([ind['Cost'] for ind in pop])
points = F[:, :3]  # f1, f2, f3
cloud = pv.PolyData(points)
points = ref_dirs[:, :3]  # f1, f2, f3
cloud2 = pv.PolyData(points)

# gen plotter
plotter = pv.Plotter()
plotter.add_points(
    cloud,
    color="red",                # color
    point_size=9,                # size
    render_points_as_spheres=True  # sphere point
)
# # gen plotter 
# plotter.add_points(
#     cloud2,
#     color="blue",                # color
#     point_size=8,                # size
#     render_points_as_spheres=True  # sphere point
# )
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
plotter.add_text("NSGA3 Pareto Front", position='upper_edge', font_size=14, color='black')
plotter.view_vector((-35, -25, 1))  # try view_isometric(), view_yx(),...
plotter.show(title="NSGA3 Pareto Front 3D")