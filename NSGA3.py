try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
"""
nsga3.py
A compact NSGA-III implementation (numpy only).
- SBX crossover + polynomial mutation
- non-dominated sorting
- Das & Dennis reference points
- reference-association selection (niche filling) for the last front

NSGA-III (numpy-only) — fixed version with 'pop' structure (list of dicts).
Fixes:
 - avoid IndexError when population temporarily not equal to pop_size
 - mating uses actual parent pool size; always generate exactly pop_size offspring
 - selection/niching uses indices relative to merged population (pop_all)
 - ensures new population length == pop_size
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.Domination_functions import NS_Sort
from utils.GA_functions import sbx_crossover, polynomial_mutation
from utils.Decompose_functions import das_dennis_generate


# ----------------------------
# Association & niching
# ----------------------------
def associate_to_reference(F, ref_dirs, ideal):
    Z = F - ideal
    Znorm = Z / (np.max(np.abs(Z), axis=0, keepdims=True) + 1e-12)
    rd = ref_dirs / (np.linalg.norm(ref_dirs, axis=1, keepdims=True) + 1e-12)
    projections = np.dot(Znorm, rd.T)
    proj_vecs = projections[:, :, None] * rd[None, :, :]
    perps = np.linalg.norm(Znorm[:, None, :] - proj_vecs, axis=2)
    ref_idx = np.argmin(perps, axis=1)
    dist = perps[np.arange(perps.shape[0]), ref_idx]
    return ref_idx, dist

def niching_selection(F, ref_dirs, ideal, chosen_indices, last_front, pop_size):
    """
    chosen_indices, last_front: lists of indices relative to F (i.e. indices in pop_all / F_all)
    returns: list of chosen indices (indices in same coordinate system as chosen_indices / last_front)
    """
    remaining = pop_size - len(chosen_indices)
    if remaining <= 0:
        return []

    all_idx = np.array(list(chosen_indices) + list(last_front), dtype=int)
    F_all = F[all_idx]
    ref_idx_all, dist_all = associate_to_reference(F_all, ref_dirs, ideal)
    K = ref_dirs.shape[0]

    niche_count = np.zeros(K, dtype=int)
    if len(chosen_indices) > 0:
        chosen_refs = ref_idx_all[:len(chosen_indices)]
        for r in chosen_refs:
            niche_count[r] += 1

    cand_rel_idx = np.arange(len(chosen_indices), len(all_idx))
    ref_to_cands = {k: [] for k in range(K)}
    for i_rel_idx, i_all_rel in enumerate(cand_rel_idx):
        r = int(ref_idx_all[i_all_rel])
        ref_to_cands[r].append(i_all_rel)

    selected_rel = []
    while remaining > 0:
        zero_refs = [r for r in range(K) if niche_count[r] == 0 and len(ref_to_cands[r]) > 0]
        if zero_refs:
            for r in zero_refs:
                if remaining == 0:
                    break
                cand_list = ref_to_cands[r]
                best_rel = min(cand_list, key=lambda idx_rel: dist_all[idx_rel])
                selected_rel.append(best_rel)
                remaining -= 1
                niche_count[r] += 1
                for lst in ref_to_cands.values():
                    if best_rel in lst:
                        lst.remove(best_rel)
        else:
            viable_refs = [r for r in range(K) if len(ref_to_cands[r]) > 0]
            if not viable_refs:
                break
            min_n = min(niche_count[r] for r in viable_refs)
            refs_min = [r for r in viable_refs if niche_count[r] == min_n]
            r = np.random.choice(refs_min)
            cand_list = ref_to_cands[r]
            best_rel = min(cand_list, key=lambda idx_rel: dist_all[idx_rel])
            selected_rel.append(best_rel)
            remaining -= 1
            niche_count[r] += 1
            for lst in ref_to_cands.values():
                if best_rel in lst:
                    lst.remove(best_rel)

    # convert relative indices back to indices in the original F_all (i.e., indices in all_idx)
    selected = [int(all_idx[int(rel)]) for rel in selected_rel]
    return selected

# ----------------------------
# Problem: DTLZ2 (3 objectives)
# ----------------------------
def problem_eval(x):
    x = np.atleast_2d(x)
    g = np.sum((x[:, 2:] - 0.5) ** 2, axis=1)
    f1 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
    f2 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
    f3 = (1 + g) * np.sin(0.5 * np.pi * x[:, 0])
    return np.vstack([f1, f2, f3]).T.squeeze()

# ----------------------------
# Main NSGA-III loop (fixed)
# ----------------------------
D = 12
n_obj = 3
pop_size = 100
max_gen = 100
xmin = np.zeros(D)
xmax = np.ones(D)
ref_divisions = 10
sbx_eta = 30
mut_eta = 20
pm = None
seed = 1

if seed is not None:
    np.random.seed(seed)

ref_dirs = das_dennis_generate(n_obj, ref_divisions)

# initialize pop (list of dicts)
pop = []
for i in range(pop_size):
    position = np.random.rand(D) * (xmax - xmin) + xmin
    cost = problem_eval(position)
    ind = {'Position': position, 'Cost': cost,
           'DominationSet': set(), 'DominationCount': 0, 'Rank': 0}
    pop.append(ind)

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

    # 3) Non-dominated sorting on merged pop_all
    # build a temporary pop_all copy to use NS_Sort which modifies DominationSet/Count
    # We can operate directly on pop_all
    fronts = NS_Sort(pop_all)

    # 4) Build new population ensuring exactly pop_size individuals
    chosen_rel_indices = []  # indices relative to pop_all
    for front in fronts:
        if len(chosen_rel_indices) + len(front) <= pop_size:
            chosen_rel_indices.extend(front)
        else:
            # need to pick some from this front
            needed = pop_size - len(chosen_rel_indices)
            chosen = niching_selection(F_all, ref_dirs, ideal, chosen_rel_indices, front, pop_size)
            # niching_selection returns indices in coordinates of pop_all already
            chosen_rel_indices.extend(chosen)
            break

    # If still short (shouldn't happen often), fill with best remaining (by rank then distance)
    if len(chosen_rel_indices) < pop_size:
        remaining_pool = [i for i in range(len(pop_all)) if i not in chosen_rel_indices]
        # sort remaining by rank (pop_all[i]['Rank']) and then by sum of objectives (as tie-breaker)
        remaining_pool_sorted = sorted(remaining_pool, key=lambda ii: (pop_all[ii]['Rank'], np.sum(pop_all[ii]['Cost'])))
        to_add = remaining_pool_sorted[:(pop_size - len(chosen_rel_indices))]
        chosen_rel_indices.extend(to_add)

    # form new pop preserving dict objects (or create shallow copies if desired)
    pop = [pop_all[i] for i in chosen_rel_indices]

    # Print progress
    if (gen + 1) % max(1, max_gen // 10) == 0 or gen == 0:
        ideal_now = np.min([ind['Cost'] for ind in pop], axis=0)
          

# %% Final plot
F = np.array([ind['Cost'] for ind in pop])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10)
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
plt.title("NSGA-III Approximate Pareto Front (fixed)")
ax.view_init(elev=25, azim=35)
plt.show()
