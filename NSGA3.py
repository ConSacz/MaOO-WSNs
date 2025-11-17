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
 - avoid IndexError when population temporarily not equal to nPop
 - mating uses actual parent pool size; always generate exactly nPop offspring
 - selection/niching uses indices relative to merged population (pop_all)
 - ensures new population length == nPop
"""

import numpy as np
import time
from utils.Domination_functions import NS_Sort
from utils.GA_functions import sbx_crossover, polynomial_mutation
from utils.Decompose_functions import das_dennis_generate, associate_to_reference
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
from utils.Plot_functions import plot3D, plot3D_adjustable

# ----------------------------
# Association & niching
# ----------------------------

def niching_selection(F, W, ideal, chosen_indices, last_front, nPop):
    """
    chosen_indices, last_front: lists of indices relative to F (i.e. indices in pop_all / F_all)
    returns: list of chosen indices (indices in same coordinate system as chosen_indices / last_front)
    """
    remaining = nPop - len(chosen_indices)
    if remaining <= 0:
        return []

    all_idx = np.array(list(chosen_indices) + list(last_front), dtype=int)
    F_all = F[all_idx]
    ref_idx_all, dist_all = associate_to_reference(F_all, W, ideal)
    K = W.shape[0]

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

# ---------- Cost Function 3 functions 1 constraint
def CostFunction(pop, stat, RP, Obstacle_Area, Covered_Area):
    return CostFunction_3F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area)

# %%----------------------------
# Main NSGA-III loop (fixed)
# ----------------------------
n_obj = 3
nPop = 200
max_fes = 50000
p_ref = 19
sbx_eta = 30
mut_eta = 20
pm = None
seed = 1
xl=0
xu=100

# Network Parameter
N = 60
rc = 20
stat = np.zeros((2, N), dtype=float)  # tạo mảng 2xN
stat[1, 0] = rc         # rc
rs = (8,12)
sink = np.array([xu//2, xu//2])
RP = np.zeros((3, 2))   
RP[:,0] = [1, 1, 1]          # first col are ideal values
RP[:,1] = [0.00001, 0.00001, 0.00001]    # second col are nadir values

xmin = np.ones((N,3))*xl
xmin[:,2] = (np.ones((N,1))*rs[0]).flatten()
xmax = np.ones((N,3))*xu
xmax[:,2] = (np.ones((N,1))*rs[1]).flatten()

# %% Initilization
if seed is not None:
    np.random.seed(seed)

W = das_dennis_generate(n_obj, p_ref)

# environment
Covered_Area = np.zeros((xu, xu), dtype=int)
Obstacle_Area = np.ones((xu, xu), dtype=int)

# population init
FES = 0
pop = []
for k in range(nPop):
    alpop = np.zeros((N, 3))
    if k == 0:
        pos0 = np.random.uniform(30, 70, (N, 2))
    else:
        pos0 = np.random.uniform(10, 90, (N, 2)) 
    pos0[0] = sink
    rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
    alpop[:,:2] = pos0
    alpop[:,2] = rs0[:, 0]
    alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
    RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
    RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
    pop.append({'Position': alpop, 'Cost': alpop_cost})
FES += nPop

# %% main loop
gen = 0
while FES < max_fes:
    start_time = time.time()
    gen += 1
    # 1) Mating: use actual parent pool size; produce exactly nPop offspring
    offspring = []
    N_parent = len(pop)
    # create a random parent sequence (with wrap) of length >= nPop
    parents_idx = np.random.permutation(N_parent)
    # if sequence too short, tile it
    while parents_idx.size < nPop:
        parents_idx = np.concatenate([parents_idx, np.random.permutation(N_parent)])
    parents_idx = parents_idx[:nPop]

    # pair parents sequentially (0-1, 2-3, ...) and wrap the partner if odd
    i = 0
    while len(offspring) < nPop:
        idx_a = parents_idx[i % parents_idx.size]
        idx_b = parents_idx[(i + 1) % parents_idx.size]
        p1 = pop[int(idx_a)]['Position']
        p2 = pop[int(idx_b)]['Position']
        c1, c2 = sbx_crossover(p1, p2, eta=sbx_eta, pc=1.0, xmin=xmin, xmax=xmax)
        c1 = polynomial_mutation(c1, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        c2 = polynomial_mutation(c2, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        offspring.append({'Position': c1, 'Cost': CostFunction(c1, stat, RP, Obstacle_Area, Covered_Area.copy()),
                          'DominationSet': set(), 'DominationCount': 0, 'Rank': 0})
        if len(offspring) < nPop:
            offspring.append({'Position': c2, 'Cost': CostFunction(c2, stat, RP, Obstacle_Area, Covered_Area.copy()),
                              'DominationSet': set(), 'DominationCount': 0, 'Rank': 0})
        i += 2
    FES += i
    # 2) Merge
    pop_all = pop + offspring
    F_all = np.array([ind['Cost'] for ind in pop_all])
    
    # Update ideal
    ideal = np.min(F_all, axis=0)
    nadir = np.max(F_all, axis=0)
    RP[:,0] = ideal.T.flatten()
    RP[:,1] = nadir.T.flatten()

    # 3) Non-dominated sorting on merged pop_all
    # build a temporary pop_all copy to use NS_Sort which modifies DominationSet/Count
    # We can operate directly on pop_all
    fronts = NS_Sort(pop_all)

    # 4) Build new population ensuring exactly nPop individuals
    chosen_rel_indices = []  # indices relative to pop_all
    for front in fronts:
        if len(chosen_rel_indices) + len(front) <= nPop:
            chosen_rel_indices.extend(front)
        else:
            # need to pick some from this front
            needed = nPop - len(chosen_rel_indices)
            chosen = niching_selection(F_all[:,0], W, ideal, chosen_rel_indices, front, nPop)
            # niching_selection returns indices in coordinates of pop_all already
            chosen_rel_indices.extend(chosen)
            break

    # If still short (shouldn't happen often), fill with best remaining (by rank then distance)
    if len(chosen_rel_indices) < nPop:
        remaining_pool = [i for i in range(len(pop_all)) if i not in chosen_rel_indices]
        # sort remaining by rank (pop_all[i]['Rank']) and then by sum of objectives (as tie-breaker)
        remaining_pool_sorted = sorted(remaining_pool, key=lambda ii: (pop_all[ii]['Rank'], np.sum(pop_all[ii]['Cost'])))
        to_add = remaining_pool_sorted[:(nPop - len(chosen_rel_indices))]
        chosen_rel_indices.extend(to_add)

    # form new pop preserving dict objects (or create shallow copies if desired)
    pop = [pop_all[i] for i in chosen_rel_indices]

    # Print progress
    end_time = time.time() - start_time
    print(f"Gen {gen}, FES {FES}/{max_fes}, executed in {end_time:.3f}s")  
    plot3D(pop)
    
# %%plot final front from pop
plot_name = 'NSGA3'
plot3D_adjustable(pop, plot_name)
