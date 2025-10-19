"""
nsga3.py
A compact NSGA-III implementation (numpy only).
- SBX crossover + polynomial mutation
- non-dominated sorting
- Das & Dennis reference points
- reference-association selection (niche filling) for the last front

Note: this is an educational implementation; for production use prefer
a tested library (e.g. pymoo).
"""
import numpy as np

# ----------------------------
# utility: non-dominated sorting (fast)
# Returns list of fronts; each front is a list of indices
# ----------------------------
def non_dominated_sort(F):
    # F: (pop_size, n_obj)
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    rank = np.zeros(N, dtype=int)
    fronts = [[]]

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # p dominates q?
            if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                S[p].append(q)
            elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    # last appended front is empty -> drop it
    return fronts[:-1]

# ----------------------------
# Das & Dennis reference point generation (simple recursive)
# returns array of reference directions (K, M)
# ----------------------------
def das_dennis_generate(N_obj, d):
    # N_obj: objectives, d: divisions (positive integer)
    def recursive_gen(N_obj, left, depth):
        if depth == N_obj - 1:
            return [[left]]
        res = []
        for i in range(left + 1):
            tails = recursive_gen(N_obj, left - i, depth + 1)
            for t in tails:
                res.append([i] + t)
        return res

    combos = recursive_gen(N_obj, d, 0)
    W = np.array(combos, dtype=float) / float(d)
    # remove duplicates (shouldn't be necessary)
    # ensure unit directions (norm not zero)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    W = W / norms
    return W

# ----------------------------
# SBX (Simulated Binary Crossover)
# ----------------------------
def sbx_crossover(parent1, parent2, eta=20, pc=1.0, xmin=None, xmax=None):
    D = parent1.size
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() <= pc:
        for i in range(D):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1 = min(parent1[i], parent2[i])
                    x2 = max(parent1[i], parent2[i])
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (x1 - xmin[i]) / (x2 - x1)) if xmin is not None else 1.0
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                    beta = 1.0 + (2.0 * (xmax[i] - x2) / (x2 - x1)) if xmax is not None else 1.0
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                    # clip to bounds if provided
                    if xmin is not None:
                        c1 = np.maximum(c1, xmin[i])
                        c2 = np.maximum(c2, xmin[i])
                    if xmax is not None:
                        c1 = np.minimum(c1, xmax[i])
                        c2 = np.minimum(c2, xmax[i])

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

# ----------------------------
# Polynomial mutation
# ----------------------------
def polynomial_mutation(x, eta=20, pm=None, xmin=None, xmax=None):
    D = x.size
    y = x.copy()
    if pm is None:
        pm = 1.0 / float(D)
    for i in range(D):
        if np.random.rand() < pm:
            xi = x[i]
            xl = xmin[i] if xmin is not None else xi - 1.0
            xu = xmax[i] if xmax is not None else xi + 1.0
            if xl == xu:
                continue
            delta1 = (xi - xl) / (xu - xl)
            delta2 = (xu - xi) / (xu - xl)
            rand = np.random.rand()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** mut_pow
            xi = xi + deltaq * (xu - xl)
            # clip
            xi = np.minimum(np.maximum(xi, xl), xu)
            y[i] = xi
    return y

# ----------------------------
# associate solutions with reference points:
# - normalize objectives by ideal point and objective max (simple)
# - compute perpendicular distance to each reference direction
# returns for each solution: (closest_ref_index, perp_distance)
# ----------------------------
def associate_to_reference(F, ref_dirs, ideal):
    Z = F - ideal
    Znorm = Z / (np.max(np.abs(Z), axis=0, keepdims=True) + 1e-12)
    Znorm = np.squeeze(Znorm)  # loại bỏ chiều dư nếu có

    rd = ref_dirs
    if rd.shape[0] == F.shape[1]:
        rd = rd.T
    rd = rd / (np.linalg.norm(rd, axis=1, keepdims=True) + 1e-12)

    projections = np.dot(Znorm, rd.T)  # (n_solutions, K)
    proj_vecs = projections[:, :, None] * rd[None, :, :]  # (n_solutions, K, M)
    diffs = Znorm[:, None, :] - proj_vecs
    perps = np.linalg.norm(diffs, axis=2)

    ref_idx = np.argmin(perps, axis=1)
    dist = perps[np.arange(perps.shape[0]), ref_idx]
    return ref_idx, dist



# ----------------------------
# niching selection for the last front (NSGA-III style)
# parents: indices of selected so far (from previous fronts)
# last_front: list of candidate indices to fill remaining slots
# returns list of chosen indices (subset of last_front) to fill pop_size
# ----------------------------
def niching_selection(F, ref_dirs, ideal, chosen_indices, last_front, pop_size):
    # chosen_indices: list of already accepted indices (from previous full fronts)
    # last_front: list of candidate indices to choose from
    remaining = pop_size - len(chosen_indices)
    if remaining <= 0:
        return []

    # build arrays for candidates + chosen
    all_idx = np.array(chosen_indices + list(last_front), dtype=int)
    F_all = F[all_idx]
    # associate all solutions
    ref_idx_all, dist_all = associate_to_reference(F_all, ref_dirs, ideal)
    K = ref_dirs.shape[0]
    # compute niche counts based only on chosen (exclude candidates positions)
    niche_count = np.zeros(K, dtype=int)
    # chosen part
    if len(chosen_indices) > 0:
        chosen_rel = np.arange(len(chosen_indices))  # these are first in all_idx
        chosen_refs = ref_idx_all[chosen_rel]
        for r in chosen_refs:
            niche_count[r] += 1

    # candidate relative indices
    cand_rel_idx = np.arange(len(chosen_indices), len(all_idx))
    # for easier handling, create lists of candidates per reference
    ref_to_cands = {k: [] for k in range(K)}
    for i_rel, i_all in enumerate(cand_rel_idx):
        r = int(ref_idx_all[i_all])
        ref_to_cands[r].append(i_all)

    selected = []
    # while need to fill remaining slots:
    while remaining > 0:
        # find refs with zero niche count and have candidates
        zero_refs = [r for r in range(K) if niche_count[r] == 0 and len(ref_to_cands[r]) > 0]
        if len(zero_refs) > 0:
            # for each such ref, select candidate with minimal distance
            for r in zero_refs:
                if remaining == 0:
                    break
                cand_list = ref_to_cands[r]
                # choose candidate with minimal dist
                best_rel = min(cand_list, key=lambda idx_rel: dist_all[idx_rel])
                selected.append(all_idx[best_rel])
                remaining -= 1
                niche_count[r] += 1
                # remove from other ref lists (it's chosen)
                for lst in ref_to_cands.values():
                    if best_rel in lst:
                        lst.remove(best_rel)
        else:
            # no zero niche refs: find ref with minimal niche_count that has candidates
            viable_refs = [r for r in range(K) if len(ref_to_cands[r]) > 0]
            if not viable_refs:
                break  # no candidates left
            min_n = min(niche_count[r] for r in viable_refs)
            refs_min = [r for r in viable_refs if niche_count[r] == min_n]
            # pick one ref (random among ties)
            r = np.random.choice(refs_min)
            cand_list = ref_to_cands[r]
            # select candidate with minimal distance
            best_rel = min(cand_list, key=lambda idx_rel: dist_all[idx_rel])
            selected.append(all_idx[best_rel])
            remaining -= 1
            niche_count[r] += 1
            for lst in ref_to_cands.values():
                if best_rel in lst:
                    lst.remove(best_rel)

    return selected
# ----------------------------
# Example: dtlz2 (3-objectives) for quick test
def problem_eval(x):
    """
    DTLZ2 test function for 3 objectives
    x: (N, D) array
    return: (N, 3) array
    """
    x = np.atleast_2d(x)
    g = np.sum((x[:, 2:] - 0.5)**2, axis=1)
    f1 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.cos(0.5 * np.pi * x[:, 1])
    f2 = (1 + g) * np.cos(0.5 * np.pi * x[:, 0]) * np.sin(0.5 * np.pi * x[:, 1])
    f3 = (1 + g) * np.sin(0.5 * np.pi * x[:, 0])
    F =np.vstack([f1, f2, f3]).T
    return F.squeeze()
# ----------------------------
# NSGA-III main loop
# ----------------------------

D = 12             # decision variables dimension
n_obj = 3         # number of objectives
pop_size=100
max_gen=200
xmin=None
xmax=None
ref_divisions=10  # Das & Dennis divisions (adjust for number of refs)
sbx_eta=30
mut_eta=20
pm=None
seed=1

if seed is not None:
    np.random.seed(seed)

# reference directions
ref_dirs = das_dennis_generate(n_obj, ref_divisions)  # (K, n_obj)

# bounds
if xmin is None:
    xmin = np.zeros(D)
if xmax is None:
    xmax = np.ones(D)

# initialize population
X = np.random.rand(pop_size, D) * (xmax - xmin) + xmin
F = np.array([problem_eval(X[i]) for i in range(pop_size)])  # (pop_size, n_obj)

for gen in range(max_gen):
    # mating: create offspring by selection + SBX + mutation
    # simple tournament selection by rank + crowding not necessary; use random mating pool
    indices = np.random.permutation(pop_size)
    offspring = []
    for i in range(0, pop_size, 2):
        p1 = X[indices[i]]
        if i + 1 < pop_size:
            p2 = X[indices[i + 1]]
        else:
            p2 = X[indices[0]]
        c1, c2 = sbx_crossover(p1, p2, eta=sbx_eta, pc=1.0, xmin=xmin, xmax=xmax)
        c1 = polynomial_mutation(c1, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        c2 = polynomial_mutation(c2, eta=mut_eta, pm=pm, xmin=xmin, xmax=xmax)
        offspring.append(c1)
        offspring.append(c2)
    offspring = np.array(offspring)[:pop_size]
    # evaluate offspring
    F_off = np.array([problem_eval(offspring[i]) for i in range(pop_size)])

    # merge
    X_all = np.vstack([X, offspring])
    F_all = np.vstack([F, F_off])
    N_all = X_all.shape[0]

    # update ideal point (min per objective)
    ideal = np.min(F_all, axis=0)

    # non-dominated sorting
    fronts = non_dominated_sort(F_all)

    # build new population
    new_indices = []
    for front in fronts:
        if len(new_indices) + len(front) <= pop_size:
            new_indices.extend(front)
        else:
            # need to select from this front using niching
            remaining = pop_size - len(new_indices)
            chosen = niching_selection(F_all, ref_dirs, ideal, new_indices, front, pop_size)
            new_indices.extend(chosen)
            break

    # form new population arrays
    X = X_all[new_indices]
    F = F_all[new_indices]

    # optional: progress print
    if (gen + 1) % (max(1, max_gen // 10)) == 0 or gen == 0:
        print(f"Gen {gen+1:4d}: population size {X.shape[0]}  ideal {ideal}")


    # show nondominated front (approx)
    # %%
    
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # cần import để tạo trục 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=10, c='b')
#ax.scatter(W[:, 0], W[:, 1], W[:, 2], s=10, c='b')
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
plt.title("NSGA-III approximate front (ZDT1)")
ax.view_init(elev=25, azim=35)
#plt.gca().invert_yaxis()
plt.show()
