import numpy as np
from .Normalize_functions import global_normalized
from .Decompose_functions import associate_to_reference

# %% WEIGHTED SELECTION
def weighted_selection(f1,f2,w,RP):
    f1_nmlized = global_normalized(f1, RP)
    f2_nmlized = global_normalized(f2, RP)
    ff1 = np.sum(f1_nmlized*(w))
    ff2 = np.sum(f2_nmlized*(w))
    if ff1 < ff2:
        return 1
    else:
        return 0
    
# %% CHECK DOMINATION
def check_domination(f1, f2):
    """
    Check Pareto domination relationship between two solutions f1 and f2.
    Returns:
        1  if f1 dominate f2
       -1  if f2 dominate f1
        0  if non-dominated
        2  if f1 == f2
    """
    f1 = np.asarray(f1)
    f2 = np.asarray(f2)
    if np.all(f1 <= f2) and np.any(f1 < f2):
        return 1
    elif np.all(f2 <= f1) and np.any(f2 < f1):
        return -1
    elif np.all(f1 == f2):
        return 2
    else:
        return 0
    
# %% GET PARETO FRONT
# function return pop
def get_pareto_front(non_dom_pop):
    N = len(non_dom_pop)
    is_dominated = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if check_domination(non_dom_pop[j]['Cost'], non_dom_pop[i]['Cost']) == 1:
                is_dominated[i] = True
                break

    # Lấy tất cả các cá thể không bị chi phối
    pareto_front_all = [ind for i, ind in enumerate(non_dom_pop) if not is_dominated[i]]

    # --- Loại bỏ cá thể trùng Cost ---
    costs = np.array([ind['Cost'] for ind in pareto_front_all])
    _, unique_indices = np.unique(costs, axis=0, return_index=True)

    # Duy trì thứ tự xuất hiện (giống 'stable' trong MATLAB)
    unique_indices = sorted(unique_indices)
    pareto_front = [pareto_front_all[i] for i in unique_indices]

    return pareto_front
# function return mask of pop
def nondominated_front(F):
    F = np.asarray(F)
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if j == i or not mask[j]:
                continue
            if check_domination(F[j], F[i]) == 1:
                mask[i] = False
                break
        if mask[i]:
            for j in range(n):
                if j == i or not mask[j]:
                    continue
                if check_domination(F[i], F[j]) == 1:
                    mask[j] = False
    return mask

# %% NON DOMINATED SORTING
def NS_Sort(pop):
    nPop = len(pop)

    for i in range(nPop):
        pop[i]['DominationSet'] = []
        pop[i]['DominatedCount'] = 0

    F = [[]]

    for i in range(nPop):
        for j in range(i + 1, nPop):
            p = pop[i]
            q = pop[j]

            if check_domination(p['Cost'], q['Cost']) == 1:
                p['DominationSet'].append(j)
                q['DominatedCount'] += 1
            elif check_domination(q['Cost'], p['Cost']) == 1:
                q['DominationSet'].append(i)
                p['DominatedCount'] += 1

            pop[i] = p
            pop[j] = q

        if pop[i]['DominatedCount'] == 0:
            F[0].append(i)
            pop[i]['Rank'] = 1

    k = 0
    while True:
        Q = []
        for i in F[k]:
            p = pop[i]
            for j in p['DominationSet']:
                q = pop[j]
                q['DominatedCount'] -= 1
                if q['DominatedCount'] == 0:
                    Q.append(j)
                    q['Rank'] = k + 2
                pop[j] = q
        if not Q:
            break
        F.append(Q)
        k += 1
    return F

# %% CROWDING DISTANCE CALCULATING
def CD_calc(pop, F):
    nF = len(F)

    for k in range(nF):
        front = F[k]
        n = len(front)
        if n == 0:
            continue

        # Lấy tất cả Cost của các cá thể trong front
        Costs = np.array([pop[i]['Cost'].flatten() for i in front]).T  # shape: (nObj, n)

        nObj = Costs.shape[0]
        d = np.zeros((n, nObj))

        for j in range(nObj):
            cj = Costs[j]  # all costs of j Obj
            so = np.argsort(cj)
            d[so[0], j] = np.inf
            d[so[-1], j] = np.inf

            denom = abs(cj[so[-1]] - cj[so[0]])
            if denom == 0:
                denom = 1e-12  # tránh chia cho 0

            for i in range(1, n - 1):
                d[so[i], j] = abs(cj[so[i + 1]] - cj[so[i - 1]]) / denom

        # Gán tổng khoảng cách cho từng cá thể trong front
        for i in range(n):
            pop[front[i]]['CrowdingDistance'] = np.sum(d[i])
    return pop

# %% Sort based on CrowdingDistance (giảm dần)
def sort_pop(pop):
    pop.sort(key=lambda p: p['CrowdingDistance'], reverse=True)

    # Sort based on Rank (tăng dần)
    pop.sort(key=lambda p: p['Rank'])

    # Update Fronts
    ranks = [p['Rank'] for p in pop]
    max_rank = max(ranks)
    F = []

    for r in range(1, max_rank + 1):
        front = [i for i, rank in enumerate(ranks) if rank == r]
        F.append(front)

    return pop, F

# %% IMDEA prune_archive
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
    
# %% OSD
def osd_selection(F, fronts, nPop, RP, W):
    chosen = []

    for front in fronts:
        if len(chosen) + len(front) <= nPop:
            chosen.extend(front)
        else:
            needed = nPop - len(chosen)
            last = np.array(front)
            lastF = F[last]

            # Decomposition assignment
            ref_idx, dpp, _ = associate_to_reference(lastF, W, RP)

            selected = []
            K = W.shape[0]

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

# %% Association & niching
def niching_selection(F, W, RP, chosen_indices, last_front, nPop):
    """
    chosen_indices, last_front: lists of indices relative to F (i.e. indices in pop_all / F_all)
    returns: list of chosen indices (indices in same coordinate system as chosen_indices / last_front)
    """
    remaining = nPop - len(chosen_indices)
    if remaining <= 0:
        return []

    all_idx = np.array(list(chosen_indices) + list(last_front), dtype=int)
    F_all = F[all_idx]
    ref_idx_all, dist_all, _ = associate_to_reference(F_all, W, RP)
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



