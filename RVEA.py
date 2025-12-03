try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
"""
RVEA (Reference Vector Guided Evolutionary Algorithm)
- pop: list of dicts, each dict has keys:
    'Position' : ndarray shape (n_var,)
    'Cost'     : ndarray shape (n_obj,)
- Uses SBX crossover + polynomial mutation, Das & Dennis reference vectors, PBI selection
"""
import numpy as np
import time
from utils.Decompose_functions import das_dennis_generate
from utils.GA_functions import sbx_crossover, polynomial_mutation
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
from utils.Plot_functions import plot3D, plot3D_adjustable
from utils.Workspace_functions import save_mat

# %%
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
def pop_positions(pop):
    return np.array([ind['Position'] for ind in pop])

def pop_costs(pop):
    return np.array([ind['Cost'] for ind in pop])

# ---------- Cost Function 3 functions 1 constraint
def CostFunction(pop, stat, RP, Obstacle_Area, Covered_Area):
    return CostFunction_3F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area)
rc_set = [20, 10]
for cases in range(2):
    for Trial in range(2):
        # %% ---------- RVEA main (pop-only) ----------
        # Algorithm Parameter
        seed=Trial
        n_obj = 3
        nPop = 200
        max_fes = 500
        p_ref=19
        crossover_prob=0.9
        eta_c=20
        eta_m=20
        pm=0.1
        theta0=5.0
        alpha=2.0
        xl=0
        xu=100
        
        # Network Parameter
        N = 60
        rc = rc_set[cases]
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
        
        # %% initialize pop
        # environment
        Covered_Area = np.zeros((xu, xu), dtype=int)
        Obstacle_Area = np.ones((xu, xu), dtype=int)
        
        # population init
        FES = 0
        pop = []
        for k in range(nPop):
            alpop = np.zeros((N, 3))
            pos0 = np.random.uniform(xu/2-k*(xu/nPop/2-1e-12), xu/2+k*(xu/nPop/2)+1e-12, (N, 2)) 
            pos0[0] = sink
            rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
            alpop[:,:2] = pos0
            alpop[:,2] = rs0[:, 0]
            alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
            RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
            RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
            pop.append({'Position': alpop, 'Cost': alpop_cost})
        FES += nPop
        
        # %% reference vectors
        W = das_dennis_generate(n_obj, p_ref)  # K x n_obj
        if W.shape[0] == 0:
            raise ValueError("No reference vectors generated. Increase p_ref or check M.")
        W = W / np.linalg.norm(W, axis=1, keepdims=True)
        
        # 3) ideal point
        F_curr = pop_costs(pop)
        ideal = get_ideal(F_curr)
        
        # %% main loop
        gen = 0
        while FES < max_fes:
            start_time = time.time()
            gen += 1
            # 4) mating & variation -> produce offspring positions
            # random mating order
            mating_order = np.random.permutation(nPop)
            offspring_positions = []
            for i in range(0, nPop, 2):
                p1 = pop[mating_order[i % nPop]]['Position']
                p2 = pop[mating_order[(i + 1) % nPop]]['Position']
                if np.random.rand() < crossover_prob:
                    c1, c2 = sbx_crossover(p1, p2, eta=eta_c, pc=1.0, xmin=xmin, xmax=xmax)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                c1 = polynomial_mutation(c1, eta=eta_m, pm=pm, xmin=xmin, xmax=xmax)
                c2 = polynomial_mutation(c2, eta=eta_m, pm=pm, xmin=xmin, xmax=xmax)
                offspring_positions.append(c1); offspring_positions.append(c2)
            U = np.array(offspring_positions)[:nPop]
            U[:,:,2] = np.clip(U[:,:,2], rs[0], rs[1])
            # 5) evaluate offspring
            FU = np.array([CostFunction_3F1C_MOO(U[i], stat, RP, Obstacle_Area, Covered_Area.copy()) for i in range(nPop)])
            FES += nPop
            
            # 6) combine pop and offspring into transient arrays for selection
            X_all = np.vstack([pop_positions(pop), offspring_positions])   # (2N x n_var)
            F_all = np.vstack([pop_costs(pop), FU])                    # (2N x n_obj)
        
            # 7) update ideal
            ideal = np.min(F_all, axis=0)
            nadir = np.max(F_all, axis=0)
            RP[:,0] = ideal.T.flatten()
            RP[:,1] = nadir.T.flatten()
            
            # 8) normalization
            
            F_norm = normalize_objectives(F_all, ideal, nadir)[:,0]
        
            # 9) associate each individual to a reference vector by minimal angle (max cosine)
            # prepare dir unit vectors (handle zero rows)
            dir_norms = np.linalg.norm(F_norm, axis=0, keepdims=True)
            # avoid division by zero
            dir_unit = np.where(dir_norms == 0, 0.0, F_norm / dir_norms)
            cos = dir_unit.dot(W.T)  # (2N) x K
            assoc = np.argmax(cos, axis=1)  # index of associated reference vector for each individual
        
            # 10) PBI-based selection per reference vector
            t = FES
            T = max_fes
            theta = theta0 * (t / T) ** alpha
        
            chosen_indices = []
            K = W.shape[0]
            for k in range(K):
                members_idx = np.where(assoc == k)[0]
                if members_idx.size == 0:
                    continue
                F_members = F_norm[members_idx]
                d1, d2 = pbi_values(F_members, W[k])
                fitness = d1 + theta * d2
                best_local = members_idx[np.argmin(fitness)]
                chosen_indices.append(best_local)
        
            # 11) fill up or trim to nPop
            if len(chosen_indices) < nPop:
                remaining = np.setdiff1d(np.arange(F_all.shape[0]), chosen_indices)
                # Use simple global score: sum of normalized objectives (lower better)
                global_score = np.sum(F_norm, axis=1)
                fill_order = remaining[np.argsort(global_score[remaining])]
                need = nPop - len(chosen_indices)
                chosen_indices.extend(list(fill_order[:need]))
            elif len(chosen_indices) > nPop:
                # trim (rare) by global score
                chosen_indices = chosen_indices[:nPop]
        
            # 12) form next pop (as list of dicts)
            new_pop = []
            for idx in chosen_indices:
                new_pop.append({
                    'Position': X_all[idx].copy(),
                    'Cost'    : F_all[idx].copy()
                })
            pop = new_pop
            end_time = time.time() - start_time
            
            # optional logging
            print(f"Gen {gen}, FES {FES}/{max_fes}, executed in {end_time:.3f}s") 
            # plot3D(pop)
            # %% ------------------------- SAVE MATRIX -------------------------- 
            folder_name = f'data/case{cases+1}'
            file_name = f'RVEA_{Trial}.mat'
            save_mat(folder_name, file_name, pop, stat, W, max_fes)
            
        # %%plot final front from pop
        # plot_name = 'RVEA'
        # plot3D_adjustable(pop, plot_name)
