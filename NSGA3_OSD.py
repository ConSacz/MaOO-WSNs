try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass

import numpy as np
import time
from utils.Domination_functions import NS_Sort, osd_selection
from utils.GA_functions import sbx_crossover, polynomial_mutation
from utils.Decompose_functions import das_dennis_generate
from utils.Multi_objective_functions import CostFunction_4F1C_MOO
from utils.Plot_functions import plot3D, plot3D_adjustable
from utils.Workspace_functions import save_mat

# %%---------- Cost Function 3 functions 1 constraint
def CostFunction(pop, stat, RP, Obstacle_Area, Covered_Area):
    return CostFunction_4F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area)

rc_set = [20, 10]
for cases in range(2):
    for Trial in range(5):
        # %%============================================================
        # Main NSGA-III-OSD loop
        # ============================================================
        N_obj = 4
        nPop = 200
        max_fes = 500000
        if N_obj == 3:
            p_ref = 13 # 19 for 200 pop, 13 for 100 pop
        elif N_obj ==4:
            p_ref = 9 # 9 for 200 pop, 7 for 100 pop
        sbx_eta = 30
        mut_eta = 20
        pm = None
        seed = Trial
        xl=0
        xu=100
        
        # Network Parameter
        N = 60
        rc = rc_set[cases]
        stat = np.zeros((2, N), dtype=float)  # tạo mảng 2xN
        stat[1, 0] = rc         # rc
        rs = (8,12)
        sink = np.array([xu//2, xu//2])
        RP = np.zeros((N_obj, 2))
        RP[:,0] = np.ones(N_obj) * 1        # first col are ideal values
        RP[:,1] = np.ones(N_obj) * 1e-12    # second col are nadir values
        
        xmin = np.ones((N,3))*xl
        xmin[:,2] = (np.ones((N,1))*rs[0]).flatten()
        xmax = np.ones((N,3))*xu
        xmax[:,2] = (np.ones((N,1))*rs[1]).flatten()
        
        # %% Initilization
        np.random.seed(seed)
        
        # keep W for comparability (not used in OSD)
        W = das_dennis_generate(N_obj, p_ref)
        
        # environment
        Covered_Area = np.zeros((xu, xu), dtype=int)
        Obstacle_Area = np.ones((xu, xu), dtype=int)
        
        # population init
        FES = 0
        pop = []
        for k in range(nPop):
            alpop = np.zeros((N, 3))
            pos0 = np.random.uniform(xu/2-k*(xu/nPop/2-1e-12), xu/2+k*(xu/nPop/2)+1e-12, (N, 2))
            # if k == 0:
            #     pos0 = np.random.uniform(30, 70, (N, 2))
            # else:
            #     pos0 = np.random.uniform(10, 90, (N, 2)) 
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
        
            # 3) Non-dominated sorting
            fronts = NS_Sort(pop_all)
        
            # 4) Selection using OSD
            chosen_indices = osd_selection(F_all[:,0], fronts, nPop, RP, W)
            pop = [pop_all[i] for i in chosen_indices]
        
            # Print progress
            end_time = time.time() - start_time
            print(f"Gen {gen}, FES {FES}/{max_fes}, executed in {end_time:.3f}s")  
            # plot3D(pop)
            
            folder_name = f'data/4F1C/case{cases+1}'
            file_name = f'NSGA3_OSD_{Trial}.mat'
            save_mat(folder_name, file_name, pop, stat, W, RP)
        # %%plot final front from pop
        # plot_name = 'NSGA3-OSD'
        # plot3D_adjustable(pop, plot_name)
