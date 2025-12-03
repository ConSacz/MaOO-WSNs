try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import time
from utils.Multi_objective_functions import CostFunction_3F1C_MOO
from utils.Domination_functions import weighted_selection
from utils.Decompose_functions import weight_assign, das_dennis_generate
from utils.Plot_functions import plot3D, plot3D_adjustable, plot_MaOO
from utils.Workspace_functions import save_mat

# ---------- Cost Function 3 functions 1 constraint
def CostFunction(pop, stat, RP, Obstacle_Area, Covered_Area):
    return CostFunction_3F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area)

# %% ------------------------- PARAMETERS --------------------------
rc_set = [20, 10]
for cases in range(2):
    for Trial in range(5): 
        np.random.seed(Trial)
        N_obj = 3
        p_ref = 13 # 19 for 200 pop, 13 for 100 pop
        max_fes = 500000
        nPop = 100
        xmin = 0
        xmax = 100
        a = 1
        
        # Network Parameter
        N = 60
        rc = rc_set[cases]
        stat = np.zeros((2, N))  # tạo mảng 2xN
        stat[1, 0] = rc         # rc
        rs = (8,12)
        sink = np.array([xmax//2, xmax//2])
        RP = np.zeros((3, 2))   
        RP[:,0] = [1, 1, 1]          # first col are ideal values
        RP[:,1] = [1e-12, 1e-12, 1e-12]    # second col are nadir values
        
        
        # %% ------------------------- INITIATION --------------------------
        Covered_Area = np.zeros((xmax, xmax), dtype=int)
        #Obstacle_Area = gen_target_area(1000, xmax)
        Obstacle_Area = np.ones((xmax, xmax), dtype=int)
    
        FES = 0
        pop = []
        for k in range(nPop):
            alpop = np.zeros((N, 3))
            # pos0 = np.random.uniform(xmax/2-k*(0.6*xmax/nPop/2-1e-12), xmax/2+k*(0.6*xmax/nPop/2)+1e-12, (N, 2))
            pos0 = np.random.uniform(xmax/2-15, xmax/2+15, (N, 2))
            pos0[0] = sink
            rs0 = np.random.uniform(rs[0], rs[1], (N, 1))
            alpop[:,:2] = pos0
            alpop[:,2] = rs0[:, 0]
            alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
            RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
            RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
            pop.append({'Position': alpop, 'Cost': alpop_cost})
        del k
        FES += nPop
        
        # %% ------------------------- MAIN LOOP --------------------------
        start_loop = time.time()
        gen = 0
        while FES < max_fes:
            start_time = time.time()
            gen+=1
            F = np.array([p['Cost'] for p in pop])[:,0]
            W = das_dennis_generate(N_obj, p_ref)
            W = weight_assign(F, W, RP)
        # %% ------------------------- EXPLORATION LOOP --------------------------
            #print("Exploration starts")
            for i in range(nPop):
                k = np.random.randint(nPop)
                phi = a * np.random.uniform(-1, 1, (N, 3)) * (1 - FES/ max_fes)**5
                alpop = pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])
                alpop[:,:2] = np.clip(alpop[:,:2], 0, xmax - 1)
                alpop[:, 2] = np.clip(alpop[:, 2], rs[0],rs[1])
                alpop[0,:2] = sink
                alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
                RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
                RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
                if weighted_selection(alpop_cost, pop[i]['Cost'],W[i,:],RP) == 1:
                    pop[i]['Position'] = alpop
                    pop[i]['Cost'] = alpop_cost
            FES += nPop
        # %% ------------------------- EXPLOITATION LOOP --------------------------
            #print("Exploitation starts")    
            for i in range(nPop):
                arr = np.arange(1, N) 
                np.random.shuffle(arr) 
                for j in range(N-1):
                    k = arr[j]
                    alpop = pop[i]['Position'].copy()
                    h = np.random.randint(N)
                    phi = a * np.random.uniform(-1, 1, (1, 3)) * (1 - FES/max_fes)**2
                    alpop[k] += phi.flatten() * (pop[i]['Position'][k] - pop[i]['Position'][h])
                    alpop[:,:2] = np.clip(alpop[:,:2], 0, xmax - 1)
                    alpop[:, 2] = np.clip(alpop[:, 2], rs[0],rs[1])
                    alpop[0,:2] = sink
                    alpop_cost = CostFunction(alpop, stat, RP, Obstacle_Area, Covered_Area.copy())
                    FES += 1
                    RP[:,0] = np.minimum(RP[:,0], alpop_cost[0])
                    RP[:,1] = np.maximum(RP[:,1], alpop_cost[0])
                    if weighted_selection(alpop_cost, pop[i]['Cost'], W[i,:],RP) == 1:
                        pop[i]['Position'] = alpop
                        pop[i]['Cost'] = alpop_cost
                        break
                #print(f"Exploitation changing of pop {i}, node {k} ")
            del i, arr, alpop, alpop_cost, k, phi, h
            
            end_time = time.time() - start_time
            print(f"Gen {gen}, FES {FES}/{max_fes}, executed in {end_time:.3}s") 
            
        # %% ------------------------- PLOT --------------------------
            # plot3D(pop)
            #plot_MaOO(F, RP)
        
        # %% ------------------------- TOTAL TIME --------------------------
        total_time = (time.time() - start_loop)/60
        print(f'{cases}.{Trial}, total time: = {total_time:.0}m')
        # end loop    
    
    # plot3D_adjustable(pop, name = 'DWABC')
    
    # %% ------------------------- SAVE MATRIX --------------------------    
        folder_name = f'data/case{cases+1}'
        file_name = f'DWABC_{Trial}.mat'
        save_mat(folder_name, file_name, pop, stat, W, max_fes)
    
# # %%
# from utils.Normalize_functions import global_normalized
# for i in range(nPop):
    # pop[i]['Cost'] = global_normalized(pop[i]['Cost'], RP)
    # pop[i]['Cost'] = CostFunction(pop[i]['Position'], stat, RP, Obstacle_Area, Covered_Area.copy())
    