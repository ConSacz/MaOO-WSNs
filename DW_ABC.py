try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat
# from scipy.spatial.distance import cdist
# import os
from utils.Multi_objective_functions import CostFunction_MOO
from utils.Graph_functions import Graph, Connectivity_graph
from utils.Domination_functions import get_pareto_front, weighted_selection
from utils.Decompose_functions import weight_assign
from utils.Workspace_functions import save_mat


# %% ------------------------- PARAMETERS --------------------------
for Trial in range(1):    
    np.random.seed(Trial)
    size = 100
    MaxIt = 500
    nPop = 50
    N = 60
    
    rc = 10
    rs = np.ones(N, dtype=int) * 10
    #rs = np.random.uniform(10, 15, N)
    stat = np.zeros((2, N))  # tạo mảng 2xN
    stat[0, :] = rs          # dòng 1 là rs
    stat[1, 0] = rc          # phần tử đầu dòng 2 = rc
    RP = np.zeros((2, 2))   # dòng 1 là nadỉ value
    RP[:,0] = [1, 1]        # dòng 2 là ideal value
    
    # %% ------------------------- INITIATION --------------------------
    Covered_Area = np.zeros((size, size), dtype=int)
    #Obstacle_Area = gen_target_area(1000, size)
    Obstacle_Area = np.ones((size, size), dtype=int)
    
    sink = np.array([size//2, size//2])
    a = 1
    L = np.zeros(nPop, dtype=int)
    
    pop = []
    for _ in range(nPop):
        alpop = np.random.uniform(sink[0]-rc/2, sink[1]+rc/2, (N, 2)) 
        alpop[0] = sink
        alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
        pop.append({'Position': alpop, 'Cost': alpop_cost})
        RP[:,0] = np.minimum(RP[:,0], alpop_cost[:,0])
        RP[:,1] = np.maximum(RP[:,1], alpop_cost[:,0])
            
    # %% ------------------------- MAIN LOOP --------------------------
    for it in range(MaxIt):
        pop, w = weight_assign(pop,RP)
    # %% ------------------------- EXPLORATION LOOP --------------------------
        #print("Exploration starts")
        for i in range(nPop):
            k = np.random.randint(nPop)
            phi = a * np.random.uniform(-1, 1, (N, 2)) * (1 - L[i] / MaxIt)**5
            alpop = pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])
            alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
            alpop[0,:] = sink
    
            if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
                RP[:,0] = np.minimum(RP[:,0], alpop_cost[:,0])
                RP[:,1] = np.maximum(RP[:,1], alpop_cost[:,0])
                if weighted_selection(alpop_cost, pop[i]['Cost'],w[i,:],RP) == 1:
                    pop[i]['Position'] = alpop
                    pop[i]['Cost'] = alpop_cost 
                else:
                    L[i] += 1
                    #continue
           
    # %% ------------------------- EXPLOITATION LOOP --------------------------
        #print("Exploitation starts")    
        for i in range(nPop):
            arr = np.arange(1, N) 
            np.random.shuffle(arr) 
            for j in range(1,N-1):
                k = arr[j]
                alpop = pop[i]['Position'].copy()
                h = np.random.randint(N)
                phi = a * np.random.uniform(-1, 1, (1, 2)) * (1 - L[i] / MaxIt)**2
                alpop[k] += phi.flatten() * (pop[i]['Position'][k] - pop[i]['Position'][h])
                alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
                alpop[0,:] = sink
    
                if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                    alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
                    RP[:,0] = np.minimum(RP[:,0], alpop_cost[:,0])
                    RP[:,1] = np.maximum(RP[:,1], alpop_cost[:,0])
                    if weighted_selection(alpop_cost, pop[i]['Cost'], w[i,:],RP) == 1:
                        pop[i]['Position'] = alpop
                        pop[i]['Cost'] = alpop_cost
                        break
            #print(f"Exploitation changing of pop {i}, node {k} ")
            
        print(f"Iter={it}, Trial = {Trial}, {len(get_pareto_front(pop))} non-dominated solutions")
    
        
    # %% ------------------------- PLOT --------------------------
    # Tạo mảng data từ Cost của Extra_archive
        data = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])  # mỗi ind là dict có key 'Cost'
        data_set = np.array([ind['Cost'].flatten() for ind in pop])
    
        # Tạo figure
        fig = plt.figure(1)
        plt.clf()
        
        # Vẽ Pareto front
        plt.plot(data_set[:, 0], data_set[:, 1], 'o', color='g')
        plt.plot(data[:, 0], data[:, 1], 'o', color='b', label = 'PF')
        #plt.plot(data2[:, 0], data2[:, 1], 'o', color='r', label = 'NSWABC')
        #plt.plot(data3[:, 0], data3[:, 1], 'o', color='g', label = 'NSWABC')
        for i in range(len(data_set)):
            x, y = data_set[i]
            plt.text(x, y, f"{w[i,0]:.1f} {w[i,1]:.1f}", fontsize=8, ha='right', va='bottom', color='blue')
        plt.legend()
        # plt.xlim(RP[0,0], RP[0,1])
        # plt.ylim(RP[1,0], RP[1,1])
        plt.xlabel('Non-coverage')
        plt.ylabel('Energy')
        None
        # Cập nhật đồ thị theo từng iteration
        plt.pause(0.001)
    
    # %% ------------------------- DELETE --------------------------    
    del alpop, alpop_cost, h, i, k, phi, size
    folder_name = 'data'
    file_name = f'DWABC_{Trial}.mat'
    save_mat(folder_name, file_name,pop,stat,MaxIt)
    
    
    