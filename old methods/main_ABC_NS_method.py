try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import matplotlib.pyplot as plt

from utils.Multi_objective_functions import CostFunction_MOO
from utils.Graph_functions import Graph, Connectivity_graph
from utils.Domination_functions import check_domination, get_pareto_front, NS_Sort, CD_calc, sort_pop
from utils.Workspace_functions import save_mat


# %% ------------------------- PARAMETERS --------------------------
for Trial in range(10):    
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
        cov = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
        pop.append({'Position': alpop, 'Cost': cov})
    
    Extra_archive = []
            
    # %% ------------------------- MAIN LOOP --------------------------
    for it in range(MaxIt):
    # %% ------------------------- EXPLORATION LOOP --------------------------
        for i in range(nPop):
            k = np.random.randint(nPop)
            phi = a * np.random.uniform(-1, 1, (N, 2)) * (1 - L[i] / MaxIt)**5
            alpop = pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])
            alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
            alpop[0,:] = sink
    
            if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
                if check_domination(alpop_cost, pop[i]['Cost']) == 1:
                    pop[i]['Position'] = alpop
                    pop[i]['Cost'] = alpop_cost
                elif check_domination(alpop_cost, pop[i]['Cost']) == 0:
                    Extra_archive.append({'Position': alpop, 'Cost': alpop_cost})
                else:
                    L[i] += 1
                    continue
                
    # %% ------------------------- SELECTION LOOP --------------------------
        w = np.array([0.5, 0.5])
        E = np.array([np.sum(p['Cost'].flatten() * w) for p in pop])
        if np.sum(E) == 0:
            E_ave = np.ones_like(E) / nPop
        else:
            E_ave = E / np.sum(E)
            
    # %% ------------------------- EXPLOITATION LOOP --------------------------
        for _ in range(nPop):
            i = np.random.choice(nPop, p=E_ave)
            for k in range(N):
                alpop = pop[i]['Position'].copy()
                h = np.random.randint(N)
                phi = a * np.random.uniform(-1, 1, (1, 2)) * (1 - L[i] / MaxIt)**2
                alpop[k] += phi.flatten() * (pop[i]['Position'][k] - pop[i]['Position'][h])
                alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
                alpop[0,:] = sink
    
                if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                    alpop_cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
                    if check_domination(alpop_cost, pop[i]['Cost']) == 1:
                        pop[i]['Position'] = alpop
                        pop[i]['Cost'] = alpop_cost
                        break
                    elif check_domination(alpop_cost, pop[i]['Cost']) == 0:
                        Extra_archive.append({'Position': alpop, 'Cost': alpop_cost})
                        break
                    else:
                        L[i] += 1
                        continue
    
        pop = pop + Extra_archive
        pop, F = NS_Sort(pop)
        pop = CD_calc(pop, F)
        pop, F = sort_pop(pop)
        pop = pop[:nPop]
        
        Extra_archive = []
        
        print(f"Iter={it}, Trial = {Trial}, {len(get_pareto_front(pop))} non-dominated solutions")
    
        
    # %% ------------------------- PLOT --------------------------
    # Tạo mảng data từ Cost của Extra_archive
        # data = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])  # mỗi ind là dict có key 'Cost'
        # data_set = np.array([ind['Cost'].flatten() for ind in pop])
    
        # # Tạo figure
        # fig = plt.figure(1)
        # plt.clf()
        
        # # Vẽ Pareto front
        # plt.plot(data_set[:, 0], data_set[:, 1], 'o', color='g')
        # plt.plot(data[:, 0], data[:, 1], 'o', color='b', label = 'PF')
        # #plt.plot(data2[:, 0], data2[:, 1], 'o', color='r', label = 'NSABC2')
        # #plt.plot(data3[:, 0], data3[:, 1], 'o', color='g', label = 'NSABC3')
        # #plt.text(data[:, 0], data[:, 1], range(0,len(Extra_archive)), fontsize=15, color='red')
        # #plt.legend()
        # plt.xlabel('Non-coverage')
        # plt.ylabel('Energy')
        # None
        # # Cập nhật đồ thị theo từng iteration
        # plt.pause(0.01)
    
    # %% ------------------------- DELETE --------------------------    
    del alpop, alpop_cost, cov, E, E_ave, h, i, k, phi, size
    folder_name = 'data'
    file_name = f'NSABC_{Trial}.mat'
    save_mat(folder_name, file_name,pop,stat,MaxIt)