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
from utils.GA_functions import Crossover, Mutate
from utils.Domination_functions import get_pareto_front, NS_Sort, CD_calc, sort_pop
from utils.Workspace_functions import save_mat


# %% ------------------------- PARAMETERS --------------------------
for Trial in range(10):    
    np.random.seed(Trial)
    size = 100
    MaxIt = 500
    nPop = 100
    N = 60
    
    rc = 10
    rs = np.ones(N, dtype=int) * 10
    #rs = np.random.uniform(10, 15, N)
    stat = np.zeros((2, N))  # tạo mảng 2xN
    stat[0, :] = rs          # dòng 1 là rs
    stat[1, 0] = rc          # phần tử đầu dòng 2 = rc
    
    pCrossover = 0.7                          # Crossover Percentage
    nCrossover = 2 * round(pCrossover * nPop / 2)  # Number of Parents (=> even number of Offsprings)
    pMutation = 0.4                           # Mutation Percentage
    nMutation = round(pMutation * nPop)       # Number of Mutants
    mu = 0.02                                 # Mutation Rate
    sigma = 0.1 * (100)           # Mutation Step Size
    
    # %% ------------------------- INITIATION --------------------------
    Covered_Area = np.zeros((size, size), dtype=int)
    Obstacle_Area = np.ones((size, size), dtype=int)
    
    sink = np.array([size//2, size//2])
    pop = []
    for _ in range(nPop):
        alpop = np.random.uniform(15, 85 , (N, 2))
        alpop[0] = sink
        if Connectivity_graph(Graph(alpop, rc),[]):
            cost = CostFunction_MOO(alpop, stat, Obstacle_Area, Covered_Area.copy())
        else: 
            cost = np.array([[1], [1]])
        pop.append({'Position': alpop, 'Cost': cost})
            
    # %% ------------------------- MAIN LOOP --------------------------
    for it in range(MaxIt):
        
        # ----- Crossover -----
        popc = []
        for k in range(nCrossover // 2):
            i1 = np.random.randint(0, nPop)
            i2 = np.random.randint(0, nPop)
            p1 = pop[i1]
            p2 = pop[i2]
    
            y1, y2 = Crossover(p1['Position'], p2['Position'])
            y1[0, :] = sink  # giữ node sink cố định
            y2[0, :] = sink
            if Connectivity_graph(Graph(y1, rc),[]):
                c1 = {
                    'Position': y1,
                    'Cost': CostFunction_MOO(y1, stat, Obstacle_Area, Covered_Area)
                }
            else:
                c1 = {
                    'Position': y1,
                    'Cost': np.array([[1], [1]])
                }            
            if Connectivity_graph(Graph(y2, rc),[]):
                c2 = {
                    'Position': y2,
                    'Cost': CostFunction_MOO(y2, stat, Obstacle_Area, Covered_Area)
                }
            else:
                c2 = {
                    'Position': y2,
                    'Cost': np.array([[1], [1]])
                }
    
            popc.extend([c1, c2])
    
        # ----- Mutation -----
        popm = []
        for k in range(nMutation):
            i = np.random.randint(0, nPop)
            p = pop[i]
    
            mutated_pos = Mutate(p['Position'], mu, sigma)
            mutated_pos = np.clip(mutated_pos, 0, 100)
    
            if Connectivity_graph(Graph(mutated_pos, rc),[]):
                m = {
                    'Position': mutated_pos,
                    'Cost': CostFunction_MOO(mutated_pos, stat, Obstacle_Area, Covered_Area)
                }
            else:
                m = {
                    'Position': mutated_pos,
                    'Cost': np.array([[1], [1]])
                }
    
            popm.append(m)
    
        # ----- Merge -----
        pop = pop + popc + popm
        pop, F = NS_Sort(pop)
        pop = CD_calc(pop, F)
        pop, F = sort_pop(pop)
        pop = pop[:nPop]
        pop, F = NS_Sort(pop)
        pop = CD_calc(pop, F)
        pop, F = sort_pop(pop)    
        print(f"Iter={it}, Trial = {Trial}, {len(get_pareto_front(pop)):.4f} non-dominated solutions")
       
    # %% ------------------------- PLOT --------------------------
    # Tạo mảng data từ Cost của Extra_archive
        # data = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])  # mỗi ind là dict có key 'Cost'
        # data_set = np.array([ind['Cost'].flatten() for ind in pop])
    
        # # Tạo figure
        # fig = plt.figure(1)
        # plt.clf()
        
        # # Vẽ Pareto front
        # #plt.plot(data_set[:, 0], data_set[:, 1], 'o', color='g')
        # plt.plot(data[:, 0], data[:, 1], 'o', color='r', label = 'NSABC')
        # #plt.plot(data2[:, 0], data2[:, 1], 'o', color='g', label = 'NSGA500it')
        # #plt.plot(data3[:, 0], data3[:, 1], 'o', color='b', label = 'NSGA200it')
        # plt.legend()
        # plt.xlabel('Non-coverage')
        # plt.ylabel('Energy')
        # None
        # # Cập nhật đồ thị theo từng iteration
        # plt.pause(0.01)
    
    # %% ------------------------- DELETE --------------------------    
    del alpop, c1, c2, cost, i, i1, i2, k, m, mutated_pos, p, p1, p2, popc, popm, y1, y2
    folder_name = 'data'
    file_name = f'NSGA_{Trial}.mat'
    save_mat(folder_name, file_name,pop,stat,MaxIt)


