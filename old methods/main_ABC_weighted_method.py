globals().clear()
# %%
import numpy as np
import matplotlib.pyplot as plt
# from scipy.io import savemat
# from scipy.spatial.distance import cdist
# import os
from utils.Multi_objective_functions import CostFunction_weighted
from utils.Graph_functions import Graph, Connectivity_graph
from utils.Single_objective_functions import Cov_Func

# %% ------------------------- PARAMETERS --------------------------
np.random.seed(0)
size = 100
MaxIt = 200
nPop = 25
N = 60

rc = 10
rs = np.ones(N, dtype=int) * 10
#rs = np.random.uniform(10, 15, N)
stat = np.zeros((2, N))  # tạo mảng 2xN
stat[0, :] = rs          # dòng 1 là rs
stat[1, 0] = rc          # phần tử đầu dòng 2 = rc
w = np.array([1, 0])

# %% ------------------------- INITIATION --------------------------
Covered_Area = np.zeros((size, size), dtype=int)
#Obstacle_Area = gen_target_area(1000, size)
Obstacle_Area = np.ones((size, size), dtype=int)


sink = np.array([size//2, size//2])
a = 1
L = np.zeros(nPop, dtype=int)

pop = []
BestSol = {'Position': None, 'Cost': 1}
BestCost = []

for _ in range(nPop):
    alpop = np.random.uniform(sink[0]-rc/2, sink[1]+rc/2, (N, 2))
    alpop[0] = sink
    cov= CostFunction_weighted(alpop, stat, w, Obstacle_Area, Covered_Area.copy())
    pop.append({'Position': alpop, 'Cost': cov})
    if cov < BestSol['Cost']:
        BestSol = {'Position': alpop.copy(), 'Cost': cov}
        
# %% ------------------------- MAIN LOOP --------------------------
for it in range(MaxIt):
    for i in range(nPop):
        k = np.random.randint(nPop)
        phi = a * np.random.uniform(-1, 1, (N, 2)) * (1 - L[i] / MaxIt)**5
        alpop = pop[i]['Position'] + phi * (pop[i]['Position'] - pop[k]['Position'])
        alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)
        alpop[0,:] = sink

        if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
            new_cost= CostFunction_weighted(alpop, stat, w, Obstacle_Area, Covered_Area.copy())
            if new_cost <= pop[i]['Cost']:
                pop[i]['Position'] = alpop
                pop[i]['Cost'] = new_cost
            else:
                L[i] += 1
            break

    E = np.array([p['Cost'] for p in pop])
    if np.sum(E) == 0:
        E_ave = np.ones_like(E) / nPop
    else:
        E_ave = E / np.sum(E)

    for _ in range(nPop):
        i = np.random.choice(nPop, p=E_ave)
        for k in range(N):
            alpop = pop[i]['Position'].copy()
            h = np.random.randint(N)
            phi = a * np.random.uniform(-1, 1, (1, 2)) * (1 - L[i] / MaxIt)**2
            alpop[k] += phi.flatten() * (pop[i]['Position'][k] - pop[i]['Position'][h])
            alpop[:, :2] = np.clip(alpop[:, :2], 0, size - 1)

            if Connectivity_graph(Graph(alpop[:, :2], rc),[]):
                new_cost= CostFunction_weighted(alpop, stat, w, Obstacle_Area, Covered_Area.copy())
                if new_cost <= pop[i]['Cost']:
                    pop[i]['Position'] = alpop
                    pop[i]['Cost'] = new_cost

    for i in range(nPop):
        if pop[i]['Cost'] < BestSol['Cost']:
            BestSol = {'Position': pop[i]['Position'].copy(), 'Cost': pop[i]['Cost']}

    BestCost.append(BestSol['Cost'])
    print(f"Iter={it}, Best Cost = {BestSol['Cost']:.4f}")
    if BestSol['Cost'] == 1:
        break
    
# %% ------------------------- PLOT --------------------------
plt.clf()
plt.grid(True)
G = Graph(BestSol['Position'],rc)
cov, Covered_Area= Cov_Func(BestSol['Position'], rs, Obstacle_Area, Covered_Area)
# Hiển thị map
#obs_row, obs_col = np.where(Obstacle_Area == 1)
#plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')  # MATLAB plot(row,col) → Python plot(x=col,y=row)
#obs_row, obs_col = np.where(Obstacle_Area == 0)
#plt.plot(obs_col, obs_row, '.', markersize=8, color='black')
#discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
#plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=5, color='red')
discovered_row, discovered_col = np.where(Covered_Area == 1)
plt.plot(discovered_col, discovered_row, '.', markersize=1, color='green')

# vẽ cảm biến
theta = np.linspace(0, 2*np.pi, 500)
for i in range(N):
    plt.plot(BestSol['Position'][i, 1], BestSol['Position'][i, 0], 'o', markersize=3, color='blue')
    plt.text(BestSol['Position'][i, 1], BestSol['Position'][i, 0], str(i+1), fontsize=15, color='red')  # chỉ số i+1 để giống MATLAB

    x = BestSol['Position'][i, 1] + rs[i] * np.cos(theta)
    y = BestSol['Position'][i, 0] + rs[i] * np.sin(theta)
    plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.7, edgecolor='k')

# vẽ kết nối
for edge in G.edges():
    i, j = edge
    x_coords = [BestSol['Position'][i, 0], BestSol['Position'][j, 0]]
    y_coords = [BestSol['Position'][i, 1], BestSol['Position'][j, 1]]
    
    plt.plot(y_coords, x_coords, color='blue', linewidth=1)
    
# Giới hạn trục
plt.xlim([0, Obstacle_Area.shape[1]])
plt.ylim([0, Obstacle_Area.shape[0]])

# Tiêu đề
plt.title(f"{(1-cov)*100:.2f}%  at time step: {it}")

#plt.gca().invert_yaxis()  # để trục y giống MATLAB (gốc ở trên)
plt.gca()
plt.draw()
plt.pause(0.001)  # giống drawnow trong MATLAB


# folder = f"data/hetero_target/{N} nodes/case4"
# os.makedirs(folder, exist_ok=True)
# savemat(os.path.join(folder, f"hetero_{trial}.mat"), {
#     'BestPosition': BestSol['Position'],
#     'BestCost': BestSol['Cost'],
#     'rs': rs,
#     'theta0': theta0
# })
