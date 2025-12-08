try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import matplotlib.pyplot as plt
from utils.Workspace_functions import load_mat
from utils.Normalize_functions import global_normalized

# %% DATA IMPORT
Trials = 5
algorithm_configs = {
    'DWABC': 'DWABC',
    'IMDEA': 'IMDEA',
    'MOEAD': 'MOEAD_DU',
    'NSGA3': 'NSGA3',
    'NSGA3OSD': 'NSGA3_OSD',
    'RVEA': 'RVEA'
}

# CASE 1
folder_name = 'data\case1'
results_case1 = {}
RP1 = np.zeros((3, 2))   
RP1[:,0] = [1, 1, 1]          # first col are ideal values
RP1[:,1] = [1e-12, 1e-12, 1e-12]    # second col are nadir values

for result_name, file_prefix in algorithm_configs.items():
    current_algorithm = []
    for i in range(Trials):
        file_name = f'{file_prefix}_{i}.mat'
        data = load_mat(folder_name, file_name)
        pop = data['pop']
        PF = np.array([ind['Cost'].flatten() for ind in pop])
        RP1[:,0] = np.minimum(RP1[:,0], np.min(PF, axis=0))
        RP1[:,1] = np.maximum(RP1[:,1], np.max(PF, axis=0))
        current_algorithm.append(PF)
    results_case1[result_name] = current_algorithm

# CASE 2
folder_name = 'data\case2'
results_case2 = {}
RP2 = np.zeros((3, 2))   
RP2[:,0] = [1, 1, 1]          # first col are ideal values
RP2[:,1] = [1e-12, 1e-12, 1e-12]    # second col are nadir values

for result_name, file_prefix in algorithm_configs.items():
    current_algorithm = []
    for i in range(Trials):
        file_name = f'{file_prefix}_{i}.mat'
        data = load_mat(folder_name, file_name)
        pop = data['pop']
        PF = np.array([ind['Cost'].flatten() for ind in pop])
        RP2[:,0] = np.minimum(RP2[:,0], np.min(PF, axis=0))
        RP2[:,1] = np.maximum(RP2[:,1], np.max(PF, axis=0))
        current_algorithm.append(PF)
    results_case2[result_name] = current_algorithm
del data, pop, PF, i, file_name, folder_name, algorithm_configs, Trials, file_prefix, result_name, current_algorithm
# %% PLOT PARETO CASE 1
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
TRIAL_INDEX_TO_PLOT = 0 

# Colors and Markers list
colors = ['b', 'r', 'c', 'm', 'y', 'k'] 
markers = ['o', 's', '^', 'D', 'p', 'h']

for idx, (alg_name, alg_PFS_list) in enumerate(results_case1.items()):
    
    # take Colors and Markers
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]
        
    # Lấy mảng PF của lần chạy được chọn (ví dụ: Trial 0)
    selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]
    X = selected_PF[:, 0]
    Y = selected_PF[:, 1]
    Z = selected_PF[:, 2]
    
    # plot scatter
    ax.scatter(X, Y, Z, 
               color=color, 
               marker=marker, 
               alpha=0.8,
               s=30, # Kích thước điểm
               label=f'{alg_name} (Trial {TRIAL_INDEX_TO_PLOT})') 

ax.set_xlabel('$f_1$ Coverage')
ax.set_ylabel('$f_2$ Comm Energy')
ax.set_zlabel('$f_3$ Sens Energy')
ax.set_title(f'Comparison of Pareto Fronts (PFs) on Case 1 (Trial {TRIAL_INDEX_TO_PLOT})')
ax.legend(loc='best')
ax.view_init(elev=20, azim=190)
plt.show()

del alg_PFS_list, alg_name, ax, color, colors, fig, idx, marker, markers, selected_PF, X, Y, Z, TRIAL_INDEX_TO_PLOT

# %% PLOT PARETO CASE 2
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
TRIAL_INDEX_TO_PLOT = 2 

# Colors and Markers list
colors = ['b', 'r', 'c', 'm', 'y', 'k'] 
markers = ['o', 's', '^', 'D', 'p', 'h']

for idx, (alg_name, alg_PFS_list) in enumerate(results_case2.items()):
    
    # take Colors and Markers
    color = colors[idx % len(colors)]
    marker = markers[idx % len(markers)]
        
    # Lấy mảng PF của lần chạy được chọn (ví dụ: Trial 0)
    selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]
    X = selected_PF[:, 0]
    Y = selected_PF[:, 1]
    Z = selected_PF[:, 2]
    
    # plot scatter
    ax.scatter(X, Y, Z, 
               color=color, 
               marker=marker, 
               alpha=0.8,
               s=30, # Kích thước điểm
               label=f'{alg_name} (Trial {TRIAL_INDEX_TO_PLOT})') 

ax.set_xlabel('$f_1$ Coverage')
ax.set_ylabel('$f_2$ Comm Energy')
ax.set_zlabel('$f_3$ Sens Energy')
ax.set_title(f'Comparison of Pareto Fronts (PFs) on Case 2 (Trial {TRIAL_INDEX_TO_PLOT})')
ax.legend(loc='best')
ax.view_init(elev=20, azim=190)
plt.show()

del alg_PFS_list, alg_name, ax, color, colors, fig, idx, marker, markers, selected_PF, X, Y, Z, TRIAL_INDEX_TO_PLOT
# %% PLOT LINE CASE 1
for i in range(6):
    TRIAL_INDEX_TO_PLOT = 0
    ALG_list = [i] 
    objective_labels = ['$f_1$', '$f_2$', '$f_3$'] 
    
    colors = ['b', 'r', 'c', 'm', 'y', 'k'] 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    legend_handles = []
    legend_labels = []
    
    for idx, (alg_name, alg_PFS_list) in enumerate(results_case1.items()):
        if idx not in ALG_list:
            continue
        color = colors[idx % len(colors)]
        selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]
        PF_to_plot = selected_PF[:, :len(objective_labels)]
        PF_to_plot = global_normalized(PF_to_plot, RP1)
        x_coords = np.arange(len(objective_labels))
        for i in range(PF_to_plot.shape[0]):
            
            individual_costs = PF_to_plot[i, :]
            line, = ax.plot(x_coords, individual_costs, 
                            color=color, 
                            linestyle='-', 
                            linewidth=1,
                            alpha=0.15) 
        
        line.set_alpha(1.0)
        legend_handles.append(line)
        legend_labels.append(alg_name)
    
    
    ax.set_xticks(range(len(objective_labels)))
    ax.set_xticklabels(objective_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Objective Function', fontsize=14)
    ax.set_ylabel('Objective Value', fontsize=14)
    ax.set_title(f'Parallel Coordinate Plot - Solution Distribution Comparison (Trial {TRIAL_INDEX_TO_PLOT})', fontsize=16)
    
    ax.legend(legend_handles, legend_labels, loc='upper center', title="Algorithm")
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
        
del alg_PFS_list, alg_name, ax, color, colors, fig, i, idx, individual_costs, legend_handles, legend_labels, line, objective_labels, PF_to_plot, selected_PF, TRIAL_INDEX_TO_PLOT, x_coords

# %% PLOT LINE CASE 2
for i in range(6):
    TRIAL_INDEX_TO_PLOT = 0
    ALG_list = [i] 
    objective_labels = ['$f_1$', '$f_2$', '$f_3$'] 
    
    colors = ['b', 'r', 'c', 'm', 'y', 'k'] 
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    legend_handles = []
    legend_labels = []
    
    for idx, (alg_name, alg_PFS_list) in enumerate(results_case2.items()):
        if idx not in ALG_list:
            continue
        color = colors[idx % len(colors)]
        selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]
        PF_to_plot = selected_PF[:, :len(objective_labels)]
        PF_to_plot = global_normalized(PF_to_plot, RP2)
        x_coords = np.arange(len(objective_labels))
        for i in range(PF_to_plot.shape[0]):
            
            individual_costs = PF_to_plot[i, :]
            line, = ax.plot(x_coords, individual_costs, 
                            color=color, 
                            linestyle='-', 
                            linewidth=1,
                            alpha=0.15) 
        
        line.set_alpha(1.0)
        legend_handles.append(line)
        legend_labels.append(alg_name)
    
    
    ax.set_xticks(range(len(objective_labels)))
    ax.set_xticklabels(objective_labels, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Objective Function', fontsize=14)
    ax.set_ylabel('Objective Value', fontsize=14)
    ax.set_title(f'Parallel Coordinate Plot - Solution Distribution Comparison (Trial {TRIAL_INDEX_TO_PLOT})', fontsize=16)
    
    ax.legend(legend_handles, legend_labels, loc='upper center', title="Algorithm")
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
        
del alg_PFS_list, alg_name, ax, color, colors, fig, i, idx, individual_costs, legend_handles, legend_labels, line, objective_labels, PF_to_plot, selected_PF, TRIAL_INDEX_TO_PLOT, x_coords
