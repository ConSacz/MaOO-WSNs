try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import numpy as np
import matplotlib.pyplot as plt
import os
from utils.Workspace_functions import load_mat
from utils.Normalize_functions import global_normalized

# %% DATA IMPORT
Trials = 5
algorithm_configs = {
    'DWIABC': 'DWABC',
    'IMDEA': 'IMDEA',
    'MOEAD_DU': 'MOEAD_DU',
    'NSGA3': 'NSGA3',
    'NSGA3OSD': 'NSGA3_OSD',
    'RVEA': 'RVEA'
}

# CASE 1
folder_name = r'data\3F1C\case1'
results_case1 = {}
RP_case1 = {}
time_case1 = {}
RP1 = np.zeros((3, 2))   
RP1[:,0] = [1, 1, 1]          # first col are ideal values
RP1[:,1] = [1e-12]*3          # second col are nadir values
 
for result_name, file_prefix in algorithm_configs.items():
    current_algorithm = []
    current_RP = []
    for i in range(Trials):
        RP = np.zeros((3, 2))
        RP[:, 0] = [1, 1, 1]            # ideal
        RP[:, 1] = [1e-12]*3  
        file_name = f'{file_prefix}_{i}.mat'
        data = load_mat(folder_name, file_name)
        pop = data['pop']
        PF = np.array([ind['Cost'].flatten() for ind in pop])
        RP1[:,0] = np.minimum(RP1[:,0], np.min(PF, axis=0))
        RP1[:,1] = np.maximum(RP1[:,1], np.max(PF, axis=0))
        RP[:, 0] = np.minimum(RP[:, 0], np.min(PF, axis=0))
        RP[:, 1] = np.maximum(RP[:, 1], np.max(PF, axis=0))
        current_algorithm.append(PF)
        current_RP.append(RP)
    results_case1[result_name] = current_algorithm
    RP_case1[result_name] = current_RP
    time_case1[result_name] = 1

# CASE 2
folder_name = r'data\3F1C\case2'
results_case2 = {}
RP_case2 = {}
time_case2 = {}
RP2 = np.zeros((3, 2))   
RP2[:,0] = [1, 1, 1]          # first col are ideal values
RP2[:,1] = [1e-12]*3             # second col are nadir values

for result_name, file_prefix in algorithm_configs.items():

    current_algorithm = []
    current_RP = []
    for i in range(Trials):
        RP = np.zeros((3, 2))
        RP[:, 0] = [1, 1, 1]            # ideal
        RP[:, 1] = [1e-12]*3 
        file_name = f'{file_prefix}_{i}.mat'
        data = load_mat(folder_name, file_name)
        pop = data['pop']
        PF = np.array([ind['Cost'].flatten() for ind in pop])
        RP2[:,0] = np.minimum(RP2[:,0], np.min(PF, axis=0))
        RP2[:,1] = np.maximum(RP2[:,1], np.max(PF, axis=0))
        RP[:, 0] = np.minimum(RP[:, 0], np.min(PF, axis=0))
        RP[:, 1] = np.maximum(RP[:, 1], np.max(PF, axis=0))
        current_algorithm.append(PF)
        current_RP.append(RP)
    results_case2[result_name] = current_algorithm
    RP_case2[result_name] = current_RP
    time_case2[result_name] = 1
    
del data, pop, PF, i, file_name, folder_name, algorithm_configs, Trials, file_prefix, result_name
del current_algorithm, RP, current_RP

# %% PLOT PARETO CASE 1
cases = {
    'case1': (results_case1, 1),
    'case2': (results_case2, 0),
}

# Colors and Markers
colors = ['b', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 's', '^', 'D', 'p', 'h']

for case_name, (results_case, TRIAL_INDEX_TO_PLOT) in cases.items():

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, (alg_name, alg_PFS_list) in enumerate(results_case.items()):

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # PF của trial được chọn
        selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]

        X = selected_PF[:, 0]
        Y = selected_PF[:, 1]
        Z = selected_PF[:, 2]

        ax.scatter(
            X, Y, Z,
            color=color,
            marker=marker,
            alpha=0.8,
            s=30,
            label=f'{alg_name}'
        )

    ax.set_xlabel('$f_1$ Coverage', fontsize=12)
    ax.set_ylabel('$f_2$ Comm Energy', fontsize=12)
    ax.set_zlabel('$f_3$ Sens Energy', fontsize=12)

    ax.set_title(
        f'Comparison of Pareto Fronts (PFs) – {case_name.upper()} ',
        fontsize=14
    )

    ax.legend(loc='center', bbox_to_anchor=(0.8, 0.2))
    ax.view_init(elev=20, azim=190)

    plt.show()
    plt.close(fig)

del alg_PFS_list, alg_name, ax, color, colors, fig, idx, marker, markers, selected_PF, X, Y, Z, TRIAL_INDEX_TO_PLOT
del case_name, cases, results_case

# %% PLOT LINE
cases = {
    'case1': (results_case1, RP1),
    'case2': (results_case2, RP2),
}

TRIAL_INDEX_TO_PLOT = 3
objective_labels = ['$f_1$', '$f_2$', '$f_3$']
colors = ['b', 'r', 'c', 'm', 'y', 'k']

for case_name, (results_case, RP_global) in cases.items():

    for alg_idx in range(len(results_case)):

        ALG_list = [alg_idx]

        fig, ax = plt.subplots(figsize=(12, 8))

        legend_handles = []
        legend_labels = []

        for idx, (alg_name, alg_PFS_list) in enumerate(results_case.items()):
            if idx not in ALG_list:
                continue

            color = colors[idx % len(colors)]
            selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]

            PF_to_plot = selected_PF[:, :len(objective_labels)]
            PF_to_plot = global_normalized(PF_to_plot, RP_global)

            x_coords = np.arange(len(objective_labels))

            for k in range(PF_to_plot.shape[0]):
                individual_costs = PF_to_plot[k, :]
                line, = ax.plot(
                    x_coords,
                    individual_costs,
                    color=color,
                    linestyle='-',
                    linewidth=1,
                    alpha=0.15
                )

            # dùng 1 line đại diện cho legend
            line.set_alpha(1.0)
            legend_handles.append(line)
            legend_labels.append(alg_name)

        ax.set_xticks(range(len(objective_labels)))
        ax.set_xticklabels(objective_labels, fontsize=12, fontweight='bold')

        ax.set_ylim(0, 1)
        ax.set_xlabel('Objective Function', fontsize=14)
        ax.set_ylabel('Objective Value', fontsize=14)

        ax.set_title(
            f'Parallel Coordinate Plot – {case_name.upper()} ',
            fontsize=16
        )

        ax.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            title='Algorithm'
        )

        ax.grid(True, linestyle='--', alpha=0.6)

        plt.show()
        plt.close(fig)

        
del alg_PFS_list, alg_name, ax, color, colors, fig, idx, individual_costs, legend_handles, legend_labels, line, objective_labels, PF_to_plot, selected_PF, TRIAL_INDEX_TO_PLOT, x_coords
del alg_idx, ALG_list, case_name, cases, k, results_case, RP_global

# %% BOX PLOT
cases = {
    'case1': (RP_case1, RP1),
    'case2': (RP_case2, RP2),
}

for case_name, (RP_case, RP_global) in cases.items():

    alg_names = list(RP_case.keys())

    for obj_idx in range(3):

        fig, ax = plt.subplots(figsize=(8, 8))

        data = []
        for alg in alg_names:
            ideals = np.array([RP[:, 0] for RP in RP_case[alg]])   # (n_trial, 3)
            ideals_norm = ideals
            # ideals_norm = global_normalized(ideals, RP_global)
            data.append(ideals_norm[:, obj_idx])

        ax.boxplot(data, showfliers=True)

        ax.set_xticks(range(1, len(alg_names) + 1))
        ax.set_xticklabels(alg_names, rotation=15)

        ax.set_ylabel('Normalized objective value')
        # ax.set_ylim(-0.05, 1.05)

        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

        # ===== lưu hình =====
        save_path = os.path.join(
            "data", "Figures", "3F1C", case_name,
            f"box_plot{obj_idx+1}_{case_name}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

del fig, ideals, ideals_norm, obj_idx, RP_case, RP_global, save_path, alg, alg_names, ax, case_name, cases
del data