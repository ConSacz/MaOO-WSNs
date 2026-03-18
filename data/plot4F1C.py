try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.Workspace_functions import load_mat, load_spydata
from utils.Normalize_functions import global_normalized

# %% DATA AUTO IMPORT
Trials = 5
algorithm_configs = {
    'DWIABC': 'DWABC',
    'IMDEA': 'IMDEA',
    'MOEAD': 'MOEAD_DU',
    'NSGA3': 'NSGA3',
    'NSGA3OSD': 'NSGA3_OSD',
    'RVEA': 'RVEA'
}

# CASE 1
folder_name = r'data\4F1C\case1'
results_case1 = {}
RP_case1 = {}
time_case1 = {}
RP1 = np.zeros((4, 2))   
RP1[:,0] = [1, 1, 1, 1]          # first col are ideal values
RP1[:,1] = [1e-12, 1e-12, 1e-12, 1e-12]    # second col are nadir values
 
for result_name, file_prefix in algorithm_configs.items():  
    current_algorithm = []
    current_RP = []
    for i in range(Trials):
        RP = np.zeros((4, 2))
        RP[:, 0] = [1, 1, 1, 1]            # ideal
        RP[:, 1] = [1e-12]*4 

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
folder_name = r'data\4F1C\case2'
results_case2 = {}
RP_case2 = {}
time_case2 = {}
RP2 = np.zeros((4, 2))   
RP2[:,0] = [1, 1, 1, 1]          # first col are ideal values
RP2[:,1] = [1e-12, 1e-12, 1e-12, 1e-12]    # second col are nadir values

for result_name, file_prefix in algorithm_configs.items():
    current_algorithm = []
    current_RP = []
    for i in range(Trials):
        RP = np.zeros((4, 2))
        RP[:, 0] = [1, 1, 1, 1]            # ideal
        RP[:, 1] = [1e-12]*4 
        
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
    
del data, pop, PF, i, file_name, folder_name, algorithm_configs, Trials, file_prefix
del result_name, current_algorithm, RP, current_RP

# %% IMPORT FILTERED DATA
filename = r'data\Res_Matrixes\Results_4F1C_v2.spydata'
data = load_spydata(filename)
results_case1 = data['results_case1']
results_case2 = data['results_case2']
RP_case1 = data['RP_case1']
RP_case2 = data['RP_case2']
time_case1 = data['time_case1']
time_case2 = data['time_case2']

N_obj = 4

ideal_global = np.full(N_obj, np.inf)
nadir_global = np.full(N_obj, -np.inf)

for alg_name, RP_trials in RP_case1.items():
    for RP in RP_trials:
        # RP shape: (4, 2)
        ideal_global = np.minimum(ideal_global, RP[:, 0])
        nadir_global = np.maximum(nadir_global, RP[:, 1])

RP1 = np.zeros((N_obj, 2))
RP1[:, 0] = ideal_global
RP1[:, 1] = nadir_global

ideal_global = np.full(N_obj, np.inf)
nadir_global = np.full(N_obj, -np.inf)

for alg_name, RP_trials in RP_case2.items():
    for RP in RP_trials:
        # RP shape: (4, 2)
        ideal_global = np.minimum(ideal_global, RP[:, 0])
        nadir_global = np.maximum(nadir_global, RP[:, 1])

RP2 = np.zeros((N_obj, 2))
RP2[:, 0] = ideal_global
RP2[:, 1] = nadir_global

del filename, data, N_obj, ideal_global, nadir_global, alg_name, RP, RP_trials

# %% PLOT LINE
cases = [
    ('case1', results_case1, RP1),
    ('case2', results_case2, RP2),
]

for case_name, results_case, RP_use in cases:

    for alg_idx in range(6):
        TRIAL_INDEX_TO_PLOT = 0
        ALG_list = [alg_idx]
        objective_labels = ['$f_1$', '$f_2$', '$f_3$', '$f_4$']
        colors = ['b', 'r', 'c', 'm', 'y', 'k']

        fig, ax = plt.subplots(figsize=(12, 8))
        legend_handles = []
        legend_labels = []

        for idx, (alg_name, alg_PFS_list) in enumerate(results_case.items()):
            if idx not in ALG_list:
                continue

            color = colors[idx % len(colors)]
            selected_PF = alg_PFS_list[TRIAL_INDEX_TO_PLOT]
            PF_to_plot = selected_PF[:, :len(objective_labels)]
            PF_to_plot = global_normalized(PF_to_plot, RP_use)

            x_coords = np.arange(len(objective_labels))

            for k in range(PF_to_plot.shape[0]):
                individual_costs = PF_to_plot[k, :]
                line, = ax.plot(
                    x_coords, individual_costs,
                    color=color,
                    linestyle='-',
                    linewidth=1,
                    alpha=0.15
                )

            line.set_alpha(1.0)
            legend_handles.append(line)
            legend_labels.append(alg_name)

            save_path = os.path.join(
                "data", "Figures", "4F1C", case_name,
                f"{alg_name}_{case_name}.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        ax.set_xticks(range(len(objective_labels)))
        ax.set_xticklabels(objective_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Objective Function', fontsize=14)
        ax.set_ylabel('Objective Value', fontsize=14)
        ax.set_title(
            f'Parallel Coordinate Plot - Solution Distribution Comparison ({case_name.capitalize()})',
            fontsize=16
        )
        ax.legend(legend_handles, legend_labels,
                  loc='upper center', title="Algorithm")
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


del alg_PFS_list, alg_name, ax, color, colors, fig, alg_idx, idx, individual_costs, legend_handles, legend_labels
del k, line, objective_labels, PF_to_plot, selected_PF, TRIAL_INDEX_TO_PLOT, x_coords, save_path, ALG_list
del case_name, cases, RP_use, results_case

# %% Spider plot case1
cases = [
    # ('case1', RP_case1, time_case1, RP1),
    ('case2', RP_case2, time_case2, RP2),
]

for case_name, RP_case, time_case, RP_use in cases:

    alg_names = list(RP_case.keys())
    N_obj = 4

    ideal_avg = {}
    time_avg = {}

    for alg_name in alg_names:
        RP_trials = RP_case[alg_name]          # list of (4,2)
        ideals = np.array([RP[:, 0] for RP in RP_trials])
        ideal_avg[alg_name] = ideals.mean(axis=0)
        time_avg[alg_name] = time_case[alg_name]
    labels = [r'$Coverage$', r'$Lifetime$', r'$Comm Energy$', r'$Sens Energy$', 'Runtime']
    num_vars = len(labels)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    # %%
    for alg in alg_names:

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        data = np.concatenate([
            global_normalized(ideal_avg[alg], RP_use),
            [time_avg[alg]]
        ])
        data = np.concatenate([data, data[:1]])

        ax.plot(angles, data, linewidth=2)
        ax.fill(angles, data, alpha=0.25)

        ax.set_ylim(0, 1)
        ax.invert_yaxis()
        ax.set_thetagrids(np.array(angles[:-1]) * 180 / np.pi, labels)

        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['best (0.0)', 'medium (0.5)', 'worst (1.0)'])

        ax.set_title(
            f'Spider Plot – Ideal Values & Runtime\n{alg} ({case_name.upper()})',
            fontsize=14
        )

        ax.grid(True, linestyle='--', alpha=0.6)

        save_path = os.path.join(
            "data", "Figures", "4F1C", case_name,
            f"spider_{alg}_{case_name}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


del alg, alg_name, alg_names, angles, ax, fig, ideal_avg, ideals, labels, N_obj, num_vars
del RP_trials, save_path, time_avg, case_name, cases, RP_use, RP_case, time_case
