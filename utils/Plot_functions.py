import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from .Domination_functions import get_pareto_front
from .Normalize_functions import global_normalized

# %% plot deployment
def plot_deployment2D(pop, Obstacle_Area, Covered_Area):
    N = pop.shape[0]
    
    plt.clf()
    obs_row, obs_col = np.where(Obstacle_Area == 1)
    plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')
    obs_row, obs_col = np.where(Obstacle_Area == 0)
    plt.plot(obs_col, obs_row, '.', markersize=2, color='black')
    discovered_obs_row, discovered_obs_col = np.where(Covered_Area == -1)
    plt.plot(discovered_obs_col, discovered_obs_row, '.', markersize=2, color='red')
    #discovered_row, discovered_col = np.where(Covered_Area == 1)
    #plt.plot(discovered_col, discovered_row, '.', markersize=5, color='green')

    theta = np.linspace(0, 2*np.pi, 500)
    for i in range(N):
        plt.plot(pop[i,1], pop[i,0], 'o', markersize=3, color='blue')
        plt.text(pop[i,1], pop[i,0], str(i+1), fontsize=10, color='red')
        x = pop[i,1] + pop[i,2] * np.cos(theta)
        y = pop[i,0] + pop[i,2] * np.sin(theta)
        plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.2, edgecolor='k')
        
    del x, y, theta
    plt.xlim([0, Obstacle_Area.shape[1]])
    plt.ylim([0, Obstacle_Area.shape[0]])
    #plt.title(f"{BestCostIt[it]*100:.2f}% at time step: {it}")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.pause(0.001)


# %% plot 2D
def plot2D(pop):
    Front = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])  # mỗi ind là dict có key 'Cost'
    F_set = np.array([ind['Cost'].flatten() for ind in pop])

    # Tạo figure
    plt.figure(1)
    plt.clf()
    
    # Vẽ Pareto front
    plt.plot(F_set[:, 0], F_set[:, 1], 'o', color='g')
    plt.plot(Front[:, 0], Front[:, 1], 'o', color='b', label = 'PF')
    #plt.plot(data2[:, 0], data2[:, 1], 'o', color='r', label = 'NSWABC')
    #plt.plot(data3[:, 0], data3[:, 1], 'o', color='g', label = 'NSWABC')
    # for i in range(len(F_set)):
    #     x, y = F_set[i]
    #     plt.text(x, y, f"{w[i,0]:.1f} {w[i,1]:.1f}", fontsize=8, ha='right', va='bottom', color='blue')
    plt.legend()
    # plt.xlim(RP[0,0], RP[0,1])
    # plt.ylim(RP[1,0], RP[1,1])
    plt.xlabel('Non-coverage')
    plt.ylabel('Energy')
    None
    # Cập nhật đồ thị theo từng iteration
    plt.pause(0.001)

# %% plot 3D
def plot3D(pop, W = None):
    # mỗi ind là dict có key 'Cost'
    Front = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    F_set = np.array([ind['Cost'].flatten() for ind in pop])

    # tạo figure và axes 3D
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # vẽ toàn bộ quần thể
    ax.scatter(F_set[:, 0], F_set[:, 1], F_set[:, 2], c='g', marker='o', label='Pareto Set')

    # vẽ pareto front
    ax.scatter(Front[:, 0], Front[:, 1], Front[:, 2], c='r', marker='o', label='Pareto Front')
    
    # draw Reference point
    if W is not None:
        ax.scatter(W[:, 0], W[:, 1], W[:, 2], c='r', marker='o', label='PF')
    
    ax.view_init(elev=20, azim=190)

    # nhãn trục
    ax.set_xlabel('Non-coverage')
    ax.set_ylabel('Communication Energy')
    ax.set_zlabel('Sensing Energy')

    # legend
    ax.legend()

    # hiển thị tạm để cập nhật liên tục theo iteration
    plt.pause(0.001)

# %% plot 3D adjustable figure
def plot3D_adjustable(pop, name = ''):
    Front = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    F_set = np.array([ind['Cost'].flatten() for ind in pop])
    points = Front[:, :3]  # f1, f2, f3
    set_points = F_set[:, :3]
    # cloud = pv.PolyData(points)
    # cloud2 = pv.PolyData(set_points)
    
    # # scaling points
    mins = set_points.min(axis=0)
    maxs = set_points.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    ranges_safe = np.where(ranges == 0, 1.0, ranges)
    scale_factors = max_range / ranges_safe
    points_scaled = (points - mins) * scale_factors
    points2_scaled = (set_points - mins) * scale_factors
    cloud = pv.PolyData(points_scaled)
    cloud2 = pv.PolyData(points2_scaled)
    
    # gen plotter
    plotter = pv.Plotter()
    plotter.add_points(
        cloud2,
        color="green",                # color
        point_size=8,                # size
        render_points_as_spheres=True  # sphere point
    )
    plotter.add_points(
        cloud,
        color="red",                # color
        point_size=8,                # size
        render_points_as_spheres=True  # sphere point
    )

    plotter.show_grid(
        xtitle='f1',
        ytitle='f2',
        ztitle='f3',
        color='gray',
        grid='back',     # vẽ lưới phía sau điểm
        location='outer' # hiển thị nhãn ngoài khung
    )
    plotter.show_bounds(grid='front', color='black')
    plotter.add_axes(line_width=10)
    #plotter.add_text("IMDEA Pareto Front", position='upper_edge', font_size=14, color='black')
    plotter.view_vector((-35, -25, 1))  # try view_isometric(), view_yx(),...
    plotter.show(title=f"{name} Pareto Front 3D")

# %% plot MaOO
def plot_MaOO(F, RP):
    """
    F: ma trận kích thước (N_obj, nPop)
       - N_obj: số hàm mục tiêu
       - nPop : số nghiệm
    """
    F = np.asarray(F)
    F = global_normalized(F, RP)
    nPop, N_obj = F.shape

    x = np.arange(1, N_obj + 1)

    plt.figure(figsize=(8, 5))

    for i in range(nPop):
        plt.plot(x, F[i,:], marker='o', linewidth=1)


    plt.xlabel("Objective Index")
    plt.ylabel("Objective Value")
    plt.title("Multi-objective Line Plot")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# %% test
# fig = plt.figure(1)
# plt.clf()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(F2[:, 0], F2[:, 1], F2[:, 2], c='g', marker='o', label='Pareto Set')
# ax.scatter(W[:, 0], W[:, 1], W[:, 2], c='r', marker='o', label='Weight')
# n = min(len(F2), len(W))
# for i in range(n):
#     x1, y1, z1 = F2[i]
#     x2, y2, z2 = W[i]

#     ax.text(x1, y1, z1, f"{i}", color='blue', fontsize=5)
#     ax.text(x2, y2, z2, f"{i}", color='black', fontsize=5)
    
# ax.view_init(elev=20, azim=190)
