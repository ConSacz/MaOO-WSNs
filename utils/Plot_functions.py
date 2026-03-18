import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from .Domination_functions import get_pareto_front
from .Normalize_functions import global_normalized
from .Single_objective_functions import CR_Func, SE_Func, CE_Func, LT_CE_SE_Func


# %% plot deployment
def plot_deployment2D(pop, stat, Obstacle_Area, Covered_Area):
    import networkx as nx
    from utils.Graph_functions import Graph
    
    N = pop.shape[0]
    rc = stat[1, 0]
    
    coverage, Covered_Area = CR_Func(pop, Obstacle_Area, Covered_Area)
    LT, CE, SE = LT_CE_SE_Func(pop, rc)
    
    plt.clf()
    # obs_row, obs_col = np.where(Obstacle_Area == 1)
    # plt.plot(obs_col, obs_row, '.', markersize=0.1, color='blue')
    # obs_row, obs_col = np.where(Obstacle_Area == 0)
    # plt.plot(obs_col, obs_row, '.', markersize=5, color='red')
    # discovered_obs_row, discovered_obs_col = np.where(Covered_Area == 0)
    # plt.plot(discovered_obs_row, discovered_obs_col, '.', markersize=1, color='red')
    # discovered_row, discovered_col = np.where(Covered_Area == 1)
    # plt.plot(discovered_row, discovered_col, '.', markersize=1, color='green')

    theta = np.linspace(0, 2*np.pi, 500)
    for i in range(N):
        plt.plot(pop[i,1], pop[i,0], 'o', markersize=3, color='blue')
        plt.text(pop[i,1], pop[i,0], str(i+1), fontsize=10, color='red')
        # x = pop[i,1] + pop[i,2] * np.cos(theta)
        # y = pop[i,0] + pop[i,2] * np.sin(theta)
        # plt.fill(x, y, color=(0.6, 1, 0.6), alpha=0.2, edgecolor='k')
        
        # draw rs

        # cx, cy = pop[i,1], pop[i,0]
        # r = pop[i,2]
        
        # # Chọn hướng bán kính (0 rad – sang phải)
        # ex = cx - r
        # ey = cy
        
        # plt.plot([cx, ex], [cy, ey], color='blue', linewidth=1)
        
        # # Ghi nhãn r_s_i ở giữa đoạn
        # mx = (cx + ex) / 2
        # my = (cy + ey) / 2 - 2
        # plt.text(mx, my, rf"$r_{{s_{i+1}}}$", fontsize=9, color='black')

    G = Graph(pop,rc)
    for i in range(N):
        try:
            path = nx.shortest_path(G, source=0, target=i, weight='weight')
        except nx.NetworkXNoPath:
            continue  # skip if no path exists
        for j in range(len(path)-1):
            x = path[j]
            y = path[j+1]
            plt.plot([pop[x,1], pop[y,1]],[pop[x,0], pop[y,0]],'-', color='gray', linewidth=0.8)
        
        
        
        
        # for j in range(i + 1, N):
        #     dx = pop[i,1] - pop[j,1]
        #     dy = pop[i,0] - pop[j,0]
        #     d = np.sqrt(dx*dx + dy*dy)

        #     if d <= rc:
        #         plt.plot([pop[i,1], pop[j,1]],
        #                  [pop[i,0], pop[j,0]],
        #                  '-', color='gray', linewidth=0.8)
                
    # del x, y, theta
    plt.xlim([0, Obstacle_Area.shape[1]])
    plt.ylim([0, Obstacle_Area.shape[0]])
    plt.title("Unconnected Network")
    # plt.title(f"Coverage cost function: {coverage:.5f}")
#     plt.title(
#     rf"$C = {coverage:.3f}, "
#     rf"L_{{inv}} = {LT:.5f}, "
#     rf"E^{{comm}} = {CE:.5f}, "
#     rf"E^{{sens}} = {SE:.5f}$"
# )
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
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # vẽ toàn bộ quần thể
    ax.scatter(F_set[:, 0], F_set[:, 1], F_set[:, 2], c='g', marker='o', label='Pareto Set')

    # vẽ pareto front
    ax.scatter(Front[:, 0], Front[:, 1], Front[:, 2], c='r', marker='o', label='Pareto Front')
    
    # draw Reference point
    if W is not None:
        ax.scatter(W[:, 0], W[:, 1], W[:, 2], c='b', marker='o', label='PF')
        
    # for i in range(len(W)):
    #     x1, y1, z1 = F_set[i]
    #     x2, y2, z2 = W[i]

    #     ax.text(x1, y1, z1, f"{i}", color='blue', fontsize=5)
    #     ax.text(x2, y2, z2, f"{i}", color='black', fontsize=5)
    ax.view_init(elev=20, azim=190)

    # nhãn trục
    ax.set_xlabel('Non-coverage')
    ax.set_ylabel('Communication Energy')
    ax.set_zlabel('Sensing Energy')

    # legend
    ax.legend()

    # hiển thị tạm để cập nhật liên tục theo iteration
    plt.pause(0.001)

# %% plot 3D global normalized
def plot3D_GN(pop, W, RP):
    # mỗi ind là dict có key 'Cost'
    F_set = np.array([ind['Cost'].flatten() for ind in pop])
    F_set = global_normalized(F_set, RP)

    # tạo figure và axes 3D
    fig = plt.figure(figsize=(6, 6))
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    

    # # vẽ toàn bộ quần thể
    # ax.scatter(F_set[:, 0], F_set[:, 1], F_set[:, 2], c='g', marker='o', label='Solution Points')
    # for i, f in enumerate(F_set):
    #     ax.text(
    #         f[0], f[1], f[2],
    #         f'({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f})',
    #         fontsize=4,
    #         color='darkgreen'
    #     )

    ax.scatter(W[:, 0], W[:, 1], W[:, 2], c='b', marker='o', label='Reference Points')
    for i, w in enumerate(W):
        ax.text(
            w[0], w[1], w[2],
            f'({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})',
            fontsize=4,
            color='blue'
        )

    
    # 
    L = 1.1
    for i in range(W.shape[0]):
        w = W[i] / np.linalg.norm(W[i])
        ax.plot(
            [0, L * w[0]],
            [0, L * w[1]],
            [0, L * w[2]],
            color='b',
            linewidth=0.5,
            linestyle='--',
            alpha=0.6
        )

    # for i in range(min(F_set.shape[0], W.shape[0])):
    #     f = F_set[i]
    #     w = W[i]
    
    #     # chuẩn hóa hướng
    #     w_hat = w / np.linalg.norm(w)
    
    #     # hình chiếu vuông góc của f lên đường thẳng w
    #     proj = np.dot(f, w_hat) * w_hat
    
    #     # vẽ đoạn thẳng khoảng cách
    #     ax.plot(
    #         [f[0], proj[0]],
    #         [f[1], proj[1]],
    #         [f[2], proj[2]],
    #         color='green',
    #         linewidth=1.0,
    #         alpha=1
    #     )
    
    # pop chỉ có 1 điểm
    
    f = F_set[4]
    ax.scatter(f[0], f[1], f[2], c='g', marker='o', label='Solution Points')
    ax.text(
        f[0], f[1], f[2],
        f'({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f})',
        fontsize=4,
        color='darkgreen'
    )
    
    projs = []
    dists = []
    
    for i in range(W.shape[0]):
        w = W[i]
        w_hat = w / np.linalg.norm(w)
    
        # hình chiếu vuông góc
        proj = np.dot(f, w_hat) * w_hat
    
        # khoảng cách
        dist = np.linalg.norm(f - proj)
    
        projs.append(proj)
        dists.append(dist)
    
    projs = np.array(projs)
    dists = np.array(dists)
    
    idx_min = np.argmin(dists)
    idx_max = np.argmax(dists)
    
    for i in range(W.shape[0]):
        if i == idx_min:
            color = 'red'
            lw = 1.0
            alpha = 1.0
        elif i == idx_max:
            color = 'green'
            lw = 1.0
            alpha = 1.0
        else:
            color = 'gray'
            lw = 0.4
            alpha = 0.4
    
        ax.plot(
            [f[0], projs[i][0]],
            [f[1], projs[i][1]],
            [f[2], projs[i][2]],
            color=color,
            linewidth=lw,
            alpha=alpha
        )

    ax.view_init(elev=10, azim=70)
    
    # nhãn trục
    ax.set_xlabel(r'$f_1$', labelpad=0)
    ax.set_ylabel(r'$f_2$', labelpad=0)
    ax.set_zlabel(r'$f_3$', labelpad=0)

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
