import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from .Domination_functions import get_pareto_front

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

def plot3D(pop):
    # mỗi ind là dict có key 'Cost'
    Front = np.array([ind['Cost'].flatten() for ind in get_pareto_front(pop)])
    F_set = np.array([ind['Cost'].flatten() for ind in pop])

    # tạo figure và axes 3D
    fig = plt.figure(1)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')

    # vẽ toàn bộ quần thể
    ax.scatter(F_set[:, 0], F_set[:, 1], F_set[:, 2], c='g', marker='o', label='Population')

    # vẽ pareto front
    ax.scatter(Front[:, 0], Front[:, 1], Front[:, 2], c='b', marker='o', label='PF')

    # nhãn trục
    ax.set_xlabel('Non-coverage')
    ax.set_ylabel('Energy')
    ax.set_zlabel('Sensing Energy')

    # legend
    ax.legend()

    # hiển thị tạm để cập nhật liên tục theo iteration
    plt.pause(0.001)

 
def plot3D_adjustable(pop):
    F = np.array([ind['Cost'] for ind in pop])[:, 0]
    points = F[:, :3]  # f1, f2, f3
    #cloud = pv.PolyData(points)
    
    # scaling points
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    ranges_safe = np.where(ranges == 0, 1.0, ranges)
    scale_factors = max_range / ranges_safe
    points_scaled = (points - mins) * scale_factors
    cloud = pv.PolyData(points_scaled)
    
    # gen plotter
    plotter = pv.Plotter()
    plotter.add_points(
        cloud,
        color="blue",                # color
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
    plotter.add_text("IMDEA Pareto Front", position='upper_edge', font_size=14, color='black')
    plotter.view_vector((-35, -25, 1))  # try view_isometric(), view_yx(),...
    plotter.show(title="IMDEA Pareto Front 3D")