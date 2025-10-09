import numpy as np
import networkx as nx


def Cov_Func(pop, rs, Obstacle_Area, Covered_Area):
    # reset vùng đã phủ
    Covered_Area[Covered_Area != 0] = 0

    inside_sector = np.zeros_like(Covered_Area, dtype=bool)
    size_x, size_y = Covered_Area.shape
# %%
    for j in range(pop.shape[0]):
        x0, y0 = pop[j]
        rsJ = rs[j]

        # ràng buộc biên
        x_ub = min(int(np.ceil(x0 + rsJ)), size_x - 1)
        x_lb = max(int(np.floor(x0 - rsJ)), 0)
        y_ub = min(int(np.ceil(y0 + rsJ)), size_y - 1)
        y_lb = max(int(np.floor(y0 - rsJ)), 0)

        # tạo lưới con
        X, Y = np.meshgrid(np.arange(x_lb, x_ub + 1), np.arange(y_lb, y_ub + 1), indexing='ij')

        D = np.sqrt((X - x0)**2 + (Y - y0)**2)
        in_circle = D <= rsJ

        # cập nhật vùng trong sector
        inside_sector[x_lb:x_ub + 1, y_lb:y_ub + 1] |= (in_circle)

# %%
    # cập nhật vùng phủ
    Covered_Area = inside_sector * Obstacle_Area

    # xử lý vùng vật cản bị phủ
    mask_obstacle = (Obstacle_Area == 0) & (Covered_Area == 1)
    Covered_Area[mask_obstacle] = -2

    count1 = np.count_nonzero(Covered_Area == 1)
    count2 = np.count_nonzero(Covered_Area == -2)
    count3 = np.count_nonzero(Obstacle_Area == 1)
    coverage = 1 - (count1 - count2) / count3 if count3 > 0 else 0

    Covered_Area[Covered_Area == -2] = -1

    return coverage, Covered_Area

# %%
def LT_Func(G):
    """
    Calculate the normalized network lifetime as a fitness function.
    Parameters:
        G (networkx.Graph): weighted graph with edge weights as distances.
    Returns:
        lifetime_normalized (float): inverse-normalized lifetime (lower is better).
    """
    N = G.number_of_nodes()
    
    # Parameters
    b = 0.1   # nJ/bit/m^a
    a = 2     # path loss exponent
    EM = 0    # nJ/bit maintain/process
    ET = 20   # nJ/bit transmit
    ER = 2    # nJ/bit receive
    maxBat = 1000
    # %%
    Bat = np.zeros(N)

    for j in range(1, N):  # from node 1 to N-1 (Python uses 0-indexing)
        try:
            path = nx.shortest_path(G, source=0, target=j, weight='weight')
        except nx.NetworkXNoPath:
            continue  # skip if no path exists
        for i in range(len(path)):
            if i == 0:
                continue  # do nothing for the source node
            elif i == len(path) - 1:
                dt = G[path[i]][path[i - 1]]['weight']
                Bat[path[i]] += ((N+1) * EM + ET + b * dt ** a)
            else:
                dt = G[path[i]][path[i - 1]]['weight']
                dr = G[path[i]][path[i + 1]]['weight']
                Bat[path[i]] += (ER + ET + b * dt ** a + b * dr ** a)


    # %%
    if np.max(Bat) == 0:
        lifetime = np.inf
    else:
        lifetime = maxBat / np.max(Bat)

    # Normalize (same logic as MATLAB version)
    lifetime_normalized = round(1 / lifetime, 5) if lifetime != 0 else 0

    return lifetime_normalized