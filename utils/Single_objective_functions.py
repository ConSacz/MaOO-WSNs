import numpy as np
import networkx as nx


def CR_Func(pop, Obstacle_Area, Covered_Area):

    """
    Calculate the Coverage Ratio as a cost function.
    Parameters:
        pop (x, y, rs) * N is the optimization variable
        Obstacle_Area: 1 means area need to cover; 0 means area dont need to cover
        Covered_Area: 1 means area covered; 0 means area not covered
    Returns:
        coverage (float): inverse-coverage ratio (lower is better).
    """
    
    # reset Covered Area
    Covered_Area[Covered_Area != 0] = 0

    inside_sector = np.zeros_like(Covered_Area, dtype=bool)
    for j in range(pop.shape[0]):
        # node position j-th
        x0 = pop[j, 0]
        y0 = pop[j, 1]
        rsJ = pop[j, 2]

        # boundary constraint
        x_ub = min(int(np.ceil(x0 + rsJ)), Covered_Area.shape[0])
        x_lb = max(int(np.floor(x0 - rsJ)), 0)
        y_ub = min(int(np.ceil(y0 + rsJ)), Covered_Area.shape[1])
        y_lb = max(int(np.floor(y0 - rsJ)), 0)

        # local grid
        X, Y = np.meshgrid(
            np.linspace(x_lb, x_ub, x_ub - x_lb + 1),
            np.linspace(y_lb, y_ub, y_ub - y_lb + 1)
        )

        # distance matrix
        D = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)

        # angle matrix
        Theta = np.arctan2(Y - y0, X - x0)
        Theta[Theta < 0] += 2 * np.pi

        # in rs condition
        in_circle = D <= rsJ

        # both conditions
        inside_sector[y_lb:y_ub + 1, x_lb:x_ub + 1] |= (in_circle)

    # covered area
    Covered_Area = inside_sector.astype(int) * Obstacle_Area

    # add obstacle to covered area
    obs_row, obs_col = np.where(Obstacle_Area == 0)
    for i in range(len(obs_col)):
        if Covered_Area[obs_row[i], obs_col[i]] == 1:
            Covered_Area[obs_row[i], obs_col[i]] = -2

    count1 = np.sum(Covered_Area == 1)     # covered points on wanted location
    count2 = np.sum(Covered_Area == -2)    # covered points on unwanted location
    count3 = np.sum(Obstacle_Area == 1)  # total wanted points

    coverage = 1 - (count1 - count2) / count3

    # recover obs covered area
    obs_row, obs_col = np.where(Covered_Area == -2)
    for i in range(len(obs_col)):
        Covered_Area[obs_row[i], obs_col[i]] = -1

    return coverage, Covered_Area

# %%
def LT_Func(G):
    """
    Calculate the normalized network lifetime as a cost function.
    Parameters:
        G (networkx.Graph): weighted graph with edge weights as distances.
    Returns:
        lifetime_normalized (float): inverse-lifetime (lower is better).
    """
    N = G.number_of_nodes()
    
    # Parameters
    b = 0.1   # nJ/bit/m^a
    a = 2     # path loss exponent
    EM = 0    # nJ/bit maintain/process
    ET = 20   # nJ/bit transmit
    ER = 2    # nJ/bit receive
    maxBat = 1000
    # 
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

    if np.max(Bat) == 0:
        lifetime = np.inf
    else:
        lifetime = maxBat / np.max(Bat)

    # Normalize (same logic as MATLAB version)
    lifetime_normalized = round(1 / lifetime, 5) if lifetime != 0 else 0

    return lifetime_normalized

# %%
def CE_Func(G):
    """
    Calculate the Communication Energy as a cost function.
    Parameters:
        G (networkx.Graph): weighted graph with edge weights as distances.
    Returns:
        E (float): total Communication Energy consumption (lower is better).
    """
    N = G.number_of_nodes()
    
    # Parameters
    b = 0.1   # nJ/bit/m^a
    a = 2     # path loss exponent
    EM = 0    # nJ/bit maintain/process
    ET = 20   # nJ/bit transmit
    ER = 2    # nJ/bit receive
    # 
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

    # Total Energy consumption
    E = np.sum(Bat)

    return E

# %%
def SE_Func(pop):
    """
    Calculate the Sensing Energy as a cost function.
    Parameters:
        pop (x, y, rs) * N is the optimization variable
    Returns:
        E (float): total Sensing Energy consumption (lower is better).
    """
    
    rs0 = pop[:, 2]
    E = 10**(-6)*np.sum(rs0*rs0) 
    
    return E
    
    
    
    
    
    
    