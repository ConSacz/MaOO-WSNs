import numpy as np
from utils.Single_objective_functions import CR_Func, CE_Func, SE_Func
from utils.Graph_functions import Graph, Connectivity_graph

# %% Cost function of 2 functions problem
def CostFunction_2F1C_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    if Connectivity_graph(G)==1:
        Cost = np.zeros((1, 2))
        Cost[0,0], _ = CR_Func(pop, Obstacle_Area, Covered_Area)
        Cost[0,1] = CE_Func(G)
        return Cost
    else:
        return np.ones((1, 2))

def CostFunction_2F1C_weighted(pop, stat, w, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    if Connectivity_graph(G)==1:
        Coverage, _ = CR_Func(pop, Obstacle_Area, Covered_Area)
        ComEnergy = CE_Func(G)
        Cost = w[0] * Coverage + w[1] * ComEnergy
        return Cost
    else:
        return np.int(2)
    
def CostFunction_nml2F1C_MOO(pop, stat, RP, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    x_nad = RP[:,0]
    x_sta = RP[:,1]
    
    if Connectivity_graph(G)==1:
        Cost = np.zeros((1, 2))
        Cost[0,0], _ = CR_Func(pop, Obstacle_Area, Covered_Area)
        Cost[0,1] = CE_Func(G)
        return (Cost - x_sta) / (x_nad - x_sta)
    else:
        return np.ones((1, 2))

# %% Cost function of 3 functions problem with constraint
def CostFunction_3F1C_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    if Connectivity_graph(G)==1:
        Cost = np.zeros((1, 3))
        Cost[0,0], _ = CR_Func(pop, Obstacle_Area, Covered_Area)
        Cost[0,1] = CE_Func(G)
        Cost[0,2] = SE_Func(pop)
        return Cost
    else:
        return np.ones((1, 3))

def CostFunction_3F1C_weighted(pop, stat, w, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    if Connectivity_graph(G)==1:
        Coverage, _ = CR_Func(pop, Obstacle_Area, Covered_Area)
        ComEnergy = CE_Func(G)
        SenEnergy = SE_Func(pop)
        Cost = w[0] * Coverage + w[1] * ComEnergy + w[2]*SenEnergy
        return Cost
    else:
        return np.int(3)
