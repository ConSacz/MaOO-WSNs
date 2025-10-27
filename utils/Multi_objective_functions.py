import numpy as np
from utils.Single_objective_functions import CR_Func, LT_Func, CE_Func, SE_Func
from utils.Graph_functions import Graph, Connectivity_graph

# %% Cost function of 2 functions problem
def CostFunction_2F_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rs = stat[0,:]
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    Coverage, _ = CR_Func(pop, rs, Obstacle_Area, Covered_Area)
    LifeTime = LT_Func(G)
    
    Cost = np.array([[Coverage], [LifeTime]])
    
    return Cost

def CostFunction_2F_weighted(pop, stat, w, Obstacle_Area, Covered_Area):
    rs = stat[0,:]
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    Coverage, _ = CR_Func(pop, rs, Obstacle_Area, Covered_Area)
    LifeTime = LT_Func(G)
    
    Cost = w[0] * Coverage + w[1] * LifeTime
    
    return Cost

# %% Cost function of 3 functions problem with constraint
def CostFunction_3F1C_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    if Connectivity_graph(G)==1:
        Cost = np.zeros((1,3))
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
