import numpy as np
from utils.Single_objective_functions import Cov_Func, LT_Func
from utils.Graph_functions import Graph

def CostFunction_MOO(pop, stat, Obstacle_Area, Covered_Area):
    rs = stat[0,:]
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    Coverage, _ = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
    LifeTime = LT_Func(G)
    
    Cost = np.array([[Coverage], [LifeTime]])
    
    return Cost

def CostFunction_weighted(pop, stat, w, Obstacle_Area, Covered_Area):
    rs = stat[0,:]
    rc = stat[1,0]
    G = Graph(pop,rc)
    
    Coverage, _ = Cov_Func(pop, rs, Obstacle_Area, Covered_Area)
    LifeTime = LT_Func(G)
    
    Cost = w[0] * Coverage + w[1] * LifeTime
    
    return Cost