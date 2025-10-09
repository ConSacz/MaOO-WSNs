import numpy as np
#from Domination_functions import get_pareto_front

def normalized(data):
    x_sta = np.min(data, axis = 0)
    x_nad = np.max(data, axis = 0)
    data_normalized = (data - x_sta) / (x_nad - x_sta)
    
    return data_normalized

def weight_assign(pop,RP):
    pop.sort(key=lambda p: p['Cost'][0], reverse=True)
    # x_sta = RP[:,0].flatten()
    # x_nad = RP[:,1].flatten()
    x_sta = np.zeros(2, dtype=int)
    x_nad = np.ones(2, dtype=int)
    Npop = len(pop)
    Nf = len(pop[0]['Cost'])
    w = np.zeros((Npop, Nf), dtype=int)
    
    weights = np.linspace(x_sta,x_nad,Npop)
    weights[:, 1] = weights[::-1, 1]
    #weights[:, [0, 1]] = weights[:, [1, 0]]
    w = weights
    
    return pop, w
    