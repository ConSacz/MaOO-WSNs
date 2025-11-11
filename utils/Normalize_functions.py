import numpy as np

# local normalized
def local_normalized(Cost):
    x_sta = np.min(Cost, axis = 0)
    x_nad = np.max(Cost, axis = 0)
    nml_Cost = (Cost - x_sta) / (x_nad - x_sta)
    
    return nml_Cost

# global normalized
def global_normalized(Cost, RP):
    x_sta = RP[:,0]
    x_nad = RP[:,1]
    nml_Cost = (Cost - x_sta) / (x_nad - x_sta)
    
    return nml_Cost

