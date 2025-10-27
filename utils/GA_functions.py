import numpy as np


# %% ----------------------------
# GA original functions
# ----------------------------
def Crossover(x1, x2):
    alpha = np.random.rand(*x1.shape)
    y1 = alpha * x1 + (1 - alpha) * x2
    y2 = alpha * x2 + (1 - alpha) * x1
    return y1, y2

def Mutate(x, mu, sigma):
    x = x.copy()
    nVar = x.size
    nMu = int(np.ceil(mu * nVar))
    
    j = np.random.choice(nVar, nMu, replace=False)

    if np.ndim(sigma) > 0 and len(sigma) > 1:
        sigma = sigma[j]
    
    if np.isscalar(sigma):
        noise = sigma * np.random.randn(len(j))
    else:
        noise = sigma * np.random.randn(len(j))
    
    x.flat[j] += noise  # dùng flat để đánh index 1 chiều

    return x
# ---------- crossover ----------

def crossover_binomial(x, v, Cr):
    D = x.shape[0]
    u = x.copy()
    jrand = np.random.randint(0, D)
    mask = np.random.rand(D) <= Cr
    mask[jrand] = True
    u[mask] = v[mask]
    return u

def crossover_exponential(x, v, Cr):
    D = x.shape[0]
    u = x.copy()
    start = np.random.randint(0, D)
    L = 0
    while L < D:
        j = (start + L) % D
        u[j] = v[j]
        L += 1
        if np.random.rand() > Cr:
            break
    return u
# %% ----------------------------
# SBX crossover
# ----------------------------
def sbx_crossover(parent1, parent2, eta=20, pc=1.0, xmin=None, xmax=None):
    D = parent1.size
    child1 = parent1.copy()
    child2 = parent2.copy()
    if np.random.rand() <= pc:
        for i in range(D):
            if np.random.rand() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    x1 = min(parent1[i], parent2[i])
                    x2 = max(parent1[i], parent2[i])
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (x1 - xmin[i]) / (x2 - x1)) if xmin is not None else 1.0
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

                    beta = 1.0 + (2.0 * (xmax[i] - x2) / (x2 - x1)) if xmax is not None else 1.0
                    alpha = 2.0 - beta ** -(eta + 1)
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

                    if xmin is not None:
                        c1 = np.maximum(c1, xmin[i])
                        c2 = np.maximum(c2, xmin[i])
                    if xmax is not None:
                        c1 = np.minimum(c1, xmax[i])
                        c2 = np.minimum(c2, xmax[i])

                    if np.random.rand() <= 0.5:
                        child1[i] = c2
                        child2[i] = c1
                    else:
                        child1[i] = c1
                        child2[i] = c2
    return child1, child2

# %% ----------------------------
# Polynomial mutation
# ----------------------------
def polynomial_mutation(x, eta=20, pm=None, xmin=None, xmax=None):
    D = x.size
    y = x.copy()
    if pm is None:
        pm = 1.0 / float(D)
    for i in range(D):
        if np.random.rand() < pm:
            xi = x[i]
            xl = xmin[i] if xmin is not None else xi - 1.0
            xu = xmax[i] if xmax is not None else xi + 1.0
            if xl == xu:
                continue
            delta1 = (xi - xl) / (xu - xl)
            delta2 = (xu - xi) / (xu - xl)
            rand = np.random.rand()
            mut_pow = 1.0 / (eta + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
                deltaq = 1.0 - val ** mut_pow
            xi = xi + deltaq * (xu - xl)
            xi = np.clip(xi, xl, xu)
            y[i] = xi
    return y