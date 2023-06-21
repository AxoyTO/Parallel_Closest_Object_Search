import numpy as np
from utils.metrics import Metrics
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

def naivehdd(model_a, model_b, distance_function=Metrics.euclidean):
    cmax = 0.0
    for i in range(len(model_a)):
        cmin = np.inf
        for j in range(len(model_b)):
            d = distance_function(model_a[i], model_b[j])
            if d < cmin:
                cmin = d
        if cmax < cmin:
            cmax = cmin
    return cmax

def earlybreak(model_a, model_b, distance_function=Metrics.euclidean):
    nA = model_a.shape[0]
    nB = model_b.shape[0]
    cmax = 0.0
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(model_a[i], model_b[j])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmax < cmin:
            cmax = cmin
    return cmax

def earlybreak_with_rs(model_a, model_b, distance_function=Metrics.euclidean, seed=0):
    nA = model_a.shape[0]
    nB = model_b.shape[0]
    data_dims = model_a.shape[1]
    rng = np.random.RandomState(seed)
    resort1 = np.arange(nA, dtype=np.int64)
    resort2 = np.arange(nB, dtype=np.int64)
    rng.shuffle(resort1)
    rng.shuffle(resort2)
    model_a = np.asarray(model_a)[resort1]
    model_b = np.asarray(model_b)[resort2]

    cmax = 0.0
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = 0.0
            #d = distance_function(model_a[i], model_b[j])
            for k in range(data_dims):
                d += (model_a[i,k] - model_b[j,k])**2
            if d < cmax:
                break

            if d < cmin:
                cmin = d

        if cmin >= cmax and d >= cmax:
            cmax = cmin


    return cmax**(1/2)

def kdtree_query(model1, model2):
    tree = KDTree(model2)
    dist_1, _ = tree.query(model1, workers=-1)
    
    tree = KDTree(model1)
    dist_2, _ = tree.query(model2, workers=-1)
    
    return np.max([np.max(dist_1), np.max(dist_2)])
