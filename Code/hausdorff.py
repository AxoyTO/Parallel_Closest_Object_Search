import numpy as np
from metrics import Metrics
from scipy.spatial import KDTree
from scipy.spatial.distance import directed_hausdorff

def NaiveHDD(model_a, model_b, distance_function=Metrics.euclidean):
    distances_list = []
    for i in range(len(model_a)):
        dist_min = np.inf
        for j in range(len(model_b)):
            dist = distance_function(model_a[i], model_b[j])
            if dist_min > dist:
                dist_min = dist
        distances_list.append(dist_min)
    return np.max(distances_list)


def EARLYBREAK(model_a, model_b, distance_function=Metrics.euclidean):
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


def KDTree_Query(model1, model2):
    tree = KDTree(model2)
    dist_1, _ = tree.query(model1)
    
    tree = KDTree(model1)
    dist_2, _ = tree.query(model2)
    
    return np.max([np.max(dist_1), np.max(dist_2)])
