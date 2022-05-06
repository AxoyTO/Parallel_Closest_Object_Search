import numpy as np
from metrics import Metrics


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
