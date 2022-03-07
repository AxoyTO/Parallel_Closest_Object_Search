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
