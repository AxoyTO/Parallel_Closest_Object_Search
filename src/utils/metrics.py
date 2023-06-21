import numpy as np
from math import sqrt, asin, cos, sin


def calculate_with_different_metrics(A, B):
    print("=========================================")
    print("Calculating Hausdorff Distance using different metrics")
    print("=========================================")
    start = time.time()
    print(f"Euclidean HD: {NaiveHDD(A, B, Metrics.euclidean):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Manhattan HD: {NaiveHDD(A, B, Metrics.manhattan):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Chebyshev HD: {NaiveHDD(A, B, Metrics.chebyshev):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Minkowski HD: {NaiveHDD(A, B, distance.minkowski):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Canberra HD:    {NaiveHDD(A, B, distance.canberra):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Cosine HD:      {NaiveHDD(A, B, Metrics.cosine):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    print("===========================")


class Metrics:
    @staticmethod
    def manhattan(array_x, array_y):
        n = array_x.shape[0]
        ret = 0.0
        for i in range(n):
            ret += abs(array_x[i] - array_y[i])
        return ret

    @staticmethod
    def chebyshev(array_x, array_y):
        n = array_x.shape[0]
        ret = -1 * np.inf
        for i in range(n):
            d = abs(array_x[i] - array_y[i])
            if d > ret:
                ret = d
        return ret

    @staticmethod
    def cosine(array_x, array_y):
        n = array_x.shape[0]
        xy_dot = 0.0
        x_norm = 0.0
        y_norm = 0.0
        for i in range(n):
            xy_dot += array_x[i] * array_y[i]
            x_norm += array_x[i] * array_x[i]
            y_norm += array_y[i] * array_y[i]
        return 1.0 - xy_dot / (sqrt(x_norm) * sqrt(y_norm))

    @staticmethod
    def haversine(array_x, array_y):
        R = 6378.0
        radians = np.pi / 180.0
        lat_x = radians * array_x[0]
        lon_x = radians * array_x[1]
        lat_y = radians * array_y[0]
        lon_y = radians * array_y[1]
        dlon = lon_y - lon_x
        dlat = lat_y - lat_x
        a = pow(sin(dlat / 2.0), 2.0) + cos(lat_x) * cos(lat_y) * pow(
            sin(dlon / 2.0), 2.0
        )
        return R * 2 * asin(sqrt(a))

    @staticmethod
    def euclidean(array_A, array_B):
        n = array_A.shape[0]
        ret = 0.0
        for i in range(n):
            ret += (array_A[i] - array_B[i]) ** 2
        return sqrt(ret)
