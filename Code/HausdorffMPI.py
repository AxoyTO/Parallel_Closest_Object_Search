import os
import numpy as np
import time
import trimesh
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from math import sqrt, asin, cos, sin
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def unique(a):
    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]


def manhattan(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += abs(array_x[i] - array_y[i])
    return ret


def chebyshev(array_x, array_y):
    n = array_x.shape[0]
    ret = -1 * np.inf
    for i in range(n):
        d = abs(array_x[i] - array_y[i])
        if d > ret:
            ret = d
    return ret


def cosine(array_x, array_y):
    n = array_x.shape[0]
    xy_dot = 0.
    x_norm = 0.
    y_norm = 0.
    for i in range(n):
        xy_dot += array_x[i] * array_y[i]
        x_norm += array_x[i] * array_x[i]
        y_norm += array_y[i] * array_y[i]
    return 1. - xy_dot / (sqrt(x_norm) * sqrt(y_norm))


def haversine(array_x, array_y):
    R = 6378.0
    radians = np.pi / 180.0
    lat_x = radians * array_x[0]
    lon_x = radians * array_x[1]
    lat_y = radians * array_y[0]
    lon_y = radians * array_y[1]
    dlon = lon_y - lon_x
    dlat = lat_y - lat_x
    a = (pow(sin(dlat / 2.0), 2.0) + cos(lat_x) *
         cos(lat_y) * pow(sin(dlon / 2.0), 2.0))
    return R * 2 * asin(sqrt(a))


def calculate_with_different_metrics(A, B):
    print("=========================================")
    print("Hausdorf Distance NaiveHDD (Rhino - Lion)")
    print("=========================================")
    start = time.time()
    print(f"Euclidean HD: {NaiveHDD(A, B, euclidean):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Manhattan HD: {NaiveHDD(A, B, manhattan):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    start = time.time()
    print(f"Chebyshev HD: {NaiveHDD(A, B, chebyshev):.6f}", end="")
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
    print(f"Cosine HD:      {NaiveHDD(A, B, cosine):.6f}", end="")
    end = time.time()
    print(f" ---- Time: {end - start :.5f} seconds.")
    print("===========================")


def euclidean(array_A, array_B):
    n = array_A.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_A[i] - array_B[i]) ** 2
    return sqrt(ret)


def NaiveHDD(model_a, model_b, distance_function=euclidean):
    distances_list = []
    for i in range(len(model_a)):
        dist_min = np.inf
        for j in range(len(model_b)):
            dist = distance_function(model_a[i], model_b[j])
            if dist_min > dist:
                dist_min = dist
        distances_list.append(dist_min)
    return np.max(distances_list)


if __name__ == "__main__":
    results = []
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        # print(f"HOST:", MPI.Get_processor_name())
        # print("----------------------------------")
        print("==================================")
        print(f"          WORLD SIZE: {world_size}          ")
        print("==================================")
        dir = os.getcwd()

        m1 = trimesh.load(dir + '/Models/GoldenRetriever.stl', force="mesh")
        m2 = trimesh.load(dir + '/Models/Rhino.stl', force="mesh")
        m3 = trimesh.load(dir + '/Models/Wolf.stl', force="mesh")

        m1_v = np.array(m1.vertices)
        m2_v = np.array(m2.vertices)
        m3_v = np.array(m3.vertices)
        models = [m1_v, m2_v, m3_v]

        print(m1_v.shape, m2_v.shape, m3_v.shape)

        start = time.time()
        
        print(f"Serial M1 -> M2: {NaiveHDD(m1_v, m2_v):.6f}")
        print(f"Serial M2 -> M1: {NaiveHDD(m2_v, m1_v):.6f}")
        print(f"Serial M1 -> M3: {NaiveHDD(m1_v, m3_v):.6f}")
        print(f"Serial M3 -> M1: {NaiveHDD(m3_v, m1_v):.6f}")
        end = time.time()
        print(f"Serial Elapsed Time: {end - start :.5f} seconds.")

        #calculate_with_different_metrics(m1_v, m2_v)


        print("---------------------------------------------------")
        scipy_res = directed_hausdorff(m1_v, m2_v)
        scipy_res_reverse = directed_hausdorff(m2_v, m1_v)
        print(
            f"Scipy HD from M1 to M2: {scipy_res[0]:.6f} -- Indexes: ({scipy_res[1]}, {scipy_res[2]})")
        print(
            f"Scipy HD from M2 to M1: {scipy_res_reverse[0]:.6f} -- Indexes: ({scipy_res_reverse[1]}, {scipy_res_reverse[2]})")

        scipy_res = directed_hausdorff(m1_v, m3_v)
        scipy_res_reverse = directed_hausdorff(m3_v, m1_v)
        print(
            f"Scipy HD from M1 to M3: {scipy_res[0]:.6f} -- Indexes: ({scipy_res[1]}, {scipy_res[2]})")
        print(
            f"Scipy HD from M3 to M1: {scipy_res_reverse[0]:.6f} -- Indexes: ({scipy_res_reverse[1]}, {scipy_res_reverse[2]})")
        


        split_m1 = np.array_split(m1_v, world_size, axis=0)
        split_m2 = np.array_split(m2_v, world_size, axis=0)
        split_m3 = np.array_split(m3_v, world_size, axis=0)
        start = MPI.Wtime()
        for i in range(1, world_size):
            comm.send(split_m1[i], dest=i, tag=1)
            comm.send(split_m2[i], dest=i, tag=2)
            comm.send(split_m3[i], dest=i, tag=3)
            # print(f"Process {rank} sent split model_1 to {i}")
            comm.send(m1_v, dest=i, tag=1)
            comm.send(m2_v, dest=i, tag=2)
            comm.send(m3_v, dest=i, tag=3)
            # print(f"Process {rank} sent model_2 to {i}")
        comm.barrier()
        split_m1 = split_m1[0]
        split_m2 = split_m2[0]
        split_m3 = split_m3[0]

    else:
        split_m1 = comm.recv(source=0)
        split_m2 = comm.recv(source=0)
        split_m3 = comm.recv(source=0)
        #print(f"Process {rank} received split model_3 of shape {split_m3.shape}")
        m1_v = comm.recv(source=0)
        m2_v = comm.recv(source=0)
        m3_v = comm.recv(source=0)
        # print(f"Process {rank} received model_2!")
        comm.barrier()

    result_m1_m2 = comm.gather(NaiveHDD(split_m1, m2_v), root=0)
    result_m2_m1 = comm.gather(NaiveHDD(split_m2, m1_v), root=0)
    result_m1_m3 = comm.gather(NaiveHDD(split_m1, m3_v), root=0)
    result_m3_m1 = comm.gather(NaiveHDD(split_m3, m1_v), root=0)
    if rank == 0:
        end = MPI.Wtime()
        print("---------------------------------------------------")
        print(f"MPI M1 -> M2: {max(result_m1_m2):.6f}")
        print(f"MPI M2 -> M1: {max(result_m2_m1):.6f}")
        print(f"MPI M1 -> M3: {max(result_m1_m3):.6f}")
        print(f"MPI M3 -> M1: {max(result_m3_m1):.6f}")
        print(f"Parallel Elapsed Time: {end - start :.5f} seconds.")
        print("---------------------------------------------------")
    MPI.Finalize()
