import os
import numpy as np
import time
import trimesh
from scipy.spatial.distance import directed_hausdorff
from math import sqrt
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

        m1 = trimesh.load(dir + '/Models/Rhino.stl', force="mesh")
        m2 = trimesh.load(dir + '/Models/Lion.stl', force="mesh")

        m1_v = np.array(m1.vertices)
        m2_v = np.array(m2.vertices)

        start = time.time()
        print(f"Serial: {NaiveHDD(m1_v, m2_v):.6f}")
        end = time.time()
        print(f"Serial Elapsed Time: {end - start :.5f} seconds.")
        scipy_res = directed_hausdorff(m1_v, m2_v)
        # print(f"Scipy: {scipy_res[0]:.6f}\nIndexes: ({scipy_res[1]}, {scipy_res[2]})")
        split = np.array_split(m1_v, world_size, axis=0)
        # print("---------------------------------------------------")
        start = MPI.Wtime()
        for i in range(1, world_size):
            comm.send(split[i], dest=i, tag=1)
            # print(f"Process {rank} sent split model_1 to {i}")
            comm.send(m2_v, dest=i, tag=2)
            # print(f"Process {rank} sent model_2 to {i}")
        comm.barrier()
        split = split[0]

    else:
        m1 = None
        m2 = None
        split = None

        split = comm.recv(source=0)
        # print(f"Process {rank} received split model_1 of shape {split.shape}")
        m2_v = comm.recv(source=0)
        # print(f"Process {rank} received model_2!")
        comm.barrier()

    result = comm.gather(NaiveHDD(split, m2_v), root=0)
    if rank == 0:
        end = MPI.Wtime()
        # print("---------------------------------------------------")
        # print(result)
        # print(f"MPI: {result}")
        print(f"MPI: {max(result):.6f}")
        print(f"MPI Elapsed Time: {end - start :.5f} seconds.")
        print("----------------------------------------")
    MPI.Finalize()
