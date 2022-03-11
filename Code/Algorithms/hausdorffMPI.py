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


def euclidean(array_A, array_B):
    n = array_A.shape[0]
    ret = 0.0
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

        m1 = trimesh.load(dir + "/Models/Lion.stl", force="mesh")
        m2 = trimesh.load(dir + "/Models/Wolf.stl", force="mesh")

        m1_v = np.array(m1.vertices)
        m2_v = np.array(m2.vertices)

        print(m1_v.shape, m2_v.shape)

        start = time.time()
        print(f"Serial M1 -> M2: {max(NaiveHDD(m1_v, m2_v), NaiveHDD(m2_v, m1_v)):.6f}")
        end = time.time()
        print(f"Serial Elapsed Time: {end - start :.5f} seconds.")

        print("---------------------------------------------------")

        split_m1 = np.array_split(m1_v, world_size, axis=0)
        split_m2 = np.array_split(m2_v, world_size, axis=0)
        split_m1 = np.array(split_m1, dtype=object)
        split_m2 = np.array(split_m2, dtype=object)
        print(split_m1.shape)
        print(split_m2.shape)

        start = MPI.Wtime()
        for i in range(1, world_size):
            comm.send(split_m1[i], dest=i, tag=1)
            print(f"Process {rank} sent split model_{1} of part {i} to process {i}")
            comm.send(split_m2[i], dest=i, tag=2)
            print(f"Process {rank} sent split model_{2} part {i} to process {i}")

            comm.send(m1_v, dest=i, tag=1)
            print(f"Process {rank} sent model_{1} to process {i}")
            comm.send(m2_v, dest=i, tag=2)
            print(f"Process {rank} sent model_{2} to process {i}")
        comm.barrier()
        split_m1 = split_m1[0]
        split_m2 = split_m2[0]
        print("split_m1 shape = ", split_m1.shape)
        print("split_m2 shape = ", split_m2.shape)

    else:
        split_m1 = comm.recv(source=0)
        split_m1 = np.array(split_m1)
        print(
            f"Process {rank} received split model_1 of shape {np.array(split_m1).shape}"
        )

        split_m2 = comm.recv(source=0)
        split_m2 = np.array(split_m2)
        print(
            f"Process {rank} received split model_2 of shape {np.array(split_m2).shape}"
        )
        m1_v = comm.recv(source=0)
        print(f"Process {rank} received model_1!")
        m2_v = comm.recv(source=0)
        print(f"Process {rank} received model_2!")
        # print(f"Process {rank} received model_2!")
        comm.barrier()

    result_m1_m2 = comm.gather(NaiveHDD(split_m1, m2_v), root=0)
    print(rank, result_m1_m2)
    result_m2_m1 = comm.gather(NaiveHDD(split_m2, m1_v), root=0)
    print(rank, result_m2_m1)

    if rank == 0:
        end = MPI.Wtime()
        print("---------------------------------------------------")
        print(f"MPI M1 -> M2: {max(max(result_m1_m2),max(result_m2_m1)):.6f}")
        print(f"Parallel Elapsed Time: {end - start :.5f} seconds.")
        print("---------------------------------------------------")
    MPI.Finalize()
