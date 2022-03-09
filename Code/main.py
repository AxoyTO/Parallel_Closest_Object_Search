import os
import numpy as np
import sys
import time
import trimesh
from hausdorff import *
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


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


def execute_in_serial():
    print("---------------------   SERIAL   ---------------------")
    start = time.time()
    for i in range(len(models)):
        if fixed_model != i:
            print(
                f"Serial M{fixed_model+1} <-> M{i+1}: {max(NaiveHDD(models[fixed_model], models[i]), NaiveHDD(models[i], models[fixed_model])):.6f}"
            )
    end = time.time()
    print(f"Serial Elapsed Time: {end - start :.5f} seconds.")


def execute_scipy_hd():
    print("---------------------   SCIPY   ---------------------")
    for i in range(len(models)):
        if fixed_model != i:
            print(
                f"Scipy HD from M{fixed_model+1} to M{i+1}: {max(directed_hausdorff(models[fixed_model], models[i]),directed_hausdorff(models[i], models[fixed_model]))[0]:.6f}"
            )


if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        print("==================================")
        print(f"          WORLD SIZE: {world_size}          ")
        print("==================================")
        dir = os.getcwd()

        models = [
            "GoldenRetriever",
            "Wolf",
            "Horse",
            "Lion",
            "Eagle",
            "Dolphin",
            "Bull",
            "Dog",
            "Camel",
            "Giraffe",
            "Human",
        ]
        fixed_model = models.index("Human")
        print(f"Picked model: {models[fixed_model]}. Total model count: {len(models)}")

        split = [0] * len(models)

        for i in range(len(models)):
            try:
                if os.path.exists(dir + f"/Models/{models[i]}.stl"):
                    print(f"{models[i]}.stl found and loaded!")
                    models[i] = trimesh.load(
                        dir + f"/Models/{models[i]}.stl", force="mesh"
                    )
                elif os.path.exists(dir + f"/Models/{models[i]}.off"):
                    print(f"{models[i]}.off found and loaded!")
                    models[i] = trimesh.load(
                        dir + f"/Models/{models[i]}.off", force="mesh"
                    )
                else:
                    raise Exception

            except:
                print(f"There is no file {models[i]} with extension .STL or .OFF!")
                comm.Abort(1)
                exit()

        for i in range(len(models)):
            models[i] = np.array(models[i].vertices)

        # for i in models:
        #    print(i.shape)

        # execute_in_serial()
        execute_scipy_hd()
        print("---------------------   PARALLEL   ---------------------")

        splits = []

        for i in range(len(models)):
            splits.append(np.array_split(models[i], world_size, axis=0))

        splits = np.array(splits, dtype=object)

        start = MPI.Wtime()
        for i in range(1, world_size):
            comm.send(fixed_model, dest=i, tag=i)
            comm.send(models, dest=i, tag=i)

            for j in range(len(models)):
                comm.send(splits[j][i], dest=i, tag=i)

        comm.barrier()

        for i in range(len(models)):
            split[i] = splits[i][0]

    else:
        fixed_model = comm.recv(source=0)
        models = comm.recv(source=0)
        split = [0] * len(models)
        for i in range(len(models)):
            split[i] = comm.recv(source=0)
            split[i] = np.array(split[i])
        comm.barrier()

    results = [0] * len(models)
    for i in range(len(models)):
        if i != fixed_model:
            result = comm.gather(NaiveHDD(split[i], models[fixed_model]), root=0)
            if result != None:
                directed_result = max(result)
            result = comm.gather(NaiveHDD(split[fixed_model], models[i]), root=0)
            if result != None:
                results[i] = max(max(result), directed_result)

    if rank == 0:
        end = MPI.Wtime()
        print("---------------------------------------------------")
        for i in range(len(results)):
            if results[i] != 0:
                print(f"MPI M{fixed_model+1} <-> M{i+1}: {results[i]:.6f}")
        print(f"Parallel Elapsed Time: {end - start :.5f} seconds.")
        print("---------------------------------------------------")
    MPI.Finalize()

# TODO: передать модели по разным процессам и между ними вычислять расстояние
