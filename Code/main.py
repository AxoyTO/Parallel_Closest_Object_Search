from copy import copy, deepcopy
import os
import time
import re
import random
import numpy as np
import trimesh
from hausdorff import *
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

LOAD_OUTPUT = 1


def execute_in_serial():
    print("---------------------   SERIAL   ---------------------")
    start = time.time()
    for i in range(len(models)):
        if fixed_model != i:
            print(
                f"Serial M{fixed_model+1} <-> M{i+1}: {max(EARLYBREAK(models[fixed_model], models[i]), EARLYBREAK(models[i], models[fixed_model])):.6f}"
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
        models_dir = "/Models/HighPoly"

        #find = re.compile(r"^[^.]*")

        files = os.listdir(models_dir[1:])

        models = [i[:-4]
                  for i in files if i.endswith(('.stl', '.STL', '.off', '.OFF'))]

        models_names = copy(models)
        fixed_model = models.index(random.choice(models))
        print(
            f"Picked model: {models[fixed_model]}. Total model count: {len(models)}")
        print("--------------------------------------------------------")

        split = [0] * len(models)

        for i in range(len(models)):
            # print(dir + f"{models_dir}/{models[i]}.off")
            try:
                if os.path.exists(dir + f"{models_dir}/{models[i]}.stl"):
                    if LOAD_OUTPUT:
                        print(f"{models[i]}.stl with", end="")
                    models[i] = trimesh.load(
                        dir + f"/{models_dir}/{models[i]}.stl", force="mesh"
                    )
                elif os.path.exists(dir + f"{models_dir}/{models[i]}.off"):
                    if LOAD_OUTPUT:
                        print(f"{models[i]}.off with", end="")
                    models[i] = trimesh.load(
                        dir + f"/{models_dir}/{models[i]}.off", force="mesh"
                    )
                else:
                    raise Exception
            except:
                print(
                    f"There is no file {models[i]} with extension .STL or .OFF!")
                comm.Abort(1)
                exit()
            finally:
                if LOAD_OUTPUT:
                    print(
                        f" {models[i].vertices.shape[0]} vertices is found and loaded!"
                    )

        for i in range(len(models)):
            models[i] = np.array(models[i].vertices)

        # execute_in_serial()
        # execute_scipy_hd()
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
            result = comm.gather(EARLYBREAK(
                split[i], models[fixed_model]), root=0)
            if result != None:
                directed_result = max(result)
            result = comm.gather(EARLYBREAK(
                split[fixed_model], models[i]), root=0)
            if result != None:
                results[i] = max(max(result), directed_result)

    if rank == 0:
        end = MPI.Wtime()
        results_pair = {}
        for i in range(len(results)):
            if results[i] != 0:
                # print(f"MPI M{fixed_model+1} <-> M{i+1}: {results[i]:.6f}")
                print(
                    f"MPI {models_names[fixed_model]} <-> {models_names[i]}: {results[i]:.6f}"
                )
                results_pair[models_names[i]] = results[i]
        print(f"Parallel Elapsed Time: {end - start :.5f} seconds.")
        print(min(results_pair, key=results_pair.get))
        print("---------------------------------------------------")
    MPI.Finalize()
