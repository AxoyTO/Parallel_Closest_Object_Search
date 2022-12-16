from copy import copy
from functools import partial
import os
import time
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


def execute_scipy_hd(model_1, model_2):
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

    print_flushed = partial(print, flush=True)

    error = 0
    dir = os.getcwd()
    models_dir = "/Models/Faces"

    if rank == 0:

        files = os.listdir(models_dir[1:])
        models = [i[:-4] for i in files if i.endswith((".stl", ".STL", ".off", ".OFF"))]

        if len(models) < world_size:
            error = 1
            print_flushed("ERROR: Not enough models to run in parallel.")
            print_flushed(f"Process count: {world_size} | Model count: {len(models)}")
            print_flushed(
                "Process count must be less than or equal to the number of models."
            )
        for i in range(1, world_size):
            comm.send(error, dest=i, tag=i)
        if error:
            MPI.Finalize()
            print_flushed("Exiting...")
            exit()

        print_flushed("==================================")
        print_flushed(f"          WORLD SIZE: {world_size}          ")
        print_flushed("==================================")
        print_flushed(f"Total model count: {len(models)}")
        print_flushed(models)

        fixed_model_index = models.index(
            random.choice(models)
        )  # Pick a random model to be fixed
        print_flushed(f"Process {rank} picked model: {models[fixed_model_index]}.")
        print_flushed("==================================")
        fixed_model = models.pop(fixed_model_index)

        splits = np.array(models, dtype=object)
        splits = np.array_split(splits, (world_size - 1))

        models = [fixed_model]

        for i in range(len(splits)):
            comm.send(
                splits[i], dest=i + 1, tag=i + 1
            )  # Send models to other processes
            # print_flushed(f"Process {rank} sent {splits[i]} to process {i+1}.")
        comm.barrier()
    else:
        for i in range(1, world_size):
            if rank == i:
                error = comm.recv(source=0, tag=i)
                if error == 1:
                    MPI.Finalize()
                    exit()
        models = None

        for i in range(1, world_size):
            if rank == i:
                models = comm.recv(source=0, tag=i)
                # print_flushed(f"Process {rank} received model(s): {models}.")  # Receive models from process 0
        # print_flushed(f"{rank} has {models}")
        comm.barrier()  # Wait for all processes to receive models

    models_names = copy(models)

    for i in range(len(models)):
        # print_flushed(f"{rank} has {models[i]}")
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
            print_flushed(f"There is no file {models[i]} with extension .STL or .OFF!")
            comm.Abort(1)
        finally:
            if LOAD_OUTPUT:
                print_flushed(
                    f" {models[i].vertices.shape[0]} vertices is found and loaded by process {rank}!"
                )

    # print(models_names)

    for i in range(len(models)):
        models[i] = np.array(models[i].vertices)

    if rank == 0:
        fixed_model = models[0]
        fixed_model_name = models_names[0]
    else:
        fixed_model = None
        fixed_model_name = None

    fixed_model = comm.bcast(fixed_model, root=0)
    fixed_model_name = comm.bcast(fixed_model_name, root=0)

    results_dict = {}
    if rank != 0:
        results = [0] * len(models)
        for i in range(len(models)):
            results[i] = max(
                directed_hausdorff(fixed_model, models[i]),
                directed_hausdorff(models[i], fixed_model),
            )[0]
            results_dict[models_names[i]] = results[i]
            print_flushed(
                f"Process {rank} ScipyHD from {fixed_model_name} to {models_names[i]}: {results[i]:.6f}"
            )

    comm.barrier()
    res = comm.gather(results_dict, root=0)
    if rank == 0:
        for i in res:
            if len(i):
                for k, v in i.items():
                    results_dict[k] = v
        print_flushed(
            f"Closest model to {fixed_model_name} is {min(results_dict, key=results_dict.get)}"
        )

    MPI.Finalize()


# TODO: add timer (MPI.Wtime())
# TODO: for serial code(1 process)
# TODO: gracefully exit(no abort) if model file is not found
