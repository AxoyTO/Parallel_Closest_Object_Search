from copy import copy
from functools import partial
import os
import time
import random
import numpy as np
import trimesh
from hausdorff import *
from scipy.spatial.distance import directed_hausdorff
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

LOAD_OUTPUT = 0
RESULT_OUTPUT = 0
SCIPY_CALC = 1

def load_model_by_name(model_name):
        try:
            if os.path.exists(dir + f"{models_dir}/{model_name}.stl"):
                if LOAD_OUTPUT:
                    print(f"{model_name}.stl with", end="")
                model = trimesh.load(
                    dir + f"/{models_dir}/{model_name}.stl", force="mesh"
                )
            elif os.path.exists(dir + f"{models_dir}/{model_name}.off"):
                if LOAD_OUTPUT:
                    print(f"{model_name}.off with", end="")
                model = trimesh.load(
                    dir + f"/{models_dir}/{model_name}.off", force="mesh"
                )
            else:
                raise Exception
        except:
            print_flushed(f"There is no file {model_name} with extension .STL or .OFF!")
            comm.Abort(1)
        finally:
            if LOAD_OUTPUT:
                print_flushed(
                    f" {models[i].vertices.shape[0]} vertices is found and loaded by process {rank}!"
                )
            return np.array(model.vertices)


if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    results_dict = {}

    print_flushed = partial(print, flush=True)

    error = 0
    dir = os.getcwd()
    models_dir = "/Models/Faces"

    files = os.listdir(models_dir[1:])
    models = [i[:-4] for i in files if i.endswith((".stl", ".STL", ".off", ".OFF"))]

    if rank == 0:

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

        start = MPI.Wtime()
        if world_size > 1:
            fixed_model = models.pop(fixed_model_index)
            for i in range(1, world_size):
                comm.send(fixed_model, dest=i, tag=i)  # Send fixed_model to other processes

                #print_flushed(f"Process {rank} sent {fixed_model} to process {i}.")
                #print_flushed(f"Process {rank} sent {models} to process {i}.")
            #comm.barrier()
            

    else:
        for i in range(1, world_size):
            if rank == i:
                error = comm.recv(source=0, tag=i)
                if error == 1:
                    MPI.Finalize()
                    exit()

        fixed_model_name = comm.recv(source=0) # Receive fixed_model from process 0
        models.pop(models.index(fixed_model_name))
        fixed_model = load_model_by_name(fixed_model_name)
        #print(rank, fixed_model[:2])
        #print_flushed(f"Process {rank} received fixed model: {fixed_model}.")
        #print_flushed(f"Process {rank} has models: {models}.")
        
    #comm.barrier()
    MPI.Finalize()
    while True:
        # Request model from process 0
        if rank == 0:
            if len(models) > 0:
                #print_flushed(f"Process 0 has {len(models)} models left. {models}")
                destination = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                #print_flushed(f"Process 0 received a request from {destination}")
                model_index = models.index(random.choice(models))
                model = models.pop(model_index)
                comm.send(model, dest=destination, tag=destination)
                #print_flushed(f"Process 0 sent {model} to {destination}.")
                
            else:
                for i in range(1, world_size):
                    comm.send(None, dest=i, tag=i)
        else:
            # Send rank to process 0 to request a model
            comm.send(rank, dest=0, tag=rank)
            #print_flushed(f"Process {rank} sent a request to 0")

            # Receive model index from process 0
            model_name = comm.recv(source=0, tag=rank)

            # If there are no more elements left, break out of the loop
            if model_name is None:
                #print_flushed(f"Process {rank} received None.")
                break
            
            model = load_model_by_name(model_name)
            results_dict[model_name] = max(EARLYBREAK(fixed_model, model),EARLYBREAK(model, fixed_model))
            print_flushed(f"Process {rank} calculated hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")
            
            # Do something with element here
            #my_result = model * rank
            #print(f"Process {rank} picked element {model_name}")
    
    #print(f"Process {rank} has results: {results_dict}")
    
    #comm.barrier()
    res = comm.gather(results_dict, root=0)
    if rank == 0:
        print(rank, res)
        end = MPI.Wtime()
        for i in res:
            if len(i):
                for k, v in i.items():
                    results_dict[k] = v
        if len(results_dict):            
            print_flushed(
                f"Closest model to {fixed_model_name} is {min(results_dict, key=results_dict.get)}"
            )
            print(f"Parallel Search Time: {end - start :.5f} seconds.")

    del model
    del results_dict
    MPI.Finalize()

    #if rank == 0:
        #MPI.Finalize()
    
    # print(fixed_model, models)
    #models_names = copy(models)

# TODO: change scipy hausdorff to custom hausdorff
# TODO: for serial code(1 process)
# TODO: gracefully exit(no abort) if model file is not found
# TODO: process pool for parallel code
# TODO: write text
# TODO: more models
