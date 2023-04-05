from copy import copy, deepcopy
import os
import sys
from functools import partial
import time
import numpy as np
import trimesh
from hausdorff import *
from scipy.spatial.distance import directed_hausdorff
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

LOAD_OUTPUT = 0
METHOD = 'SCIPY_DIRECTED_HAUSDORFF'

if METHOD == 'KDTREE':
    sys.setrecursionlimit(10000)

def load_model_by_name(model_name):
        dir = os.getcwd()
        try:
            if os.path.exists(dir + f"{models_dir}/{model_name}.stl"):
                if LOAD_OUTPUT:
                    print(f"{model_name}.stl with", end="")
                print("1")
                model = trimesh.load(dir + f"/{models_dir}/{model_name}.stl", force="mesh")
                
            elif os.path.exists(dir + f"{models_dir}/{model_name}.off"):
                if LOAD_OUTPUT:
                    print(f"{model_name}.off with", end="")
                model = trimesh.load(dir + f"/{models_dir}/{model_name}.off", force="mesh")

            else:
                raise Exception
        except:

            print_flushed(f"There is no file {model_name} with extension .STL or .OFF!")
            comm.Abort(1)

        finally:

            if LOAD_OUTPUT:
                print_flushed(f" {model.vertices.shape[0]} vertices is found and loaded by process {rank}!")
            return np.array(model.vertices)

def calculate_distance(model_name):
        model = load_model_by_name(model_name)
        if METHOD == 'SCIPY_DIRECTED_HAUSDORFF':
            results_dict[model_name] = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))[0]
        elif METHOD == 'EARLYBREAK':
            results_dict[model_name] = max(EARLYBREAK(fixed_model, model),EARLYBREAK(model, fixed_model))
        elif METHOD == 'NAIVEHDD':
            results_dict[model_name] = max(NaiveHDD(fixed_model, model),NaiveHDD(model, fixed_model))
        elif METHOD == 'KDTREE':
            results_dict[model_name] = max(KDTree_Query(fixed_model, model),KDTree_Query(model, fixed_model))
        #print_flushed(f"Process {rank} calculated Hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")


if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    print_flushed = partial(print, flush=True)
    results_dict = {}
    models_dir = "/ModelSet"

    if rank == 0:
        models = [os.path.splitext(i)[0] for i in os.listdir(models_dir[1:]) if os.path.splitext(i)[1].lower() in {".stl", ".off"}]

        print_flushed("==================================")
        print_flushed(f"          WORLD SIZE: {world_size}          ")
        print_flushed("==================================")
        print_flushed(f"Chosen method: {METHOD}")
        print_flushed(f"Total model count: {len(models)}")
        print_flushed("--------------------------------------------------------")

        models_names = copy(models)
        # fixed_model = models.index(random.choice(models))
        fixed_model_index = models.index("airplane_0627")
        fixed_model_name = models.pop(fixed_model_index)

        splits = np.array(models, dtype=object)
        splits = np.array_split(splits, (world_size))
        models = splits[0]
    
        start = MPI.Wtime()
        for i in range(1, world_size):
            comm.send(fixed_model_name, dest=i, tag=1)
            comm.send(splits[i], dest=i, tag=0)

    else:
        fixed_model_name = comm.recv(source=0, tag=1)
        models = comm.recv(source=0, tag=0)

    for i in range(len(models)):
        fixed_model = load_model_by_name(fixed_model_name)
    
    for model in models:
        calculate_distance(model)

    if world_size > 1:
        res = comm.gather(results_dict, root=0)
    else:
        res = []
        res.append(results_dict)

    if rank == 0:
        end = MPI.Wtime()
        for i in res:
            if len(i):
                for k, v in i.items():
                    results_dict[k] = v
        if len(results_dict):            
            print_flushed(
                f"Closest model to {fixed_model_name} is {min(results_dict, key=results_dict.get)}"
            )
            print_flushed(f"Parallel Search Time: {end - start :.5f} seconds.")
        print_flushed("---------------------------------------------------")
    MPI.Finalize()
