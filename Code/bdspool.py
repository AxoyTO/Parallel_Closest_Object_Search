from functools import partial
import os
import random
import numpy as np
import trimesh
import asyncio
import threading
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

async def control_requests():
    print_flushed(f"Process {rank} is ready to receive requests")
    probe = comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    if probe:
        destination = comm.recv()
        print_flushed(f"Process {rank} received request from {destination}")
        if len(models) > 0:
            model_index = models.index(random.choice(models))
            model_name = models.pop(model_index)
            comm.send(model_name, dest=destination, tag=destination)

def receive_model_and_calculate_distance():

    if rank == 0:
        model_index = models.index(random.choice(models))
        model_name = models.pop(model_index)

    else:
        #print_flushed(f"Process {rank} is ready to receive model")
        model_name = comm.sendrecv(rank, dest=0, source = 0, sendtag=rank, recvtag=rank)

        if model_name is None:
            return 0
            
    calculate_distance(model_name)
    
def calculate_distance(model_name):
        model = load_model_by_name(model_name)
        results_dict[model_name] = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))[0]
        #results_dict[model_name] = max(EARLYBREAK(fixed_model, model),EARLYBREAK(model, fixed_model))
        print_flushed(f"Process {rank} calculated Hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")

async def receive_model_and_calculate_distance_for_0():
    model_index = models.index(random.choice(models))
    model_name = models.pop(model_index)
    calculate_distance(model_name)
    #print_flushed(f"Process {rank} says hi")


def async_control():
    asyncio.run(control_requests())

def async_receive():
    asyncio.run(receive_model_and_calculate_distance_for_0())

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
        print_flushed("==================================")
        print_flushed(f"          WORLD SIZE: {world_size}          ")
        print_flushed("==================================")
        print_flushed(f"Total model count: {len(models)}")
        #print_flushed(models)

        fixed_model_index = models.index(
            'Darren'
        )  # Pick a random model to be fixed
        print_flushed(f"Process {rank} picked model: {models[fixed_model_index]}.")
        print_flushed("==================================")

        start = MPI.Wtime()
        fixed_model_name = models.pop(fixed_model_index)
        fixed_model = load_model_by_name(fixed_model_name)

        if world_size > 1:
            for i in range(1, world_size):
                comm.send(fixed_model_name, dest=i, tag=i)  # Send fixed_model to other processes

    else:
        fixed_model_name = comm.recv(source=0) # Receive fixed_model from process 0
        models.pop(models.index(fixed_model_name))
        fixed_model = load_model_by_name(fixed_model_name)

        
    while True and world_size > 1:
        if rank == 0:
            if len(models)> 0:
                t1 = threading.Thread(target=async_control)
                t1.start()
                t1.join()
                
            else:
                for i in range(1, world_size):
                    comm.send(None, dest=i, tag=i)
                break
                
        else:
            result = receive_model_and_calculate_distance()
            if result != 0:
                pass
            else:
                break

    if world_size > 1:
        res = comm.gather(results_dict, root=0)
    else:
        res = []
        model_name = models
        for model_name in models:
            model = load_model_by_name(model_name)
            results_dict[model_name] = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))[0]
            #results_dict[model_name] = max(EARLYBREAK(fixed_model, model),EARLYBREAK(model, fixed_model))
            #print_flushed(f"Process {rank} calculated Hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")
        res.append(results_dict)
        results_dict = {}
        
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
            print(f"Parallel Search Time: {end - start :.5f} seconds.")
    
    MPI.Finalize()


# TODO: писать текст
# TODO: больше моделей
# TODO: запускать на POLUS
