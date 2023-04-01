from functools import partial
#import multiprocessing as mp
import os
import random
import trimesh
from hausdorff import *
import mpi4py
import sys

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

LOAD_OUTPUT = 0
RESULT_OUTPUT = 0
METHOD = 'EARLYBREAK'

if METHOD == 'SCIPY_DH':
    sys.setrecursionlimit(10000)

def load_model_by_name(model_name):
        dir = os.getcwd()
        try:

            if os.path.exists(dir + f"{models_dir}/{model_name}.stl"):
                if LOAD_OUTPUT:
                    print(f"{model_name}.stl with", end="")
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

def control_requests():
    while len(models) > 0:
        destination = comm.recv()
        #print_flushed(f"Process {rank} received request from {destination}")
        model_index = models.index(random.choice(models))
        model_name = models.pop(model_index)
        comm.send(model_name, dest=destination, tag=destination)
    else:
        for i in range(1, world_size):
            comm.send(None, dest=i, tag=i)

def receive_model_and_calculate_distance():
    #print_flushed(f"Process {rank} is ready to receive model")
    while True:
        model_name = comm.sendrecv(rank, dest=0, source = 0, sendtag=rank, recvtag=rank)

        if model_name is None:
            break
        
        calculate_distance(model_name)
    
def calculate_distance(model_name):
        model = load_model_by_name(model_name)
        if METHOD == 'SCIPY_DH':
            results_dict[model_name], _, _ = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))
        elif METHOD == 'EARLYBREAK':
            results_dict[model_name] = max(EARLYBREAK(fixed_model, model),EARLYBREAK(model, fixed_model))
        elif METHOD == 'NAIVEHDD':
            results_dict[model_name] = max(NaiveHDD(fixed_model, model),NaiveHDD(model, fixed_model))
        elif METHOD == 'KDTREE':
            results_dict[model_name] = max(KDTree_Query(fixed_model, model),KDTree_Query(model, fixed_model))
        print_flushed(f"Process {rank} calculated Hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")

if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    results_dict = {}

    print_flushed = partial(print, flush=True)

    models_dir = "/ModelSet"
    models = sorted([os.path.splitext(i)[0] for i in os.listdir(models_dir[1:]) if os.path.splitext(i)[1].lower() in {".stl", ".off"}])[:15]

    if rank == 0:
        print_flushed("==================================")
        print_flushed(f"          WORLD SIZE: {world_size}          ")
        print_flushed("==================================")
        print_flushed(f"Chosen method: {METHOD}")
        print_flushed(f"Total model count: {len(models)}")
        #print_flushed(models)

        fixed_model_index = models.index('airplane_0627')
        print_flushed(f"Process {rank} picked model: {models[fixed_model_index]}.")
        print_flushed("==================================")
        fixed_model_name = models[fixed_model_index]
        start = MPI.Wtime()
    else:
        fixed_model_name = None

    
    fixed_model_name = comm.bcast(fixed_model_name, root = 0)
    models.pop(models.index(fixed_model_name))
    fixed_model = load_model_by_name(fixed_model_name)

        
    if world_size > 1:
        if rank == 0:
            #p1 = mp.Process(target = control_requests)
            #p1.start()
            #p1.join()
            control_requests()
        else:
            receive_model_and_calculate_distance()

    #print_flushed(f"{rank} finished")

    if world_size > 1:
        res = comm.gather(results_dict, root=0)
        #print_flushed(f"{rank} passed")
    else:
        res = []
        model_name = models
        for model_name in models:
            calculate_distance(model_name)
        res.append(results_dict)
        results_dict = {}
        
    if rank == 0:
        end = MPI.Wtime()
        for i in res:
            if len(i):
                for k, v in i.items():
                    results_dict[k] = v
        if len(results_dict):            
            print_flushed(f"Closest model to {fixed_model_name} is {min(results_dict, key=results_dict.get)}")
            print_flushed(f"Parallel Search Time: {end - start :.5f} seconds.")
    
    MPI.Finalize()