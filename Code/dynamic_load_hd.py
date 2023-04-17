import random
from utilities import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def control_requests():
    while len(models) > 0:
        destination = comm.recv()
        model_index = models.index(random.choice(models))
        model_name = models.pop(model_index)
        comm.send(model_name, dest=destination, tag=destination)
    else:
        for i in range(1, world_size):
            comm.send(None, dest=i, tag=i)

def receive_model_and_calculate_distance():
    while True:
        model_name = comm.sendrecv(rank, dest=0, source = 0, sendtag=rank, recvtag=rank)

        if model_name is None:
            break
        
        calculate_distance(fixed_model, model_name, comm)
        print_flushed(f"Process {rank} calculated Hausdorff distance from {fixed_model_name} to {model_name}: {results_dict[model_name]:.6f}")
    

if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        models = sorted([os.path.splitext(i)[0] for i in os.listdir(models_dir[1:]) if os.path.splitext(i)[1].lower() in {".stl", ".off"}])[0:5]
        print_opening(world_size, len(models), fixed_model_name, "DLB")

        models.pop(models.index(fixed_model_name))

        start = MPI.Wtime()
    else:
        fixed_model_name = None

    
    fixed_model_name = comm.bcast(fixed_model_name, root = 0)
    fixed_model = load_model_by_name(fixed_model_name, comm)
        
    if world_size > 1:
        if rank == 0:
            control_requests()
        else:
            receive_model_and_calculate_distance()

    if world_size > 1:
        res = comm.gather(results_dict, root=0)
    else:
        res = []
        model_name = models
        for model_name in models:
            calculate_distance(fixed_model, model_name, comm)
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