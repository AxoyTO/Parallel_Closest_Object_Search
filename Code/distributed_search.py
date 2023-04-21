from copy import copy
from utilities import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

if __name__ == "__main__":
    MPI.Init()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        models = [os.path.splitext(i)[0] for i in os.listdir(models_dir[1:]) if os.path.splitext(i)[1].lower() in {".stl", ".off"}]
        print_opening(world_size, len(models), fixed_model_name, "DS")
        
        if fixed_model_name in models:
            models.pop(models.index(fixed_model_name))

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

    fixed_model = load_model_by_name(fixed_model_name, comm)
    
    for model in models:
        calculate_distance(fixed_model, model, comm)

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
    MPI.Finalize()
