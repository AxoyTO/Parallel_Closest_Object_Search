import os
from utils.hausdorff import *
from config import *
import trimesh

def load_model_by_name(models_dir, model_name, comm):
    try:
        if os.path.exists(f"{models_dir}/{model_name}.stl"):
            if LOAD_OUTPUT:
                print(f"{model_name}.stl with ", end="")
            model = trimesh.load(f"{models_dir}/{model_name}.stl", force="mesh")
        elif os.path.exists(f"{models_dir}/{model_name}.off"):
            if LOAD_OUTPUT:
                print(f"{model_name}.off with ", end="")
            model = trimesh.load(f"{models_dir}/{model_name}.off", force="mesh")
        else:
            print(f"{models_dir}/{model_name}")
            raise Exception
    except:
        print_flushed(f"File {model_name} couldn't be loaded. Ensure it exists in the models directory.")
        comm.Abort(1)
    finally:
        if LOAD_OUTPUT:
            print_flushed(f"{model.vertices.shape[0]} vertices is found and loaded by process {comm.Get_rank()}!")
        return np.array(model.vertices)

def calculate_distance(results_dict, models_dir, fixed_model, model_name, comm):
    model = load_model_by_name(models_dir, model_name, comm)
    if METHOD == 'SCIPY_DH':
        results_dict[model_name], _, _ = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))
    elif METHOD == 'EB':
        results_dict[model_name] = max(earlybreak(fixed_model, model),earlybreak(model, fixed_model))
    elif METHOD == 'EB_RS':
        results_dict[model_name] = max(earlybreak_with_rs(fixed_model, model),earlybreak_with_rs(model, fixed_model))
    elif METHOD == 'NAIVEHDD':
        results_dict[model_name] = max(naivehdd(fixed_model, model),naivehdd(model, fixed_model))
    elif METHOD == 'KDTREE':
        results_dict[model_name] = max(kdtree_query(fixed_model, model),kdtree_query(model, fixed_model))

def print_opening(world_size, models_count, fixed_model_name, alg):
    print_flushed("==================================")
    print_flushed(f"          WORLD SIZE: {world_size}          ")
    print_flushed("==================================")
    if alg == "DLB":
        print_flushed(f"Chosen method: Dynamic Load Balancing — {METHOD}")
    elif alg == "DS":
        print_flushed(f"Chosen method: Static Distribution — {METHOD}")
    print_flushed(f"Total model count: {models_count}")
    print_flushed(f"Fixed model: {fixed_model_name}.")
    print_flushed("-----------------------------------------------")

def print_launch():
    print("* Program should be launched as mpiexec -n <procs> python -m mpi4py main.py and with 3 parameters:\n\
    -- 1st parameter must be either S or D. S for static loading and D for dynamic loading.\n\
    -- 2nd parameter must be the path to the directory with STL or OFF models.\n\
    -- 3rd parameter must be the filename of the model, distance of which to other models will be calculated.'\
** For Example:\n\
    $> mpiexec -n 12 python -m mpi4py main.py D C:models/ModelSet model_1.stl\n\
    $> mpiexec -n 4 python -m mpi4py main.py S D:DataSet/Models airplane_0627.off\n",file=sys.stdout,flush=True)

def print_model_not_exists(directory, model):
    print(f"File {model} does not exist in {directory}",file=sys.stdout,flush=True)