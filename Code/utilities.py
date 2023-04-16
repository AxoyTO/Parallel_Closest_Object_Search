import os
from hausdorff import *
from data import *
import trimesh

def load_model_by_name(model_name, comm):
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

def calculate_distance(fixed_model, model_name, comm):
        model = load_model_by_name(model_name, comm)
        if METHOD == 'SCIPY_DH':
            results_dict[model_name], _, _ = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))
        elif METHOD == 'EARLYBREAK':
            results_dict[model_name] = max(earlybreak(fixed_model, model),earlybreak(model, fixed_model))
        elif METHOD == 'EB_RS':
            results_dict[model_name] = max(earlybreak_with_rs(fixed_model, model),earlybreak_with_rs(model, fixed_model))
        elif METHOD == 'NAIVEHDD':
            results_dict[model_name] = max(naivehdd(fixed_model, model),naivehdd(model, fixed_model))
        elif METHOD == 'KDTREE':
            results_dict[model_name] = max(kdtree_query(fixed_model, model),kdtree_query(model, fixed_model))
