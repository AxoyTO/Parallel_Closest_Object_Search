import random
from utils import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

class ClosestObjectSearch:
    def __init__(self, models_dir, fixed_model_name, MPI) -> None:
        self.models_dir = models_dir
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.fixed_model_name = fixed_model_name
        self.results_dict = {}

    def start():
        raise NotImplementedError()

class DynamicLoad(ClosestObjectSearch):
    def __init__(self, models_dir, fixed_model_name, MPI) -> None:
        super().__init__(models_dir,fixed_model_name.split('.')[0], MPI)
    
    def start(self):
        if self.rank == 0:
            self.models = sorted([os.path.splitext(i)[0] for i in os.listdir(self.models_dir) 
                                  if os.path.splitext(i)[1].lower() in {".stl", ".off"}])[:10]
            
            print_opening(self.world_size, len(self.models), self.fixed_model_name, "DLB")

            if self.fixed_model_name in self.models:
                self.models.pop(self.models.index(self.fixed_model_name))

            start = MPI.Wtime()
        else:
            self.fixed_model_name = None

        self.fixed_model_name = self.comm.bcast(self.fixed_model_name, root = 0)
        self.fixed_model = load_model_by_name(self.models_dir, self.fixed_model_name, self.comm)
            
        if self.world_size > 1:
            if self.rank == 0:
                self.__control_requests()
            else:
                self.results_dict = self.__receive_model_and_calculate_distance()

            res = self.comm.gather(self.results_dict, root=0)
        else:
            res = []
            model_name = self.models
            for model_name in self.models:
                calculate_distance(self.results_dict, self.models_dir, self.fixed_model, model_name, self.comm)
            res.append(self.results_dict)
            
        if self.rank == 0:
            end = MPI.Wtime()
            for i in res:
                if len(i):
                    for k, v in i.items():
                        self.results_dict[k] = v
            if len(self.results_dict):
                print_flushed(f"Closest model to {self.fixed_model_name} is {min(self.results_dict, key=self.results_dict.get)}")
                print_flushed(f"Parallel Search Time: {end - start :.5f} seconds.")

    def __control_requests(self):
        while len(self.models) > 0:
            destination = self.comm.recv()
            model_index = self.models.index(random.choice(self.models))
            model_name = self.models.pop(model_index)
            self.comm.send(model_name, dest=destination, tag=destination)
        else:
            for i in range(1, self.world_size):
                self.comm.send(None, dest=i, tag=i)

    def __receive_model_and_calculate_distance(self):
        while True:
            model_name = self.comm.sendrecv(self.rank, dest=0, source = 0, sendtag=self.rank, recvtag=self.rank)

            if model_name is None:
                break
            
            calculate_distance(self.results_dict, self.models_dir, self.fixed_model, model_name, self.comm)

        return self.results_dict

class StaticLoad(ClosestObjectSearch):
    def __init__(self, models_dir, fixed_model_name, MPI) -> None:
        super().__init__(models_dir, fixed_model_name.split('.')[0], MPI)

    def start(self):
        if self.rank == 0:
            models = [os.path.splitext(i)[0] for i in os.listdir(self.models_dir) if os.path.splitext(i)[1].lower() in {".stl", ".off"}][:5]
            print_opening(self.world_size, len(models), self.fixed_model_name, "DS")
            
            if self.fixed_model_name in models:
                models.pop(models.index(self.fixed_model_name))

            splits = np.array(models, dtype=object)
            splits = np.array_split(splits, (self.world_size))
            models = splits[0]
        
            start = MPI.Wtime()
            for i in range(1, self.world_size):
                self.comm.send(self.fixed_model_name, dest=i, tag=1)
                self.comm.send(splits[i], dest=i, tag=0)

        else:
            self.fixed_model_name = self.comm.recv(source=0, tag=1)
            models = self.comm.recv(source=0, tag=0)

        fixed_model = load_model_by_name(self.models_dir, self.fixed_model_name, self.comm)
        
        for model in models:
            calculate_distance(self.results_dict, self.models_dir, fixed_model, model, self.comm)

        if self.world_size > 1:
            res = self.comm.gather(self.results_dict, root=0)
        else:
            res = []
            res.append(self.results_dict)

        if self.rank == 0:
            end = MPI.Wtime()
            for i in res:
                if len(i):
                    for k, v in i.items():
                        self.results_dict[k] = v
            if len(self.results_dict):            
                print_flushed(
                    f"Closest model to {self.fixed_model_name} is {min(self.results_dict, key=self.results_dict.get)}"
                )
                print_flushed(f"Parallel Search Time: {end - start :.5f} seconds.")


if __name__ == "__main__":
    MPI.Init()
    if len(sys.argv) > 1:
        dynamic = ['d', 'D']
        static = ['s', 'S']

        if not (sys.argv[1] in dynamic or sys.argv[1] in static):
            if MPI.COMM_WORLD.Get_rank() == 0:
                print_launch()
            
        else:
            if len(sys.argv) > 3:
                if sys.argv[3] in os.listdir(f'{sys.argv[2]}'):
                    if sys.argv[1] in dynamic:
                        app = DynamicLoad(sys.argv[2], sys.argv[3], MPI)
                    elif sys.argv[1] in static:
                        app = StaticLoad(sys.argv[2], sys.argv[3], MPI)
                    app.start()
                else:
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print_model_not_exists(sys.argv[2], sys.argv[3])
            else:
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print_launch()
    else:
        if MPI.COMM_WORLD.Get_rank() == 0:
                print_launch()

    MPI.Finalize()