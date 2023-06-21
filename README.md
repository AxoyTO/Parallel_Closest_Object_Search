# Parallel Hausdorff Distance
*A better readme will be here soon!*

* pip install -r requirements.txt

* Program should be launched as mpiexec -n <procs> python -m mpi4py main.py and with 3 parameters:__
    -- 1st parameter must be either S or D. S for static loading and D for dynamic loading.__
    -- 2nd parameter must be the path to the directory with STL or OFF models.__
    -- 3rd parameter must be the filename of the model, distance of which to other models will be calculated.'__
    ** For Example:__
    $> mpiexec -n 12 python -m mpi4py main.py D C:models/ModelSet model_1.stl__
    $> mpiexec -n 4 python -m mpi4py main.py S D:DataSet/Models airplane_0627.off
