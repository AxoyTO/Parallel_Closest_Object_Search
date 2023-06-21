# Parallel Hausdorff Distance
https://www.researchgate.net/publication/371733465_Parallelnyj_algoritm_poiska_naibolee_blizkogo_obekta_v_kollekcii_poligonalnyh_modelej
*A better readme will be here soon!*

* pip install -r requirements.txt

* Program should be launched as mpiexec -n <procs> python -m mpi4py main.py and with 3 parameters:  
    -- 1st parameter must be either S or D. S for static loading and D for dynamic loading.  
    -- 2nd parameter must be the path to the directory with STL or OFF models.  
    -- 3rd parameter must be the filename of the model, distance of which to other models will be calculated.'  
**For Example:**  
    $> mpiexec -n 12 python -m mpi4py main.py D C:models/ModelSet model_1.stl  
    $> mpiexec -n 4 python -m mpi4py main.py S D:DataSet/Models airplane_0627.off
