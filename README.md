# Parallel Closest Obj Search
https://www.researchgate.net/publication/371733465_Parallelnyj_algoritm_poiska_naibolee_blizkogo_obekta_v_kollekcii_poligonalnyh_modelej  
*A better readme will be here soon!*

> _In this paper, we consider the problem of finding an element of a collection of polygonal models that is closest to a given object using parallel computing. Options for parallelizing the solution of this problem are proposed, with static and dynamic load balancing, which made it possible to significantly speed up the search process. The developed algorithm was implemented in the Python programming language using the MPI for Python library, which supports parallel computing, and tested on a collection of several thousand models on a local machine and on the Polus high-performance computing system. The results of computational experiments have shown that the developed algorithm significantly outperforms the sequential search for the closest object in terms of computation speed. The algorithm proposed in this paper can be used in various fields, such as computer graphics, computer vision, robotics, and others._

* pip install -r requirements.txt

* Program should be launched as mpiexec -n <procs> python -m mpi4py main.py and with 3 parameters:  
    -- 1st parameter must be either S or D. S for static loading and D for dynamic loading.  
    -- 2nd parameter must be the path to the directory with STL or OFF models.  
    -- 3rd parameter must be the filename of the model, distance of which to other models will be calculated.'  
**For Example:**  
    $> mpiexec -n 12 python -m mpi4py main.py D C:models/ModelSet model_1.stl  
    $> mpiexec -n 1000 python -m mpi4py main.py S D:DataSet/Models airplane_0627.off
