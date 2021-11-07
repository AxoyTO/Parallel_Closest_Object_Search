import numpy as np
import time
from stl import mesh
from mpi4py import MPI
from math import sqrt


def print_info(model):
    print(f"Vertices: {len(get_vertices(model))}")
    print(f"Faces: {len(model.vectors)}")
    print(f"Shape: {np.shape(model.vectors)}")
    print(f"Normals: {len(model.normals)}")
    print("----------------------------")


def get_vertices(model):
    return np.around(np.unique(model.vectors.reshape([int(model.vectors.size / 3), 3]), axis=0), 2)


def euclidean(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i] - array_y[i]) ** 2
    return sqrt(ret)


def EARLYBREAK(model_a, model_b, distance_function):
    nA = model_a.shape[0]
    nB = model_b.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = distance_function(model_a[i], model_b[j])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmax < cmin < np.inf:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = distance_function(model_a[i], model_b[j])
            if d < cmin:
                cmin = d
            if cmin < cmax:
                break
        if cmax < cmin < np.inf:
            cmax = cmin
    return cmax


def NaiveHDD(model_a, model_b, distance_function):
    distances_list = []  # List чтобы хранить
    for i in range(len(model_a)):  # Итерация по первой модели
        dist_min = np.inf  # Просто инициализация большим числом
        for j in range(len(model_b)):  # Итерация по второй модели
            dist = distance_function(model_a[i], model_b[j])
            if dist_min > dist:
                dist_min = dist
        distances_list.append(dist_min)  # Сохранить полученную dist_min
        # Максимум среди всех точек в А минимального расстояния до B есть dH.
    return np.max(distances_list)


if __name__ == "__main__":
    tank1 = mesh.Mesh.from_file('C:\\Users\\toaxo\\Desktop\\Sem5\\Hausdorff\\Models\\tank.stl')
    tank2 = mesh.Mesh.from_file('C:\\Users\\toaxo\\Desktop\\Sem5\\Hausdorff\\Models\\tank2.stl')

    tank1_vertices = get_vertices(tank1)
    tank2_vertices = get_vertices(tank2)
    print_info(tank1)
    print_info(tank2)

    start = time.time()
    NaiveHDD_result = NaiveHDD(tank1_vertices[0:1000], tank2_vertices[0:1000], euclidean)
    end = time.time()
    NaiveHDD_time = end - start
    print(f"NaiveHDD Result: {NaiveHDD_result}")
    print(f"NaiveHDD Elapsed Time: {NaiveHDD_time :.12f} seconds.")
    start = time.time()
    stmpi = MPI.Wtime()
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    proc_rank = comm.Get_rank()
    EARLYBREAK_result = EARLYBREAK(tank1_vertices[0:1000], tank2_vertices[0:1000], euclidean)
    end = time.time()
    endmpi = MPI.Wtime()
    EARLYBREAK_time = end - start
    eb_mpitime = endmpi - stmpi
    print(f"EARLYBREAK Result: {EARLYBREAK_result}")
    print(f"EARLYBREAK Elapsed Time: {EARLYBREAK_time:.12f} seconds.")
    print(f"EARLYBREAK MPI Elapsed Time: {eb_mpitime:.12f} seconds.")
    speedup = NaiveHDD_time / EARLYBREAK_time
    print(f"SPEEDUP: {speedup:.2f}x")
