import numpy as np
import time
from stl import mesh
from math import sqrt
from scipy.spatial.distance import directed_hausdorff


def print_info(model):
    print(f"Vertices: {len(get_vertices(model))}")
    # print(f"Faces: {len(model.vectors)}")
    # print(f"Shape: {np.shape(model.vectors)}")
    # print(f"Normals: {len(model.normals)}")
    print("----------------------------")


def get_vertices(model):
    return np.around(np.unique(model.vectors.reshape([int(model.vectors.size / 3), 3]), axis=0), 2)


def euclidean(array_A, array_B):
    n = array_A.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_A[i] - array_B[i]) ** 2
    return sqrt(ret)


def EARLYBREAK(model_a, model_b, distance_function):
    """
    В отличие от NaiveHDD,  когда находится число меньше текущего cmax,
    продолжение прохождения оставшихся точек множества B не имеет смысла.
    Поэтому внутренний цикл заверщает работу как только находится это число.
    """
    nA = model_a.shape[0]  # Размер множества точек в А
    nB = model_b.shape[0]  # Размер множества точек в B
    cmax = 0.  # Инициализация cmax = 0, cmax в итоге даст нам dH
    for i in range(nA):  # Итерация по точкам первой модели
        cmin = np.inf  # Инициализация cmin большим числом
        for j in range(nB):  # Итерация по точкам второй модели
            d = distance_function(model_a[i], model_b[j])  # Вычислить расстояние между точками
            if d < cmin:
                cmin = d
            if cmin < cmax:  # Когда находится число меньше текущего cmax -> break
                break
        if cmax < cmin:  # Если полученный cmin больше чем cmax, то cmax(
            cmax = cmin
    return cmax


def NaiveHDD(model_a, model_b, distance_function):
    """
    Внешний цикл проходит все точки в множестве A, пока внутренний цикл проходит все точки в множестве B.
    """
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
    # print(np.array([1]) / 0.) # -> infinite

    start = time.time()
    NaiveHDD_result = NaiveHDD(tank1_vertices[0:1000], tank2_vertices[0:1000], euclidean)
    end = time.time()
    NaiveHDD_time = end - start
    print(f"NaiveHDD Result: {NaiveHDD_result}")
    print(f"NaiveHDD Elapsed Time: {NaiveHDD_time :.5f} seconds.")
    start = time.time()
    EARLYBREAK_result = EARLYBREAK(tank1_vertices[0:1000], tank2_vertices[0:1000], euclidean)
    end = time.time()
    EARLYBREAK_time = end - start
    print(f"EARLYBREAK Result: {EARLYBREAK_result}")
    print(f"EARLYBREAK Elapsed Time: {EARLYBREAK_time:.5f} seconds.")

    speedup = NaiveHDD_time / EARLYBREAK_time
    print(f"SPEEDUP(NaiveHDD -> EARLYBREAK): {speedup:.2f}x")

    scipy_hausdorff = directed_hausdorff(tank1_vertices[0:1000], tank2_vertices[0:1000])
    print(f"Scipy Result: {scipy_hausdorff[0]}")

'''
TODO:
    1) Улучшить реализацию алгоритма NaiveHDD -> EARLYBREAK. + 
    2) Выучить как перевести модель в матричное представление +
    3) Попробовать с моделями большого размера вершин +
    4) Улучшить реализацию алгоритма EARLYBREAK -> NOHD
    5) Реализовать параллельную версию -> CUDA, MPI
'''
