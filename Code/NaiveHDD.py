import bpy
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance

# from mpi4py import MPI

# Удалять объект
'''
def deleteObject(name):
    obj = bpy.context.scene.objects.get(name)
    if not obj:
        return
    object_to_delete = bpy.data.objects[name]
    bpy.data.objects.remove(object_to_delete, do_unlink=True)


# Удалять все сеточные объекты MESH
def deleteAllObjects():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            deleteObject(obj.name)


def changeLocation(name, x, y, z):
    obj = bpy.context.scene.objects[name]
    obj.location.x = x
    obj.location.y = y
    obj.location.z = z
'''

'''# Создаю 2 кубы 'Cube1' и 'Cube2'
for i in (1, 2):
    bpy.ops.mesh.primitive_cube_add()
    obj_name = 'Cube' + str(i)
    bpy.context.active_object.name = obj_name

# Удаляю все объекты которые создаются при повторении и лишние
deleteObject('Cube1.001')
deleteObject('Cube2.001')
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        print(obj.name)
changeLocation('Cube1', 20, 0, 0)
changeLocation('Cube2', 0, 20, 0)'''

# Вершины куба

cube1 = np.array([[-1, -1, -1],
                  [1, -1, -1],
                  [1, 1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1],
                  [1, -1, 1],
                  [1, 1, 1],
                  [-1, 1, 1]])
# Грани
edges = []

cube2 = np.zeros((8, 3))

# Поверхности
faces = np.array([
    [0, 3, 1],
    [1, 3, 2],
    [0, 4, 7],
    [0, 7, 3],
    [4, 5, 6],
    [4, 6, 7],
    [5, 1, 2],
    [5, 2, 6],
    [2, 3, 6],
    [3, 7, 6],
    [0, 1, 5],
    [0, 5, 4]])

'''
deleteAllObjects()
'''


class ndarray_pydata(np.ndarray):
    def __bool__(self) -> bool:
        return len(self) > 0


'''print(faces.view(ndarray_pydata))
print(faces)'''
'''
cube_mesh1 = bpy.data.meshes.new("CubeMesh")
cube_mesh1.from_pydata(cube1, edges, faces.view(ndarray_pydata))
cube_mesh1.validate()

cube_mesh2 = bpy.data.meshes.new("CubeMesh")
cube_mesh2.from_pydata(cube2, edges, faces.view(ndarray_pydata))
cube_mesh2.validate()

cube_object = bpy.data.objects.new('Cube1', cube_mesh1)
# bpy.context.scene.collection.objects.link(cube_object)
view_layer = bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(cube_object)

cube_object = bpy.data.objects.new('Cube2', cube_mesh2)
view_layer = bpy.context.view_layer
view_layer.active_layer_collection.collection.objects.link(cube_object)


deleteObject('Cube1.001')
deleteObject('Cube2.001')
changeLocation('Cube1', 20, 0, 0)
changeLocation('Cube2', 0, 20, 0)
'''


# NaiveHDD
def HausdorffDistance(model_a, model_b):
    distances_list = []  # List чтобы хранить
    for i in range(len(model_a)):  # Итерация по первой модели
        dist_min = 10000000000  # Просто инициализация большим числом
        for j in range(len(model_b)):  # Итерация по второй модели
            dist = distance.euclidean(model_a[i], model_b[j])
            if dist_min > dist:
                dist_min = dist
        distances_list.append(dist_min)  # Сохранить полученную dist_min
        # Максимум среди всех точек в А минимального расстояния до B есть dH.
    return np.max(distances_list)


obj1 = np.array([(1.0, 0.0),
                 (0.0, 1.0),
                 (-1.0, 0.0),
                 (0.0, -1.0)])

obj2 = np.array([(2.0, 0.0),
                 (0.0, 2.0),
                 (-2.0, 0.0),
                 (0.0, -4.0)])

# print('Hausdorff Distance (cube1, cube2): ', HausdorffDistance(cube1, cube2))
# print('Hausdorff Distance (cube2, cube1): ', HausdorffDistance(cube1, cube2))
# print('Scipy (cube1, cube2): ', directed_hausdorff(cube1, cube2)[0])
# print('Scipy (cube2, cube1): ', directed_hausdorff(cube2, cube1)[0])
# print('Scipy: ', max(directed_hausdorff(obj1, obj2)[0], directed_hausdorff(obj2, obj1)[0]))

print('Hausdorff Distance (obj1, obj2): ', HausdorffDistance(obj1, obj2))
print('Scipy (obj1, obj2): ', directed_hausdorff(obj1, obj2)[0])
print('Hausdorff Distance (obj2, obj1): ', HausdorffDistance(obj2, obj1))
print('Scipy (obj2, obj1): ', directed_hausdorff(obj2, obj1)[0])
print('Hausdorff Distance: ', max(HausdorffDistance(obj1, obj2), HausdorffDistance(obj2, obj1)))
print('Scipy: ', max(directed_hausdorff(obj1, obj2)[0], directed_hausdorff(obj2, obj1)[0]))

'''
TODO:
    1) Улучшить реализацию алгоритма NaiveHDD -> EarlyBreak.
    2) Выучить как перевести модель в матрицу и попробовать с разными моделями.
    3) Увеличить количество сеток в моделях и стараться ускориться с помощью параллелизации 
'''
