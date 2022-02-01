# examples/Python/Basic/working_with_numpy.py
import random
import os
import json
# from scipy.spatial import Delaunay

from utils.utils import *

if __name__ == "__main__":

    with open("task.json", "r") as read_file:
        task = json.load(read_file)
    print(task)
    root = "C:/Users/user/pt_cloud_data/"
    e57item = task['e57item']
    e57iter = task['e57iter']




    print (e57iter)
    # points = read_ply_point(path_ply)
    # pre-processing
    pcd = o3d.geometry.PointCloud()

    points = e57_pre_processing(fnames=e57iter, voxel_size=0.1, nb_neighbors=30, std_ratio=0.5,
                                radius=1.0)
    # points = remove_nan(points)
    # points = down_sample(points,voxel_size=0.1)
    # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
    # points = estimate_pt_normals(points)
    # draw_point_cloud(points)
    pcdpt = numpy_to_pcd(points)
    write_pcd_ply(root+'step1poly.ply', pcdpt)
    results = detect_multi_planes(points, min_ratio=0.25, threshold=0.1, iterations=2000)

    planes = []
    colors = []
    index = []
    for _, plane in results:

        r = random.random()
        g = random.random()
        b = random.random()

        color = np.zeros((plane.shape[0], plane.shape[1]))
        color[:, 0] = r
        color[:, 1] = g
        color[:, 2] = b

        planes.append(plane)
        colors.append(color)

    # ply_mesh_processing(planes, colors)
    # o3d.io.write_point_cloud(f'{root }meshes.ply', pcd)
    # o3d.visualization.draw_geometries(meshlist)
    print('run Poisson surface reconstruction')
    meshlist = ply_mesh_processing(planes, colors)
    #meshlist = alpf_mesh_proc(planes, colors)
    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)

    # sorting = get_clusters(planes, index)

    # draw_result(planes, colors)
    write_result(f'{root}planes_colors.ply', planes, colors)

    print('run Poisson surface reconstruction')

    ppcd = o3d.geometry.PointCloud()
    ppcd.points = o3d.utility.Vector3dVector(planes)
    ppcd.colors = o3d.utility.Vector3dVector(colors)
    meshlist.append(ppcd)

    o3d.visualization.draw_geometries(meshlist, mesh_show_back_face=True)

    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
