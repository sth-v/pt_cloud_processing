# examples/Python/Basic/working_with_numpy.py
import random
import os
import json
# from scipy.spatial import Delaunay

from utils.utils import *

if __name__ == "__main__":
    root = "C:/Users/user/pt_cloud_data/"
    with open(root+"task.json", "r") as read_file:
        task = json.load(read_file)
    print(task)
    
    e57item = task['e57item']
    e57iter = task['e57iter']


    for cloud in e57item:
        print(cloud)
        id = str(e57item.index(cloud))
        points = e57_pre_processing(fnames=[cloud], voxel_size=0.02, nb_neighbors=50, std_ratio=0.5,
                                radius=1.0)
        # points = remove_nan(points)
        # points = down_sample(points,voxel_size=0.1)
        # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
        # points = estimate_pt_normals(points)
        # draw_point_cloud(points)


        pcdpt = numpy_to_pcd(points)
        write_pcd_ply(root+id+'step1poly.ply', pcdpt)
        results = detect_multi_planes(points, min_ratio=0.05, threshold=0.05, iterations=2000)
        planes = []
        colors = []
        colors_ = []
        index = []
        for _, plane in results:

            #r = random.random()
            #g = random.random()
            #b = random.random()
            clrs = plane_to_color(_)
            clrs_ = plane_to_color_(_)
            color = np.zeros((plane.shape[0], plane.shape[1]))
            color_ = np.zeros((plane.shape[0], plane.shape[1]))
            color[:, 0] = clrs[0]
            color[:, 1] = clrs[1]
            color[:, 2] = clrs[2]
            color_[:, 0] = clrs_[0]
            color_[:, 1] = clrs_[1]
            color_[:, 2] = clrs_[2]
            index.append(list(_))
            planes.append(plane)
            colors.append(color)
            colors_.append(color_)

        print(f'dumping planes to {id}index.json')
        with open(root+id+"index.json", "w") as f:
          json.dump(index, f)
        
        #print('meshing...')
        #meshlist = ply_mesh_processing(planes, colors)
        #for mesh in meshlist:
            #im = meshlist.index(mesh)
            #o3d.io.write_triangle_mesh(f'{root}{id}_{im}meshes.ply', mesh)
        #meshlist = alpf_mesh_proc(planes, colors)
        planes = np.concatenate(planes, axis=0)
        colors = np.concatenate(colors, axis=0)
        colors_ = np.concatenate(colors_, axis=0)
        print('write planes...')
        write_result(f'{root}{id}planes_colors.ply', planes, colors)
        write_result(f'{root}{id}planes_colors.ply', planes, colors_)
   
    # points = read_ply_point(path_ply)
    # pre-processing
"""    pcd = o3d.geometry.PointCloud()

    points = e57_pre_processing(fnames=e57iter, voxel_size=0.02, nb_neighbors=50, std_ratio=0.5,
                                radius=1.0)
    # points = remove_nan(points)
    # points = down_sample(points,voxel_size=0.1)
    # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
    # points = estimate_pt_normals(points)
    # draw_point_cloud(points)
    pcdpt = numpy_to_pcd(points)
    write_pcd_ply(root+'step1poly.ply', pcdpt)
    results = detect_multi_planes(points, min_ratio=0.05, threshold=0.05, iterations=2000)

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
        index.append(list(_))
        planes.append(plane)
        colors.append(color)

    with open(root+"index.json", "w") as f:
        json.dump(index, f)
    # ply_mesh_processing(planes, colors)
    # o3d.io.write_point_cloud(f'{root }meshes.ply', pcd)
    # o3d.visualization.draw_geometries(meshlist)
    print('meshing...')
    meshlist = ply_mesh_processing(planes, colors)
    write_result(f'{root}planes_colors.ply', planes, colors)
    #meshlist = alpf_mesh_proc(planes, colors)
    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)

    # sorting = get_clusters(planes, index)

    # draw_result(planes, colors)
    write_result(f'{root}planes_colors.ply', planes, colors)

    print('vis')

    ppcd = o3d.geometry.PointCloud()
    ppcd.points = o3d.utility.Vector3dVector(planes)
    ppcd.colors = o3d.utility.Vector3dVector(colors)
    meshlist.append(ppcd)

    o3d.visualization.draw_geometries(meshlist, mesh_show_back_face=True)"""

    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
