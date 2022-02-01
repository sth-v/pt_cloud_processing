# examples/Python/Basic/working_with_numpy.py
import random
import json
# from scipy.spatial import Delaunay

from utils.utils import *

if __name__ == "__main__":

    with open("task.json", "r") as read_file:
        task = json.load(read_file)

    root = task['path']
    e57item = task['e57item']
    e57iter = task['e57iter']

    # points = read_ply_point(path_ply)
    # pre-processing
    pcd = o3d.geometry.PointCloud()

    points = e57_pre_processing(root, fname=e57iter, is_iter=True, voxel_size=0.1, nb_neighbors=20, std_ratio=0.5,
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

    planes = np.concatenate(planes, axis=0)
    colors = np.concatenate(colors, axis=0)

    # sorting = get_clusters(planes, index)

    draw_result(planes, colors)
    write_result(f'{root }meshes.ply', planes, colors)

    """print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh])
    draw_result(planes, colors)"""

"""
if __name__ == "__main__":

    data = {}
    point_filee = ["D:/testptcld/atom-005.e57", "D:/testptcld/atom-006.e57"]
    for pf in point_filee:
        e57 = pye57.E57(pf)

        dt = e57.read_scan(0, ignore_missing_fields=True)
        assert isinstance(dt["cartesianX"], np.ndarray)
        assert isinstance(dt["cartesianY"], np.ndarray)
        assert isinstance(dt["cartesianZ"], np.ndarray)
        data |= dt

    x = np.array(data["cartesianX"])
    y = np.array(data["cartesianY"])
    z = np.array(data["cartesianZ"])
    print('xy', x, y)
    point_file = "D:/testptcld/"
    # generate some neat n times 3 matrix using a variant of sync function
    #
    print(x)
    # mesh_x, mesh_y = np.meshgrid(x, x)

    # print('mesh',x,  mesh_x)
    # z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))

    # z_norm = (z - z.min()) / (z.max() - z.min())
    # print(z_norm)
    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = x
    xyz[:, 1] = y
    xyz[:, 2] = z
    print('xyz')
    print(xyz)
    # print(np.reshape(mesh_x, -1), x)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(f'{point_file}sync.ply', pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud(f'{point_file}sync.ply')
    # o3d.visualization.draw_geometries([pcd_load])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd_load.voxel_down_sample(voxel_size=0.5)
    # o3d.visualization.draw_geometries([downpcd])
    # normal colors 9
    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    o3d.visualization.draw_geometries([downpcd], point_show_normal=True)

'''
    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("../../TestData/sync.png", img)
    o3d.visualization.draw_geometries([img])'''
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
"""