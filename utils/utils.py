import numpy as np
import open3d as o3d
import pye57


# imports/exports

def add_root(root, fname, is_iter: bool = False):
    if is_iter:
        fnames = []
        for f in fname:
            fnames.append(root + f)
        return fnames
    else:
        return root + fname


def read_ply_point(fname):
    """ read point from ply
    Args:
        fname (str): path to .ply file
    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = o3d.io.read_point_cloud(fname)

    return pcd_to_numpy(pcd)


def write_pcd_ply(fname, pcd):
    return o3d.io.write_point_cloud(fname, pcd)


def read_e57_point(fname, **kwargs):
    """read e57 to numpy **kwargs dict
    Args:
        fname (str): path to .e57 file
    Returns:
        [dict]: **kwargs
    """

    e57 = pye57.E57(fname)
    dt = e57.read_scan(0, **kwargs)
    assert isinstance(dt["cartesianX"], np.ndarray)
    assert isinstance(dt["cartesianY"], np.ndarray)
    assert isinstance(dt["cartesianZ"], np.ndarray)
    return dt


def read_e57_multiply(fnames, **kwargs):
    """read e57 to numpy **kwargs dict
    Args:
        fname list[str]: multy paths to .e57 file
    Returns:
        [dict]: **kwargs
    """

    e57_data = {}
    for fname in fnames:
        dt = read_e57_point(fname, **kwargs)
        e57_data |= dt
    return e57_data


def numpy_to_pcd(xyz):
    """ convert numpy ndarray to open3D point cloud 
    Args:
        xyz (ndarray): 
    Returns:
        [open3d.geometry.PointCloud]: 
    """

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    return pcd


def pcd_to_numpy(pcd):
    """  convert open3D point cloud to numpy ndarray
    Args:
        pcd (open3d.geometry.PointCloud): 
    Returns:
        [ndarray]: 
    """

    return np.asarray(pcd.points)


def e57_to_numpy(e57_data):
    """read e57 to numpy **kwargs dict
    Args:
        e57_data dict: e57 dictionary
    Returns:
        [ndarray]: x, y, z
    """

    x = np.array(e57_data["cartesianX"])
    y = np.array(e57_data["cartesianY"])
    z = np.array(e57_data["cartesianZ"])
    # print(f'e57 x: {x}')
    points = np.zeros((np.size(x), 3))
    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z
    # print(f'xyz: {points}')
    return points


# processsing

def remove_nan(points):
    """ remove nan value of point clouds
    Args:
        points (ndarray): N x 3 point clouds
    Returns:
        [ndarray]: N x 3 point clouds
    """

    return points[~np.isnan(points[:, 0])]


def remove_noise(pc, nb_neighbors=20, std_ratio=2.0):
    """ remove point clouds noise using statitical noise removal method
    Args:
        pc (ndarray): N x 3 point clouds
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.
    Returns:
        [ndarray]: N x 3 point clouds
    """

    pcd = numpy_to_pcd(pc)
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    return pcd_to_numpy(cl)


def down_sample(pts, voxel_size=0.5):
    """ down sample the point clouds
    Args:
        pts (ndarray): N x 3 input point clouds
        voxel_size (float, optional): voxel size. Defaults to 0.003.
    Returns:
        [ndarray]: 
    """

    p = numpy_to_pcd(pts).voxel_down_sample(voxel_size=voxel_size)

    return pcd_to_numpy(p)


def estimate_pt_normals(pts, radius=1.0, max_nn=30):
    # p = numpy_to_pcd(pts).voxel_down_sample(voxel_size=voxel_size)
    # p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    pcd = numpy_to_pcd(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    return pcd_to_numpy(pcd)


# analysis algorithms

def plane_regression(points, threshold=0.01, init_n=3, iterations=1000):
    """ plane regression using ransac
    Args:
        points (ndarray): N x3 point clouds
        threshold (float, optional): distance threshold. Defaults to 0.003.
        init_n (int, optional): Number of initial points to be considered inliers in each iteration
        iterations (int, optional): number of iteration. Defaults to 1000.
    Returns:
        [ndarray, List]: 4 x 1 plane equation weights, List of plane point index
    """

    pcd = numpy_to_pcd(points)

    w, index = pcd.segment_plane(
        threshold, init_n, iterations)

    return w, index


def draw_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def draw_result(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def write_result(fname, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    write_pcd_ply(fname, pcd)


# definition of the main function

def e57_pre_processing(root, fname, is_iter: bool = False, voxel_size=0.1, nb_neighbors=20, std_ratio=2.0, radius=1.0):
    """ pre processing .e57 data 
    Args:
        fname (str): 
        root (str):
        is_iter (bool): Defaults to False
        voxel_size (float, optional): voxel size. Defaults to 0.1.
        nb_neighbors (int, optional): Defaults to 20.
        std_ratio (float, optional): Defaults to 2.0.
        radius

    Returns:
        [ndarray]
    """
    fn = add_root(root, fname, is_iter=is_iter)
    if is_iter:
        dt = read_e57_multiply(fn)
    else:

        dt = read_e57_point(fn)

    points = e57_to_numpy(dt)
    points = remove_nan(points)
    points = down_sample(points, voxel_size)
    points = remove_noise(points, nb_neighbors, std_ratio)

    points = estimate_pt_normals(points, radius, max_nn=nb_neighbors)
    return points


def ply_mesh_processing(planes, colors):
    meshlist = []

    for i in range(len(planes)):
        plane = np.array(planes[i])
        col = colors[i][0]
        print(plane)
        pcd = numpy_to_pcd(plane)
        # points = down_sample(points,voxel_size=0.1)
        # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
        # points = estimate_pt_normals(points)
        dpcd = pcd.voxel_down_sample(voxel_size=0.1)
        dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(1.0, 10))
        radii = [0.1, 0.2, 0.5, 2]
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            dpcd, o3d.utility.DoubleVector(radii))
        rec_mesh.paint_uniform_color(col)
        meshlist.append(rec_mesh)
    return meshlist


def detect_multi_planes(points, min_ratio=0.05, threshold=0.01, iterations=1000):
    """ Detect multiple planes from given point clouds
    Args:
        points (np.ndarray): 
        min_ratio (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
        threshold (float, optional): RANSAC threshold in (m). Defaults to 0.01.
        iterations
    Returns:
        [List[tuple(np.ndarray, List)]]: Plane equation and plane point index
    """

    plane_list = []
    n = len(points)
    target = points.copy()
    count = 0

    while count < (1 - min_ratio) * n:
        w, index = plane_regression(
            target, threshold=threshold, init_n=3, iterations=iterations)

        count += len(index)
        plane_list.append((w, target[index]))
        target = np.delete(target, index, axis=0)

    return plane_list


# compython geometry base methods

def get_clusters(data, labels):
    """
    :param data: The dataset
    :param labels: The label for each point in the dataset
    :return: List[np.ndarray]: A list of arrays where the elements of each array
    are data points belonging to the label at that ind
    """
    return [data[np.where(labels == i)] for i in range(np.amax(labels) + 1)]


def get_rh_model():
    pass
