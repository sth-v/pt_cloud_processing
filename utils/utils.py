import numpy as np
import open3d as o3d
import pye57
import operator
from itertools import*
# imports/exports


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


def read_e57_point(fname):
    """read e57 to numpy **kwargs dict
    Args:
        fname (str): path to .e57 file
    Returns:
        [dict]: **kwargs
    """
    print("\033[92mprocessing... \033[0;37;40m {}".format(fname))
    e57 = pye57.E57(fname)
    dt = e57.read_scan(0, ignore_missing_fields=True)

    assert isinstance(dt["cartesianX"], np.ndarray)
    assert isinstance(dt["cartesianY"], np.ndarray)
    assert isinstance(dt["cartesianZ"], np.ndarray)
    return dt


def read_e57_multiply(fnames,voxel_size, nb_neighbors, std_ratio):
    """read e57 to numpy **kwargs dict
    Args:
        fname list[str]: multy paths to .e57 file
    Returns:
        [ndarray]
    """

    e57_data = []
    for a in fnames:
        
        dt = read_e57_point(a)
        
        points = e57_to_numpy(dt)
        points = remove_nan(points)
        points = down_sample(points, voxel_size)
        points=  remove_noise(points, nb_neighbors, std_ratio)
        e57_data.append(points)
            

    res = np.concatenate(e57_data, axis=0)
    

    return res


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

def e57_pre_processing(fnames, voxel_size=0.1, nb_neighbors=20, std_ratio=0.5, radius=0.1):
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

    points = read_e57_multiply(fnames, voxel_size, nb_neighbors, std_ratio)
    points = down_sample(points, voxel_size)
    points = remove_noise(points, nb_neighbors, std_ratio)
    points = estimate_pt_normals(points, radius, max_nn=nb_neighbors)
    return points


def meshing_analys(pcd, k=3):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = k * avg_dist
    return radius


def mesh_clean(mesh, quadric_decimation=100000):
    dec_mesh = mesh.simplify_quadric_decimation(100000)
    dec_mesh.remove_degenerate_triangles()
    dec_mesh.remove_duplicated_triangles()
    dec_mesh.remove_duplicated_vertices()
    dec_mesh.remove_non_manifold_edges()
    return dec_mesh


def ply_mesh_processing(pcd, radii, color, r, nn, voxel_size=0.1):
    

    #dpcd = pcd.voxel_down_sample(voxel_size)
    #dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(r, nn))
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radii, 0.6]))    
    rec_mesh.paint_uniform_color(color)
    
    return rec_mesh


def ply_f_mesh_processing(pcd, radii, color, r, nn, voxel_size=0.1, depth=6):
    

    #dpcd = pcd.voxel_down_sample(voxel_size)
    #dpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(r, nn))

    rec_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth, width=0, scale=1, linear_fit=False)
    rec_mesh.paint_uniform_color(color)
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = rec_mesh.crop(bbox)
    return p_mesh_crop
    

    
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


def alpf_mesh_proc(planes, colors):
    meshlist = []

    for i in range(len(planes)):
        plane = np.array(planes[i])
        col = colors[i][0]
        print(plane)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plane)
        
        # points = down_sample(points,voxel_size=0.1)
        # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
        # points = estimate_pt_normals(points)
        #dpcd = pcd.voxel_down_sample(voxel_size=0.1)
        pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(1.0, 9))


        alpha = 0.3
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(col)
        meshlist.append(mesh)
    return meshlist

# compython geometry base methods

def get_clusters(data, labels):
    """
    :param data: The dataset
    :param labels: The label for each point in the dataset
    :return: List[np.ndarray]: A list of arrays where the elements of each array
    are data points belonging to the label at that ind
    """
    return [data[np.where(labels == i)] for i in range(np.amax(labels) + 1)]


def map_domains(values,s_min,s_max,t_min,t_max):
    
    # Remap numbers into new numeric domain
    mapped = []
    for v in values:
        if s_max-s_min > 0:
            rv = ((v-s_min)/(s_max-s_min))*(t_max-t_min)+t_min
        else:
            rv = (t_min+t_max)/2
        mapped.append(rv)
    return mapped


def plane_to_color(plane: list[float],k=0.001):
    a = abs(plane[0])
    b = abs(plane[1])
    c = abs(plane[2])
    d = abs(plane[3]*k)
    targeta=(0.0,255.0)
    tmin, tmax = targeta
    m = map_domains([a+d, b+d, c+d], 0.0, 2.0, tmin, tmax)
    mm=[]
    for j in m:
        mm.append(int(j))
    
    m.clear()
    #print(mm)
    r = map_domains(mm, tmin, tmax, 0.0, 1.0 )
    return(mm)


def plane_to_color_(plane: list[float],k=0.001):
    a = plane[0]
    b = plane[1]
    c = plane[2]
    d = plane[3]*k
    targeta=(0.0,254.0)
    tmin, tmax = targeta
    m = map_domains([a+d, b+d, c+d], -1.0, 1.0, tmin, tmax)

    sma=m[0]+m[1]+m[2]
    smb=(254-m[0])+(254-m[1])+(254-m[2])

    if sma>smb:
        m=[m[0],m[1],254-m[2]]
    if smb>sma:
        m=[254-m[0],254-m[1],254-m[2]]
    if sma==smb:
        m=[m[0],m[1],m[2]]

    mm=[]
    for j in m:

        mm.append(int(j))
    
    m.clear()
    #print(mm)
    r = map_domains(mm, tmin, tmax, 0.0, 1.0 )
    return(mm)


def plane_to_pt(plane: list[float], yz=[(0, 0), (0, 1), (1, 0)]):
    a = plane[0]
    b = plane[1] 
    c = plane[2] 
    d = plane[3]
    xyz = []
    for y, z in yz:
        x = (-b*y-c*z-d)/a
        xyz.append([x,y,z])
    return xyz


def plane_to_pt_multy(plane: list[float]):
 
    a = plane[0]
    b = plane[1] 
    c = plane[2] 
    d = plane[3]
    xyz = []

    x = (-b*1-c*0-d)/a
    y = (-a*0-c*1-d)/b
    z = (-a*1-b*0-d)/c
    xyz = [(x,1,0),(0,y,1),(1,0,z)]
    return xyz


def mns(a, b, k=1.1):
    l = list(zip(a, b))
    sl = list(starmap(lambda x, y: x - y, l))
    v = abs(sl[0]) < k and abs(sl[1]) < k and abs(sl[2]) < k

    return v


def match_planes(pts):

    p = [pts[0]]
    indxs = [p.index(pts[0])]

    ptsq = pts[1:]

    while True:
        
        if len(ptsq) == 0:
            break
        pp = ptsq[0]

        p.reverse()
        for i in p:

            #print("\033[0, 37m{} {}\033".format(pp, i))

            if i == pts[0]:
                p.append(pp)

            if mns(pp, i):

                pii = p.index(i)
                indxs.append(pii)
                print("\033[1;35;40mMatch(\033[0;37;40m{}&{}\033[1;35;40m)\033".format(pp, i))
                print(":{}\033[1;35;40m{}\033".format(i, pii))
                break
        p.reverse()

        ptsq = ptsq[1:]
    set_indx = set([a for a in indxs])
    set_indx.__len__()
    li = [a for a in set_indx]
    li.sort()
    rli = range(set_indx.__len__())
    
    dli = dict(zip(li,rli))
    icc=[]
    for i in indxs:
        icc.append(dli[i])
    
    return icc


"""planes = [
    [-0.7070897228208075, 0.7071237039189138, 0.00043730674572993943, -290.92373799967623],
    [0.7073591552420904, -0.7068542257852676, -0.00035914440886667054, 284.52671035465465],
    [0.7091583588299021, 0.7049927065615538, -0.008927810296750669, -495.7036010715342]]

plane_points=[]
plane_colors=[]
for j in planes:
    plane_points.append(plane_to_pt(j))
    plane_colors.append(plane_to_color_(j))

print(plane_points)
print(plane_colors)



def get_rh_model():
    pass

"""

"""with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh])
    draw_result(planes, colors)"""


"""if __name__ == "__main__":

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
    
    o3d.visualization.draw_geometries([downpcd], point_show_normal=True)"""

'''
    # convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = o3d.geometry.Image((z_norm * 255).astype(np.uint8))
    o3d.io.write_image("../../TestData/sync.png", img)
    o3d.visualization.draw_geometries([img])'''