# examples/Python/Basic/working_with_numpy.py
import random
import os
import json
from numpy import asarray
# from scipy.spatial import Delaunay

from utils.utils import *

if __name__ == "__main__":
    root = "C:/Users/user/pt_cloud_data/"
    with open(root+"task.json", "r") as read_file:
        task = json.load(read_file)
    #print(task)
    
    e57item = task['e57item']
    e57iter = task['e57iter']

    for cloud in e57iter:
       
        id = str(e57iter.index(cloud))
        points = e57_pre_processing(fnames=[cloud], voxel_size=0.01, nb_neighbors=50, std_ratio=0.5,
                                radius=1.0)
        # points = remove_nan(points)
        # points = down_sample(points,voxel_size=0.1)
        # points = remove_noise(points, nb_neighbors=50, std_ratio=0.5)
        # points = estimate_pt_normals(points)
        # draw_point_cloud(points)

        clouds=[]
        pcdpt = numpy_to_pcd(points)
        write_pcd_ply(root+id+'step1poly.ply', pcdpt)
        results = detect_multi_planes(points, min_ratio=0.05, threshold=0.01, iterations=2000)
        planes = []
        colors = []
        colors_ = []
        color_marks = []
        color_marks_ = []
        index = []
        for _, plane in results:
            r = plane_to_color(_)
            #r_ = plane_to_color_(_)
            #clrs_ = map_domains(r_, 0.0, 254.0, 0.0, 1.0 )
            clrs = map_domains(r, 0.0, 254.0, 0.0, 1.0 )
            color = np.zeros((plane.shape[0], plane.shape[1]))
            #color_ = np.zeros((plane.shape[0], plane.shape[1]))
            color[:, 0] = clrs[0]
            color[:, 1] = clrs[1]
            color[:, 2] = clrs[2]
            #color_[:, 0] = clrs_[0]
            #color_[:, 1] = clrs_[1]
            #color_[:, 2] = clrs_[2]
            index.append(list(_))
            colors.append(color)

            color_marks.append(r)
            clouds.append(plane)
            #colors_.append(color_)
            
    
    cloudss=np.asarray(clouds, dtype=object)
    print(f'marks = {np.asarray(color_marks)}')

    lables= match_planes(color_marks)
    print(lables)
    mxx=lables.copy()

    print("\033[95mlables are calculated\033 ")

    mxx.sort(reverse=True)
    mx = mxx[0]
    lbnum=np.array(lables)

    concate_col=[]
    concate_clo=[]
    lablslice=[]
    cloudslice=[]
    colorslice=[]

    for i in range(mx):
        searchval=i
        ii = np.where(lbnum == searchval)[0]
        ii = ii.tolist()
        clo = [clouds[j] for j in ii]
        col = [colors[j] for j in ii]
        cloudslice.append(clo)
        colorslice.append(col)
        lablslice.append(ii)
        cloconc = np.concatenate(clo, axis=0)
        colconc = np.concatenate(col, axis=0)
        concate_clo.append(cloconc)
        concate_col.append(colconc)
    print("\033[95mlables slice \033[0;37;40m{}".format(lablslice))
    print("\033[92mmapping done\033 ")
    print("\033[0;37;40mshow clouds, colors? y/n\033")
    
    #draw_result(points=concate_clo[0], colors=concate_col[0])
    pcdlist=[]
    meshlist=[]
    for clo in concate_clo:
        i=concate_clo.index(clo)
        col = concate_col[i]


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(clo)
        pcd.colors = o3d.utility.Vector3dVector(col)
        print('voxelisation...')
        pcd.voxel_down_sample(voxel_size=0.02)
        print('denoise...')
        pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.5)
        print('normals...')
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 50))
        print("\033[92mclouds dump...\033 ")
        print("\033[0;37;40m.../clouds/{}_planes_colors.ply'\033".format(i))
        o3d.io.write_point_cloud(f'{root}/clouds/{i}_planes_colors.ply', pcd)

        pcdlist.append(pcd)
        
        print("\033[33mmeshing ...\033 ")
        print("\033[0;37;40mparameters solving...".format(i))
        rad = meshing_analys(pcd, k=3)
        print(rad)
        print("\033[0;37;40mrecommend radii: \033[91m{}\033[0;37;40m, use: 0.06".format(rad))

        print("\033[33mmesh processing...\033[0;37;40m ")
        colr=col[0]
        print(colr)
        mesh = ply_mesh_processing(pcd, radii = 0.04, color=colr, r=0.2, nn=50, voxel_size=0.02)
        #mesh = ply_f_mesh_processing(pcd, radii = 0.05, color=colr, r=0.2, nn=50, voxel_size=0.02, depth=10)
        print("\033[33mmesh done\033")
        print("\033[0;37;40mcleaning...\033")
        meshd = mesh_clean(mesh, quadric_decimation=1000000)
        meshlist.append(mesh)
        print("\033[33mclouds dump...\033 ")
        print("\033[0;37;40m.../meshes/{}_meshes.ply\033".format(i))
        o3d.io.write_triangle_mesh(f'{root}/meshes/{i}_meshes.ply', meshd)
        print("\033[92msucsess!\033 ")
    
        """
        a = input()
        a='n'

        if a=='n':
            pass
        elif a=='':
            print("033[92mclouds:033[0;37;40m {}033".format(np.asarray(concate_clo)))
        else:
            pass"""

    o3d.visualization.draw_geometries(pcdlist, point_show_normal=True)
