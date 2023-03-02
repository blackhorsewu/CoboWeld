'''
Try to visualize normals of a point cloud using open3d
'''
import open3d as o3d

pcd = o3d.io.read_point_cloud('50x50new.pcd')

pcd.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(
    radius = 0.05, max_nn = 100
    )
)
pcd.normalize_normals()
pcd.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

o3d.visualization.draw_geometries([pcd], point_show_normal=True)