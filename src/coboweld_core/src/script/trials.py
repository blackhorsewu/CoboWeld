import open3d as o3d
patch = o3d.io.read_point_cloud('patch.pcd')
def show(pcd):
  o3d.visualization.draw_geometries([pcd],
    window_name='patch',
    point_show_normal=True)

def normal(search_radius, max_pts):
  patch.estimate_normals(
    search_param = o3d.geometry.KDTreeSearchParamHybrid(
      radius = search_radius,
      max_nn = max_pts
    ))
  patch.normalize_normals()
  patch.orient_normals_towards_camera_location(camera_location = [0.0, 0.0, 0.0])

