import open3d as o3d
import numpy as np
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

normal(0.025, 50)

patch_tree = o3d.geometry.KDTreeFlann(patch)

pt_number = np.asarray(patch.points).shape[0]
print("\nNumber of points in the Point Cloud: ", pt_number)

neighbour = min(pt_number//100, 50)
print("\nNumber of neighbours: ", neighbour)

feature_value_list = []
n_list = np.asarray(patch.normals)

for index in range(pt_number):
  [k, idx, _] = patch_tree.search_knn_vector_3d(patch.points[index], neighbour)
  centroid = np.mean(n_list[idx, :], axis=0)
  feature_value = np.linalg.norm(
    centroid - n_list[index, :] * np.dot(centroid, n_list[index, :])/
    np.linalg.norm(n_list[index,:])
  )
  feature_value_list.append(feature_value)
  print(feature_value)









