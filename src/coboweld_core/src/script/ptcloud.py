import open3d as o3d
import numpy as np
from pyntcloud import PyntCloud
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

pcd = PyntCloud.from_file('patch1.pcd')
k_neighbours = pcd.get_neighbors(k=30)
ev = pcd.add_scalar_field('eigen_values', k_neighbors=k_neighbours)
pcd.add_scalar_field('curvature', ev=ev)
