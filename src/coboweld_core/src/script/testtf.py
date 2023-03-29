#!/usr/bin/env python3
import roslib
import rospy
import tf
from scipy.spatial.transform import Rotation as R

rospy.init_node('testtf')

listener = tf.TransformListener()

rate = rospy.Rate(10.0)
listener.waitForTransform("/base", "/d435_depth_optical_frame", rospy.Time(), rospy.Duration(4.0))
# while not rospy.is_shutdown():
try:
  now = rospy.Time.now()
  listener.waitForTransform("/base", "/d435_depth_optical_frame", now, rospy.Duration(4.0))
  (trans, rot) = listener.lookupTransform("/base", "/d435_depth_optical_frame", now)
except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
  print('Error!')

r = R.from_quat(rot)
print('rotation: ', r.as_matrix)
print('quaternion: ', rot)
print('translation: ',trans)

